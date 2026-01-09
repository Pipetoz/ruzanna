import sys
import time
import math
import json
import torch
import pynvml
import bitsandbytes as bnb

from pathlib import Path
from datetime import datetime
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, BitsAndBytesConfig

from model_utils import TrainingMode, ValidationMode, GenerationMode
from monitor_utils import AdvancedTrainingMonitor

MAX_STEP_TIME = 30  # Максимальное время шага в секундах

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm не установлен: pip install tqdm")

HAS_PYNVML = False
try:
	import pynvml
	pynvml.nvmlInit()
	HAS_PYNVML = True
	pynvml.nvmlShutdown()
except:
	print("⚠️  pynvml не доступен, мощность GPU не отслеживается")

def get_gpu_power():
	if not HAS_PYNVML:
		return "N/A"
	try:
		pynvml.nvmlInit()
		handle = pynvml.nvmlDeviceGetHandleByIndex(0)
		power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
		pynvml.nvmlShutdown()
		return f"{power:.0f}"
	except:
		return "N/A"

print("=" * 80)
print("🧠 ОБУЧЕНИЕ ПСИХОЛОГИЧЕСКОГО БОТА - ПРОДВИНУТАЯ ВЕРСИЯ")
print("=" * 80)

# ДОБАВЬТЕ ВЫБОР РЕЖИМА
print("\n🔧 Выберите режим вывода:")
print("1. 🚀 Быстрый (только LR, Loss, прогресс)")
print("2. 🐛 Отладочный (все детали)")
print("3. 📊 Профессиональный (метрики + графики)")
debug_mode = input("Выберите (1-3, по умолчанию 1): ").strip() or "1"
DEBUG_MODE = int(debug_mode)

print("=" * 80)
print("🧠 ОБУЧЕНИЕ ПСИХОЛОГИЧЕСКОГО БОТА - ПРОДВИНУТАЯ ВЕРСИЯ")
print("   С АДАПТИВНЫМИ НАСТРОЙКАМИ И МЕТРИКАМИ")
print("=" * 80)

print("\n🔧 Выберите режим точности для обучения:")
print("1. 🚀 16-битный (FP16, смешанная точность) - быстрее, экономит память")
print("2. 🐘 32-битный (FP32, полная точность) - максимальная стабильность")
precision_choice = input("Выберите (1 или 2, по умолчанию 1): ").strip() or "1"
PRECISION_MODE = int(precision_choice)  # 1 для 16-бит, 2 для 32-бит

# Конфигурация для смешанной точности (AMP)
USE_AMP = PRECISION_MODE == 1
GRAD_SCALER = torch.cuda.amp.GradScaler(enabled=USE_AMP)

print(f"\n   ✅ Выбран режим: {'16-битный (AMP)'if USE_AMP else '32-битный'} обучения")
print(f"   • Финальная модель будет сохранена в 32-битном формате (FP32)")

# ================= ПРАВИЛЬНЫЕ ПАРАМЕТРЫ =================
BATCH_SIZE = 3 
MAX_LENGTH = 729
GRADIENT_ACCUMULATION = 6
LEARNING_RATE = 2e-4
EPOCHS = int(input("Введи количество эпох..."))
WARMUP_RATIO = 0.9

print("\n🎯 ПАРАМЕТРЫ ОБУЧЕНИЯ:")
print(f"   • Batch size: {BATCH_SIZE}")
print(f"   • Max length: {MAX_LENGTH}")
print(f"   • Gradient accumulation: {GRADIENT_ACCUMULATION}")
print(f"   • Learning rate: {LEARNING_RATE:.1e}")
print(f"   • Epochs: {EPOCHS}")
print(f"   • Warmup: {WARMUP_RATIO*100}%")

# ================= КОНТЕКСТНЫЕ МЕНЕДЖЕРЫ ДЛЯ РЕЖИМОВ =================

# ================= КОНСТАНТЫ =================
class TrainingConfig:
	"""Конфигурационные константы"""
	
	# Оптимальный шедулер
	COSINE_DECAY_RATIO = 0.6      # 60% шагов на cosine decay
	FINAL_LINEAR_START = 0.1      # Начальное значение LR в финальной фазе
	FINAL_LINEAR_DECAY = 0.5      # Скорость затухания в финальной фазе
	
	# Качество ответов
	MIN_WORDS = 5                 # Минимальное количество слов
	MAX_WORDS = 80                # Максимальное количество слов
	UNIQUE_WORDS_RATIO = 0.6      # Минимальный процент уникальных слов
	MAX_EMPATHY_WORDS = 5         # Для нормализации оценки эмпатии
	
	# Ранняя остановка
	MIN_DELTA = 0.001             # Минимальное значимое улучшение
	MAX_PATIENCE = 3              # Максимальное количество эпох без улучшений
	MAX_NAN_TOLERANCE = 3         # Максимальное количество NaN до перезагрузки
	
	# Генерация
	BASE_TEMPERATURE = 0.729      # Базовая температура генерации
	MIN_TEMPERATURE = 0.6         # Минимальная температура
	MAX_TEMPERATURE = 0.9         # Максимальная температура
	TOP_P_HIGH = 0.95             # top_p для высокой температуры
	TOP_P_LOW = 0.9               # top_p для низкой температуры
	
	# Структура
	MIN_SENTENCES = 2             # Минимальное количество предложений
	MAX_SENTENCES = 5             # Максимальное количество предложений
	MIN_WORDS_PER_SENTENCE = 5    # Минимальное количество слов в предложении
	MAX_WORDS_PER_SENTENCE = 20   # Максимальное количество слов в предложении

# ================= ОПТИМАЛЬНЫЙ ШЕДУЛЕР =================

class OptimalScheduler:
	"""
	Оптимальный шедулер для психологической модели
	Warmup → Cosine Decay → Linear Final
	"""
	
	def __init__(self, optimizer, total_steps, initial_lr, warmup_ratio):
		self.optimizer = optimizer
		self.total_steps = total_steps
		self.initial_lr = initial_lr
		self.warmup_steps = int(total_steps * warmup_ratio)
		self.cosine_steps = int(total_steps * 0.6)
		self.linear_steps = total_steps - self.warmup_steps - self.cosine_steps
		self.current_step = 0
		self.cosine_steps = int(total_steps * TrainingConfig.COSINE_DECAY_RATIO)
		
		print("\n 🎯 ОПТИМАЛЬНЫЙ ШЕДУЛЕР (3 фазы):")
		print(f"   • Всего шагов: {total_steps}")
		print(f"   • Warmup: {self.warmup_steps} шагов ({warmup_ratio*100}%)")
		print(f"   • Cosine decay: {self.cosine_steps} шагов ({TrainingConfig.COSINE_DECAY_RATIO*100}%)")
		print(f"   • Linear final: {self.linear_steps} шагов (остальное)")
	
	def step(self):
		"""Выполняет один шаг шедулера"""
		self.current_step += 1
		
		if self.current_step <= self.warmup_steps:
			# 1. Warmup: линейный рост
			lr = self.initial_lr * (self.current_step / self.warmup_steps)
			phase = "WARMUP"
			
		elif self.current_step <= self.warmup_steps + self.cosine_steps:
			# 2. Cosine decay
			progress = (self.current_step - self.warmup_steps) / self.cosine_steps
			lr = self.initial_lr * 0.5 * (1 + math.cos(math.pi * progress))
			phase = "COSINE"
			
		else:
			# 3. Линейное финальное падение
			progress = (self.current_step - self.warmup_steps - self.cosine_steps) / self.linear_steps
			lr = self.initial_lr * TrainingConfig.FINAL_LINEAR_START * (1 - progress * TrainingConfig.FINAL_LINEAR_DECAY)
			phase = "FINAL"
		
		# Устанавливаем LR для всех групп параметров
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr
		
		return lr, phase
# ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =================

def check_scaler_health(scaler, context="шаг"):
	"""Проверка состояния scaler для AMP"""
	if not scaler._enabled:
		return True
	
	try:
		scale = scaler.get_scale()
		
		# Критерии неконсистентности
		if scale <= 0:
			print(f"   ❌ {context}: scale <= 0 ({scale:.2e})")
			return False
		if math.isnan(scale):
			print(f"   ❌ {context}: scale = NaN")
			return False
		if math.isinf(scale):
			print(f"   ❌ {context}: scale = бесконечность")
			return False
		if scale > 1e6:  # Слишком большой
			print(f"   ⚠️  {context}: scale слишком большой ({scale:.2e})")
			return False
		if scale < 1e-6:  # Слишком маленький
			print(f"   ⚠️  {context}: scale слишком маленький ({scale:.2e})")
			return False
			
		return True
		
	except Exception as e:
		print(f"   ❌ {context}: ошибка проверки scaler: {e}")
		return False

def monitor_scaler_state(step, scaler, prefix=""):
	"""Мониторинг состояния scaler для отладки"""
	if not scaler._enabled:
		return
	
	scale = scaler.get_scale()
	growth_factor = scaler._growth_factor
	backoff_factor = scaler._backoff_factor
	growth_interval = scaler._growth_interval
	
	print(f"{prefix} Шаг {step}:")
	print(f"   • Scale: {scale:.4e}")
	print(f"   • Growth factor: {growth_factor}")
	print(f"   • Backoff factor: {backoff_factor}")
	print(f"   • Growth interval: {growth_interval}")
	
	# Проверка на неконсистентное состояние
	if scale > 1e6 or scale < 1e-6:
		print(f"   ⚠️  ПРЕДУПРЕЖДЕНИЕ: Неконсистентный scale!")
		return False
	
	return True

# Использование в основном цикле:
if global_step % 50 == 0 and USE_AMP:
	monitor_scaler_state(global_step, GRAD_SCALER, "   🎛️ ")

def handle_nan_loss(loss_value, step_info):
	"""
	Обработка NaN loss с правильным управлением состоянием scaler
	
	Args:
		loss_value: значение loss (может быть NaN)
		step_info: информация о текущем шаге (global_step, accumulation_count и т.д.)
	"""
	# Увеличиваем счётчик NaN
	step_info['nan_count'] += 1
	
	if step_info['nan_count'] >= MAX_NAN_TOLERANCE:
		# Критическое количество NaN - перезагружаем модель
		reload_checkpoint()
		
		# Полный сброс scaler
		if USE_AMP:
			step_info['scaler'] = torch.cuda.amp.GradScaler(enabled=True)
		
		step_info['nan_count'] = 0
		step_info['accumulation'] = 0
		return 'reload'
	
	# Пропуск одного батча с обновлением scaler
	step_info['optimizer'].zero_grad()
	step_info['accumulation'] = 0
	
	if USE_AMP and step_info['scaler']._scale is not None:
		# Важно: обновляем scaler даже при пропуске шага
		step_info['scaler'].update()
	
	return 'skip'

class TrainingState:
	"""Класс для управления состоянием обучения с AMP"""
	
	def __init__(self, use_amp=False):
		self.use_amp = use_amp
		self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
		self.accumulation_count = 0
		self.nan_loss_count = 0
		self.max_nan_losses = 3
		
	def handle_nan(self, optimizer, model):
		"""Обработка NaN с поддержанием консистентности scaler"""
		self.nan_loss_count += 1
		
		if self.nan_loss_count >= self.max_nan_losses:
			# Перезагрузка модели
			self._reload_checkpoint(model, optimizer)
			return 'reload'
		
		# Сброс состояния для пропуска батча
		optimizer.zero_grad()
		self.accumulation_count = 0
		
		# Обновление scaler для поддержания консистентности
		if self.use_amp:
			try:
				self.scaler.update()
				print(f"   🔄 Scaler обновлён после NaN (состояние: {self.scaler.get_scale():.4e})")
			except:
				# Пересоздание scaler в крайнем случае
				self.scaler = torch.cuda.amp.GradScaler(enabled=True)
		
		return 'skip'
	
	def _reload_checkpoint(self, model, optimizer):
		"""Перезагрузка с полным сбросом состояния"""
		# Загрузка чекпоинта
		# ...
		
		# Полный сброс scaler
		if self.use_amp:
			self.scaler = torch.cuda.amp.GradScaler(enabled=True)
		
		self.nan_loss_count = 0
		self.accumulation_count = 0



# ================= УЛУЧШЕННОЕ СОХРАНЕНИЕ =================

def save_checkpoint(model, tokenizer, optimizer, step, loss, epoch, checkpoint_dir, 
					is_best=False, scheduler=None, monitor=None):
	"""
	Улучшенное сохранение чекпоинта с метриками
	Чекпоинты сохраняются в float32 для совместимости, даже если обучение шло в mixed precision.
	"""
	try:
		checkpoint_dir = Path(checkpoint_dir)
		checkpoint_dir.mkdir(parents=True, exist_ok=True)
		
		print(f"   💾 Сохранение чекпоинта шаг {step}...")
		
		# Сохраняем модель
		model_to_save.save_pretrained(str(checkpoint_dir))
		tokenizer.save_pretrained(str(checkpoint_dir))
		
		# Подготовка состояния
		checkpoint_state = {
			'step': step,
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': float(loss),
			'precision_mode': 'amp_16bit' if USE_AMP else 'full_32bit',
			'batch_size': BATCH_SIZE,
			'learning_rate': LEARNING_RATE,
			'timestamp': datetime.now().isoformat(),
		}
		
		if scheduler:
			checkpoint_state['scheduler_step'] = scheduler.current_step
		
		if monitor and monitor.quality_scores:
			checkpoint_state['last_quality'] = monitor.quality_scores[-1] if monitor.quality_scores else None
		
		torch.save(checkpoint_state, checkpoint_dir / "checkpoint.pt")
		
		# Сохраняем информацию о чекпоинте
		info_file = checkpoint_dir / "checkpoint_info.txt"
		with open(info_file, 'w', encoding='utf-8') as f:
			f.write(f"ЧЕКПОИНТ {step}\n")
			f.write(f"Эпоха: {epoch}\n")
			f.write(f"Loss: {loss:.6f}\n")
			f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			if is_best:
				f.write("\n 🏆 СТАТУС: ЛУЧШАЯ МОДЕЛЬ\n")
		
		print("    ✅ Чекпоинт сохранён")
		return True
		
	except Exception as e:
		print(f"   ❌ Ошибка при сохранении: {e}")
		return False

def load_last_checkpoint(checkpoint_dir, model, optimizer=None):
	"""Загрузка последнего чекпоинта при ошибках"""
	try:
		checkpoint_dir = Path(checkpoint_dir)
		checkpoints = sorted(checkpoint_dir.glob("step_*"), 
						   key=lambda x: int(x.name.split('_')[1]) if x.name.split('_')[1].isdigit() else 0,
						   reverse=True)
		
		if checkpoints:
			last_checkpoint = checkpoints[0]
			checkpoint = torch.load(last_checkpoint / "checkpoint.pt", map_location='cpu')
			
			model.load_state_dict(checkpoint['model_state_dict'])
			if optimizer and 'optimizer_state_dict' in checkpoint:
				optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			
			print(f"✅ Загружен чекпоинт: {last_checkpoint.name}")
			return checkpoint['step'], checkpoint['loss'], checkpoint['epoch']
	
	except Exception as e:
		print(f"❌ Ошибка загрузки чекпоинта: {e}")
	
	return 0, float('inf'), 0

# ================= ПУТИ =================
try:
	with open('paths.json', 'r') as pa:
		base_paths = json.load(pa)
except Exception as e:
	print(f"❌ Ошибка загрузки paths.json: {e}")
	sys.exit(1)

BASE_DIR = Path(base_paths.get('base_dir'))
CHECKPOINTS_DIR = base_paths.get('checks_dir')
FINAL_MODEL_DIR = base_paths.get('final_model_dir')
LOGS_DIR = base_paths.get('logs_dir')
DATA_DIR = base_paths.get('data_dir')

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Инициализируем продвинутый мониторинг с выбранным режимом
monitor = AdvancedTrainingMonitor(LOGS_DIR, tokenizer, debug_mode=DEBUG_MODE)

# Только в отладочном режиме показываем проверку
if DEBUG_MODE >= 2:
	print("\n 🔍 ПРОВЕРКА МОНИТОРИНГА:")
	print(f"   • log_dir: {monitor.log_dir}")
	
	# Тестовая запись
	monitor.save_to_csv(0, 1.0, 1e-4, 5.0, "TEST", 10.0, 0.5)

# ПРОВЕРКА СРАЗУ ПОСЛЕ СОЗДАНИЯ
print("\n 🔍 ПРОВЕРКА МОНИТОРИНГА:")
print(f"   • log_dir: {monitor.log_dir}")
print(f"   • Существует: {monitor.log_dir.exists()}")

# Тестовая запись через monitor
monitor.save_to_csv(0, 1.0, 1e-4, 5.0, "TEST", 10.0, 0.5)

# Проверим файл
csv_file = monitor.log_dir / "training_log.csv"
print(f"   • CSV файл создан: {csv_file.exists()}")
if csv_file.exists():
	print(f"   • Размер файла: {csv_file.stat().st_size} байт")
	with open(csv_file, 'r') as f:
		print(f"   • Содержимое:\n{f.read()}")

# ================= ЗАГРУЗКА ДАННЫХ =================
print("\n 📂 Загрузка диалогов...")

data_file = base_paths.get('dataset_file')
if data_file.exists():
	with open(data_file, 'r', encoding='utf-8') as f:
		dialogues = json.load(f)
	
	print(f"✅ Загружено {len(dialogues)} диалогов")
	
	texts = [dialogue['text'] for dialogue in dialogues]
	
else:
	print(f"❌ Файл не найден: {data_file}")
	sys.exit(1)

# ================= ТОКЕНИЗАЦИЯ =================
print("\n 🔤 Токенизация данных...")

tokenizer = GPT2Tokenizer.from_pretrained(base_paths.get('source_model_dir'))
if not Path(base_paths.get('source_model_dir')).exists():
	print(f"❌ Директория модели не найдена: {base_paths.get('source_model_dir')}")
	sys.exit(1)
tokenizer.pad_token = tokenizer.eos_token

all_tokens = []
for text in texts:
	tokens = tokenizer.encode(
		text,
		max_length=MAX_LENGTH,
		truncation=True,
		padding='max_length',
		return_tensors='pt'
	)
	all_tokens.append(tokens)

all_tokens = torch.cat(all_tokens, dim=0)

# Создаем простой Dataset
class TensorDataset(torch.utils.data.Dataset):
	def __init__(self, tensors):
		self.tensors = tensors
	def __len__(self):
		return len(self.tensors)
	def __getitem__(self, idx):
		return self.tensors[idx]

# Разделение
split_idx = int(0.85 * len(all_tokens))
train_dataset = TensorDataset(all_tokens[:split_idx])
val_dataset = TensorDataset(all_tokens[split_idx:])

# DataLoader с автоматическим shuffle
train_loader = torch.utils.data.DataLoader(
	train_dataset, 
	batch_size=BATCH_SIZE,
	shuffle=True,  # ✅ Автоматическое перемешивание
	num_workers=0   # Для начала 0, можно увеличить если много ядер
)

print(f"   Train: {len(train_dataset)} примеров")
print(f"   Validation: {len(val_dataset)} примеров")

# ================= ЗАГРУЗКА МОДЕЛИ =================
print("\n 🧠 Загрузка модели...")

# Без квантования
model = GPTNeoForCausalLM.from_pretrained(
	base_paths.get('source_model_dir'),
	device_map="auto",
	torch_dtype=torch.float16 if USE_AMP else torch.float32,  # Загружаем веса сразу в выбранной точности
)

print(" ✅ Модель загружена")

# ================= ОПТИМИЗАТОР =================
print("\n ⚡ Настройка оптимизатора...")

optimizer = bnb.optim.AdamW8bit(
	model.parameters(),
	lr=LEARNING_RATE,
	betas=(0.9, 0.95),
	weight_decay=0.01,
)

# ================= РАСЧЕТ ШАГОВ И ШЕДУЛЕР =================

# НАЙДИТЕ ЭТУ СТРОКУ (~740) И ИСПРАВЬТЕ:
total_batches = len(train_data) // BATCH_SIZE
# total_steps = (total_batches // GRADIENT_ACCUMULATION) * EPOCHS  # ❌ СТАРОЕ

# ⬇️ НОВОЕ:
if GRADIENT_ACCUMULATION > 0:
	total_steps = max(1, math.ceil((total_batches + GRADIENT_ACCUMULATION - 1) // GRADIENT_ACCUMULATION * EPOCHS))
else:
	total_steps = max(1, total_batches * EPOCHS)

print("\n 📈 ПЛАН ОБУЧЕНИЯ:")
print(f"   • Всего шагов: {total_steps}")

scheduler = OptimalScheduler(optimizer, total_steps, LEARNING_RATE, WARMUP_RATIO)

# Настройки для улучшенного раннего стоппинга
checkpoint_steps = [25, 50, 100, 200, 400, 600, 800]
best_loss = float('inf')
best_model_step = 0
patience = 3
patience_counter = 0
previous_val_loss = float('inf')
min_delta = 0.001  # Минимальное значимое улучшение

# Счетчики для обработки ошибок
nan_loss_count = 0
max_nan_losses = 3

# ================= ОБУЧЕНИЕ С АДАПТИВНЫМИ НАСТРОЙКАМИ =================
print("\n 🎯 НАЧИНАЮ ОБУЧЕНИЕ С АДАПТИВНЫМИ НАСТРОЙКАМИ...")

with TrainingMode(model):  # ✅ Автоматически настраивает use_cache и gradient_checkpointing
	print("    • Режим: ОБУЧЕНИЕ")
	print(f"   • use_cache: {model.config.use_cache}")
	print(f"   • gradient_checkpointing: {model.is_gradient_checkpointing}")

global_step = 0
start_time = datetime.now()

# Сохраняем начальный чекпоинт
initial_checkpoint_dir = CHECKPOINTS_DIR / "initial_model"
save_checkpoint(model, tokenizer, optimizer, 0, float('inf'), 0, initial_checkpoint_dir)

for epoch in range(EPOCHS):
	print(f"\n{'='*60}")
	print(f"📚 ЭПОХА {epoch+1}/{EPOCHS}")
	print(f"{'='*60}")
	
	if USE_AMP:
		try:
	# Проверка, что scaler в валидном состоянии
			current_scale = GRAD_SCALER.get_scale()
			if current_scale <= 0 or math.isnan(current_scale) or math.isinf(current_scale):
				print(f"   ⚠️  Обнаружен некорректный scale ({current_scale:.2e}), пересоздаём scaler")
				GRAD_SCALER = torch.cuda.amp.GradScaler(enabled=True)
				print(f"   ✅ Новый scale: {GRAD_SCALER.get_scale():.2e}")
		except Exception as e:
			print(f"   ⚠️  Ошибка проверки scaler: {e}, пересоздаём")
			GRAD_SCALER = torch.cuda.amp.GradScaler(enabled=True)
	# ================= КОНЕЦ ПРОВЕРКИ =================

	epoch_loss = 0.0
	batch_count = 0
	accumulation_count = 0
	epoch_start_time = time.time()
	last_print_time = time.time()
	
	# Перемешиваем данные
	train_indices = torch.randperm(len(train_data))
	train_data_shuffled = train_data[train_indices]
	
	check_scaler_health()

	with TrainingMode(model):
		if HAS_TQDM and DEBUG_MODE <= 2:
    		pbar = tqdm(total=total_batches, desc=f"Эпоха {epoch+1}", unit="батч")
		for batch_idx in range(0, len(train_data_shuffled), BATCH_SIZE):
			total_batches = len(train_data_shuffled) // BATCH_SIZE

			# 🛡️ ЗАЩИТА SCALER ПЕРЕД НАЧАЛОМ ЭПОХИ
			if USE_AMP and epoch > 0:  # Проверяем со второй эпохи
				scaler_ok = check_scaler_health(GRAD_SCALER, f"Эпоха {epoch+1}")
				if not scaler_ok:
					GRAD_SCALER = torch.cuda.amp.GradScaler(enabled=True)
					print(f"   ✅ Scaler пересоздан перед эпохой {epoch+1}")

			step_start = time.time()

			if batch_idx + BATCH_SIZE > len(train_data_shuffled):
				continue
				
			
			batch = train_data_shuffled[batch_idx:batch_idx+BATCH_SIZE].cuda()
			if time.time() - step_start > MAX_STEP_TIME:
				print(f"⚠️  Загрузка батча заняла {MAX_STEP_TIME} секунд")
			batch_start_time = time.time()
			try:
				optimizer.zero_grad()

				with torch.cuda.amp.autocast(enabled=USE_AMP):
					outputs = model(batch, labels=batch)
					loss = outputs.loss
					loss_value = loss.item()

				# Проверка на NaN
				if math.isnan(loss_value):
					nan_loss_count += 1
					print(f"   ⚠️  NaN loss detected ({nan_loss_count}/{max_nan_losses})")
					
					if nan_loss_count >= max_nan_losses:
						print("    🔄 Перезагрузка последнего чекпоинта...")
						global_step, _, _ = load_last_checkpoint(CHECKPOINTS_DIR, model, optimizer)
						# Сбрасываем состояние scaler при перезагрузке чекпоинта
						if USE_AMP:
							GRAD_SCALER = torch.cuda.amp.GradScaler(enabled=True)
						nan_loss_count = 0
						optimizer.zero_grad()
						accumulation_count = 0  # ⬅️ СБРАСЫВАЕМ accumulation_count
						continue
					
					# Пропускаем проблемный батч
					optimizer.zero_grad()
					accumulation_count = 0  # ⬅️ СБРАСЫВАЕМ accumulation_count
					# Мы не делаем шаг оптимизатора, но должны обновить scaler
					# ВАЖНО: При пропуске шага из-за NaN обновляем scaler
					if USE_AMP and accumulation_count > 0:
						# Мы не делаем шаг оптимизатора, но должны обновить scaler
						# для поддержания консистентности
						try:
							# Пропускаем шаг, но обновляем scaler
							GRAD_SCALER.update()  # ⬅️ КРИТИЧЕСКИ ВАЖНО!
							print(f"   🔄 Обновлен scaler после пропуска NaN батча")
						except Exception as e:
							print(f"   ⚠️  Ошибка при обновлении scaler: {e}")
							# Пересоздаём scaler в случае ошибки
							GRAD_SCALER = torch.cuda.amp.GradScaler(enabled=True)
					continue
				
				 # ОБРАТНЫЙ ПРОХОД с поддержкой AMP и скейлером
				epoch_loss += loss_value
				batch_count += 1
				
				# Gradient accumulation (делим накопленный loss)
				accumulated_loss = loss / GRADIENT_ACCUMULATION
   
				if USE_AMP:
					GRAD_SCALER.scale(accumulated_loss).backward()
				else:
					accumulated_loss.backward()

				accumulation_count += 1
	
				if accumulation_count % GRADIENT_ACCUMULATION == 0:
					if USE_AMP:
						# Применяем gradient clipping к масштабированным градиентам
						GRAD_SCALER.unscale_(optimizer)
						grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		
						# Шаг оптимизатора через скейлер
						GRAD_SCALER.step(optimizer)
						try:
						# Проверка scale перед обновлением
						current_scale = GRAD_SCALER.get_scale()
						if current_scale > 1e6 or current_scale < 1e-6:
							print(f"   ⚠️  Подозрительный scale: {current_scale:.2e}, сбрасываем")
							GRAD_SCALER = torch.cuda.amp.GradScaler(enabled=True)
					except:
						pass
						GRAD_SCALER.update()  # Обновляем масштаб для следующей итерации
					else:
						grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
						optimizer.step()

					current_lr, phase = scheduler.step()
					optimizer.zero_grad()
		
					global_step += 1

					if HAS_TQDM and DEBUG_MODE <= 2:
        				pbar.update(1)
				# ================= ВЫВОД ПРОГРЕССА =================
				current_lr = LEARNING_RATE  # начальное значение

				current_time = time.time()
				if current_time - last_print_time > 10:  # Каждые 10 секунд
					progress = (batch_idx / len(train_data_shuffled)) * 100
					avg_loss_so_far = epoch_loss / (batch_count + 1e-8)
	
					# РАСЧЕТ СКОРОСТИ
					elapsed_since_last_print = current_time - last_print_time
					batches_since_last_print = (batch_idx // BATCH_SIZE) - last_batch_count if 'last_batch_count' in locals() else 1
					last_batch_count = batch_idx // BATCH_SIZE
	
					dialogs_per_second = batches_since_last_print * BATCH_SIZE / elapsed_since_last_print if elapsed_since_last_print > 0 else 0
					tokens_per_second = dialogs_per_second * MAX_LENGTH  # Примерная скорость в токенах
	
					if DEBUG_MODE == 1:
					# Цветной вывод (если терминал поддерживает)
						try:
						# Определяем цвет для скорости
							if dialogs_per_second > 0.5:
								speed_color = "\033[92m"  # зеленый
								speed_icon = "🚀"
							elif dialogs_per_second > 0.2:
								speed_color = "\033[93m"  # желтый
								speed_icon = "⚡"
							else:
								speed_color = "\033[91m"  # красный
								speed_icon = "🐌"
			
							reset_color = "\033[0m"
			
							print(f"\r   🔄 {progress:5.1f}% | 📉 {loss_value:7.4f} | 🎛️ {current_lr:.1e} | 🧺 {batch_idx//BATCH_SIZE:4d} | {speed_icon} {speed_color}{dialogs_per_second:5.2f} диал/с{reset_color}", end='', flush=True)
						except:
							# Без цветов если не поддерживается
							print(f"\r   🔄 {progress:5.1f}% | Loss: {loss_value:7.4f} | LR: {current_lr:.2e} | Батч: {batch_idx//BATCH_SIZE:4d} | 🚀 {dialogs_per_second:5.2f} д/с", end='', flush=True)
	
					elif DEBUG_MODE == 2:
						# Подробный вывод
						print(f"\n   ⏰ {datetime.now().strftime('%H:%M:%S')}")
						print(f"   📍 Батч {batch_idx//BATCH_SIZE} ({progress:.1f}%)")
						print(f"   📉 Loss: {loss_value:.4f} (средн: {avg_loss_so_far:.4f})")
						print(f"   🚀 Скорость: {dialogs_per_second:.2f} д/с (~{tokens_per_second/1000:.1f}K токенов/сек)")
						print(f"   💾 GPU память: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
						print(f"   ⚡ GPU мощность: {get_gpu_power()}W")  # если есть функция получения мощности
	
					last_print_time = current_time

				elif DEBUG_MODE == 1:
					# Быстрое обновление (без расчета скорости)
					progress = (batch_idx / len(train_data_shuffled)) * 100
					print(f"\r   🔄 {progress:5.1f}% | Loss: {loss_value:7.4f} | LR: {current_lr:.2e} | Батч: {batch_idx//BATCH_SIZE:4d} | ⏳...", end='', flush=True)
				# ================= КОНЕЦ ВЫВОДА ПРОГРЕССА =================
				
				# Gradient accumulation
				loss = loss / GRADIENT_ACCUMULATION
				loss.backward()
				
				accumulation_count += 1
				
				# Step с gradient accumulation
				if accumulation_count % GRADIENT_ACCUMULATION == 0:
					# Gradient clipping
					grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
					
					# Оптимизатор
					optimizer.step()
					current_lr, phase = scheduler.step()  # ⬅️ ТЕПЕРЬ current_lr обновлен
					optimizer.zero_grad()
					
					global_step += 1
					step_time = time.time() - batch_start_time
					
					# Мониторинг
					memory_gb = torch.cuda.memory_allocated() / 1024**3
					monitor.log_batch(global_step, loss_value, current_lr, grad_norm, memory_gb, step_time, phase)
					
					# Логирование каждые 10 шагов
					if global_step % 10 == 0:
						avg_loss = epoch_loss / batch_count
						elapsed = (datetime.now() - start_time).seconds / 60
						
						print(f"\n   Шаг {global_step} [{phase}]:")
						print(f"   • Loss: {loss_value:.4f} | Avg: {avg_loss:.4f}")
						print(f"   • LR: {current_lr:.2e}")
						print(f"   • Время: {elapsed:.1f} мин")
					
					# Продвинутая проверка качества
					if global_step % 50 == 0:
						quality_score, empathy_score, current_temp = monitor.advanced_quality_check(
							model, tokenizer, global_step, adaptive_temp=True
						)
						print(f"   • Качество: {quality_score:.2f} | Эмпатия: {empathy_score:.2f} | Temp: {current_temp:.3f}")
					
					# Сохранение чекпоинтов
					if global_step in checkpoint_steps:
						checkpoint_dir = CHECKPOINTS_DIR / f"step_{global_step}_epoch_{epoch+1}"
						save_checkpoint(model, tokenizer, optimizer, global_step, 
									  epoch_loss/batch_count, epoch+1, checkpoint_dir, 
									  scheduler=scheduler, monitor=monitor)
				
			except torch.cuda.OutOfMemoryError:  # ← ДОБАВЬТЕ ЭТУ СТРОКУ
				print("⚠️  Не хватило памяти GPU, очищаю...")
				torch.cuda.empty_cache()
				continue

			# Проверка времени всего шага
			if time.time() - step_start > MAX_STEP_TIME:
				print(f"⚠️  Весь шаг занял {MAX_STEP_TIME} секунд")
	
	# ================= КОНЕЦ ЭПОХИ =================
	if DEBUG_MODE == 1:
		print()  # Переводим строку после прогресс-бара
	
	# Итоги эпохи
	avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
	print(f"\n✅ ЭПОХА {epoch+1} завершена:")
	print(f"   • Train Loss: {avg_epoch_loss:.4f}")
	print(f"   • Шагов: {global_step}")
	
	# Расчет perplexity на валидации
	perplexity = monitor.calculate_perplexity(model, val_data, BATCH_SIZE, epoch=epoch+1)
	print(f"   • Perplexity: {perplexity:.2f}")
	
	# Улучшенный ранний стоппинг
	if previous_val_loss != float('inf'):
		improvement = previous_val_loss - perplexity
		
		if improvement < min_delta:
			patience_counter += 1
			print(f"   ⚠️  Малое улучшение perplexity ({improvement:.4f} < {min_delta}). Patience: {patience_counter}/{patience}")
		else:
			patience_counter = 0
			print(f"   ✅ Значительное улучшение perplexity: {improvement:.4f}")
		
		if patience_counter >= patience:
			print(f"\n🚫 РАННЯЯ ОСТАНОВКА: нет значимых улучшений {patience} эпохи подряд")
			break
	
	# Сохранение лучшей модели
	if perplexity < best_loss:
		best_loss = perplexity
		best_model_step = global_step
		
		best_dir = CHECKPOINTS_DIR / f"BEST_epoch_{epoch+1}_perplexity_{best_loss:.2f}"
		save_checkpoint(model, tokenizer, optimizer, global_step, 
					  best_loss, epoch+1, best_dir, is_best=True, 
					  scheduler=scheduler, monitor=monitor)
		print(f"   🏆 НОВАЯ ЛУЧШАЯ МОДЕЛЬ: perplexity={best_loss:.2f}")
	
	previous_val_loss = perplexity
	
	# Сохраняем чекпоинт эпохи
	epoch_checkpoint_dir = CHECKPOINTS_DIR / f"epoch_{epoch+1}_final"
	save_checkpoint(model, tokenizer, optimizer, global_step, avg_epoch_loss, 
				   epoch+1, epoch_checkpoint_dir, scheduler=scheduler, monitor=monitor)
	monitor.flush()

# ================= СОХРАНЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ =================
print("\n 💾 Конвертация и сохранение финальной модели в 32-битном формате...")

try:
	# Приводим модель к float32, независимо от режима обучения
	model = model.float()  # Эта операция гарантирует 32-битные веса
	
	model.save_pretrained(str(FINAL_MODEL_DIR))
	tokenizer.save_pretrained(str(FINAL_MODEL_DIR))
	
	training_info = {
		'total_steps': global_step,
		'final_train_loss': avg_epoch_loss,
		'best_perplexity': best_loss,
		'best_step': best_model_step,
		'epochs_completed': epoch + 1,
		'early_stopped': patience_counter >= patience,
		'training_precision': 'float16 (AMP)' if USE_AMP else 'float32',
		'saved_in_precision': 'float32',  # Всегда 32 бита
		'final_perplexity': perplexity,
		'batch_size': BATCH_SIZE,
		'learning_rate': LEARNING_RATE,
		'training_time_minutes': (datetime.now() - start_time).seconds / 60,
		'completion_time': datetime.now().isoformat(),
		'adaptive_training': True,
		'advanced_metrics': True,
		'gradient_checkpointing': True,
		'use_cache_strategy': 'adaptive'
	}
	
	with open(FINAL_MODEL_DIR / "training_info.json", 'w', encoding='utf-8') as f:
		json.dump(training_info, f, ensure_ascii=False, indent=2)
	
	print(" ✅ Финальная модель сохранена")
	
except Exception as e:
	print(f"❌ Ошибка сохранения: {e}")

# ================= ФИНАЛЬНЫЙ ТЕСТ =================
print("\n 🧪 ФИНАЛЬНЫЙ ТЕСТ С АДАПТИВНЫМИ НАСТРОЙКАМИ...")
with GenerationMode(model):  # ✅ ГЕНЕРАЦИЯ: cache=ON, gc=OFF
	print("    • Режим: ГЕНЕРАЦИЯ")
	print(f"   • use_cache: {model.config.use_cache}")
	print(f"   • gradient_checkpointing: {model.is_gradient_checkpointing}")

test_prompts = [
	"Пациент: Не могу перестать волноваться.",
	"Пациент: Чувствую себя очень одиноко.",
	"Пациент: Как найти смысл в жизни?",
	"Пациент: Всё бессмысленно, не вижу причин продолжать.",
	"Пациент: Боюсь, что никогда не изменюсь."
]

for i, prompt in enumerate(test_prompts):
	try:
		# Используем адаптивную температуру на основе качества модели
		last_quality = monitor.quality_scores[-1][1] if monitor.quality_scores else 0.5
		adaptive_temp = max(0.6, 0.9 - (last_quality * 0.3))
		
		with GenerationMode(model):  # Каждый генерационный вызов в правильном режиме
			response = monitor.generate_adaptive_response(model, tokenizer, prompt, adaptive_temp)
			score = monitor.evaluate_response_comprehensive(prompt, response)
			empathy_score = monitor.calculate_empathy_score(response)
		
		print(f"\n{i+1}. 💭 {prompt}")
		print(f"   🌡️  Temp: {adaptive_temp:.3f}")
		print(f"   💬 {response[:120]}{'...' if len(response) > 120 else ''}")
		print(f"   📊 Оценка: {score:.2f} | Эмпатия: {empathy_score:.2f}")
		
	except Exception as e:
		print(f"\n{i+1}. ❌ Ошибка: {e}")

print(f"\n{'='*80}")
print("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print(f"{'='*80}")
print(" 📊 ИТОГОВЫЕ МЕТРИКИ:")
print(f"   • Шагов: {global_step}")
print(f"   • Лучший perplexity: {best_loss:.2f}")
print(f"   • Финальный perplexity: {perplexity:.2f}")
print(f"   • Ранняя остановка: {'Да' if patience_counter >= patience else 'Нет'}")
print(f"   • NaN обработок: {nan_loss_count}")
print(f"   • Время: {(datetime.now() - start_time).seconds/60:.1f} мин")
print("    • Использованные режимы:")
print(f"      - Обучение: cache=OFF, gradient_checkpointing={model.is_gradient_checkpointing}")
print("       - Валидация: cache=OFF, gradient_checkpointing=OFF")
print("       - Генерация: cache=ON, gradient_checkpointing=OFF")
print(f"{'='*80}")
