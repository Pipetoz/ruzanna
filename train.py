# train.py
ma = False
if __name__ == "__main__":
	ma = True
import time
tt = lambda: time.time()
tot = tt()
from colorama import Back, Fore, Style, init
init()
def ts(t, n):
	v = float(format(tt() - t, ".2f"))
	cp = ""
	if v <= 0.3: cp = f"{Fore.GREEN}{v}"
	if v > 0.3 and v <= 1.2: cp = f"{Fore.YELLOW}{v}"
	if v > 1.2: cp = f"{Fore.RED}{v}"
	c = f"{n}: {cp} сек{Fore.RESET}"
	return c

st = tt()
if ma: print()

import os
if ma: print(ts(st, "os"))

import sys
if ma: print(ts(st, "sys"))

from contextlib import redirect_stdout, redirect_stderr
if ma: print(ts(st, "contextlib"))

import math
if ma: print(ts(st, "math"))

import pickle
if ma: print(ts(st, "pickle"))

import yaml
if ma: print(ts(st, "yaml"))

import numpy as np
if ma: print(ts(st, "numpy"))

from pathlib import Path
if ma: print(ts(st, "pathlib"))

from config import config, clear_screen, error, header, info, progress_bar, success, title, warning, rulables
if ma: print(ts(st, "config"))

from tqdm import tqdm
if ma: print(ts(st, "tqdm"))

import torch
if ma: print(ts(st, "torch"))

import torch.nn as nn
if ma: print(ts(st, "torch.nn"))

from sklearn.preprocessing import MultiLabelBinarizer
if ma: print(ts(st, "sklearn"))

from torch.amp import GradScaler, autocast
if ma: print(ts(st, "torch.amp"))

from torch.optim import AdamW
if ma: print(ts(st, "torch.optim"))

from torch.utils.data import DataLoader, Dataset
if ma: print(ts(st, "torch.utils.data"))

from torch.utils.tensorboard import SummaryWriter
if ma: print(ts(st, "torch.utils.tensorboard"))

from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_constant_schedule_with_warmup
if ma: print(ts(st, "transformers"))

if ma: print("\n" + ts(tot, "Общее время импортов") + "\n")

def training_menu():
	"""Меню выбора режима обучения"""
	#clear_screen()
	title("РЕЖИМ ОБУЧЕНИЯ МОДЕЛИ")

	print(f"{Style.BRIGHT}1.{Style.RESET_ALL} Полное обучение ({EPOCHS} эпох, все данные)")
	print(f"{Style.BRIGHT}2.{Style.RESET_ALL} Тестовый режим (1 эпоха, ограниченные данные)")
	print(f"{Style.BRIGHT}3.{Style.RESET_ALL} Настроить параметры обучения")
	print(f"{Style.BRIGHT}4.{Style.RESET_ALL} Назад в главное меню")
	print()

	while True:
		choice = input(f"{Fore.CYAN}Выберите режим [1-4]: {Style.RESET_ALL}").strip()

		if choice == "1":
			return False, None  # Обычный режим
		elif choice == "2":
			print(f"\n{Fore.YELLOW}Тестовый режим:{Style.RESET_ALL}")
			print(f"  • 1 эпоха обучения")
			print(f"  • Ограниченное количество примеров")
			print(f"  • Модель не сохраняется")
			print(f"  • Быстрая проверка работоспособности")

			confirm = input(f"\n{Fore.GREEN}Запустить тестовый режим? [y/N]: {Style.RESET_ALL}").strip().lower()
			if confirm in ['y', 'yes', 'д', 'да']:
				test_size = input(f"{Fore.YELLOW}Примеров для теста (по умолчанию 100): {Style.RESET_ALL}").strip()
				test_size = int(test_size) if test_size.isdigit() else 100
				return True, test_size
			else:
				continue
		elif choice == "3":
			# Показать текущие параметры
			print(f"\n{Fore.YELLOW}Текущие параметры обучения:{Style.RESET_ALL}")
			print(f"  • Модель: {MODEL_NAME}")
			print(f"  • Эпохи: {EPOCHS}")
			print(f"  • Batch size: {BATCH_SIZE}")
			print(f"  • Learning rate: {LEARNING_RATE}")
			print(f"  • Max length: {MAX_LEN}")
			print(f"\nДля изменения параметров отредактируйте файл: config.yaml")
			input(f"\n{Fore.CYAN}Нажмите Enter для продолжения...{Style.RESET_ALL}")
			continue
		elif choice == "4":
			return None, None  # Выход
		else:
			error("Неверный выбор. Введите 1, 2, 3 или 4.")

CHECKPOINTS_DIR = config['checks_dir']
LOG_DIR = config['logs_dir']
OUTPUT_DIR = config['final_model_dir']
MODEL_NAME = config['source_model_dir']

EPOCHS = int(config['epochs'])
MAX_LEN = int(config['max_len'])
BATCH_SIZE = int(config['batch_size'])
ACCUMULATION_STEPS = int(config['accumulation_steps'])
LEARNING_RATE = float(config['learning_rate'])
WARMUP_STEPS = int(config['warmup_steps'])
WEIGHT_DECAY = float(config['weight_decay'])
FP32 = config['fp32']
USE_TRITON = config['use_triton']

# После загрузки констант из config, добавьте проверку:
print(f"{Fore.CYAN}{Style.BRIGHT}=== КОНФИГУРАЦИЯ ОБУЧЕНИЯ ==={Style.RESET_ALL}")
print(f"Модель: {MODEL_NAME}")
print(f"Эпохи: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE} (аккумуляция: {ACCUMULATION_STEPS})")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Max length: {MAX_LEN}")
print(f"FP32: {FP32} (AMP: {not FP32})")
print(f"Triton: {USE_TRITON}")
print()

def suppress_output(func):
	def wrapper(*args, **kwargs):
		import os
		# Кросс-платформенное определение null-устройства
		null_device = os.devnull  # 'NUL' на Windows, '/dev/null' на Linux/Mac

		with open(null_device, 'w') as f:
			old_stdout = sys.stdout
			old_stderr = sys.stderr
			sys.stdout = f
			sys.stderr = f
			try:
				return func(*args, **kwargs)
			finally:
				sys.stdout = old_stdout
				sys.stderr = old_stderr
	return wrapper

@suppress_output
def compile_model(model):
	return torch.compile(model, backend="inductor", mode="default")

# ========== КЛАСС ДАТАСЕТА ДЛЯ МНОГОМЕТОЧНОЙ КЛАССИФИКАЦИИ ==========
class MultiLabelEmotionsDataset(Dataset):
	"""
	Датасет для многометочной классификации (несколько эмоций на текст).
	Принимает метки в формате one-hot векторов от MultiLabelBinarizer.
	"""
	def __init__(self, texts, labels, tokenizer, max_len):
		self.texts = texts
		self.labels = labels  # one-hot векторы [[0,1,0,...], ...]
		self.tokenizer = tokenizer
		self.max_len = max_len

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, item):
		text = str(self.texts[item])
		label = torch.tensor(self.labels[item], dtype=torch.float)  # ← Важно: torch.float

		encoding = self.tokenizer(
			text,
			truncation=True,
			padding='max_length',
			max_length=self.max_len,
			return_tensors='pt',
		)

		return {
			'input_ids': encoding['input_ids'][0],
			'attention_mask': encoding['attention_mask'][0],
			'labels': label  # One-hot вектор с float типом
		}

		# Проверка формата меток
		assert len(train_labels[0]) == 28, f"Ожидалось 28 классов, получено {len(train_labels[0])}"
		assert train_labels[0].dtype == np.int64 or train_labels[0].dtype == np.float64, f"Неверный тип меток: {train_labels[0].dtype}"
		print(f"✅ Метки преобразованы в one-hot формат: {train_labels.shape[1]} классов")

# ========== ГЛАВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ ==========
def train(test_mode=False, test_sample_size=100):
	"""
	Основная функция обучения для многометочной классификации эмоций
	с поддержкой тестового режима и параметров из config.yaml

	Args:
		test_mode (bool): Если True, запускается тестовый режим
		test_sample_size (int): Количество примеров для теста
	"""

	global USE_TRITON

	# ========== НАСТРОЙКА ПАРАМЕТРОВ ==========
	print(f"{Fore.CYAN}{Style.BRIGHT}=== {'ТЕСТОВЫЙ РЕЖИМ' if test_mode else 'ОБУЧЕНИЕ'} ==={Style.RESET_ALL}")

	# Загрузка параметров из конфига (уже сделано глобально)
	# Используем глобальные константы, прочитанные из config
	if test_mode:
		# В тестовом режиме меняем некоторые параметры
		test_epochs = 1
		test_batch_size = BATCH_SIZE
		test_logging_steps = 5
		info(f"ТЕСТОВЫЙ РЕЖИМ: {test_epochs} эпоха, ~{test_sample_size} примеров")
	else:
		test_epochs = EPOCHS
		test_batch_size = BATCH_SIZE
		test_logging_steps = 50

	# ========== ПОДГОТОВКА УСТРОЙСТВА ==========
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == "cuda":
		success(f"Используется GPU: {torch.cuda.get_device_name(0)}")
		# Оптимизация для CUDA
		torch.backends.cudnn.benchmark = True
	else:
		warning("Используется CPU: обучение будет медленным")

	# ========== ЗАГРУЗКА ТОКЕНИЗАТОРА ==========
	info(f"Загрузка токенизатора: {MODEL_NAME}")
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

	# ========== ЗАГРУЗКА ДАННЫХ ==========
	data_path = Path(config['data_dir']) / "ru_goemotions_metadata.pkl"
	info(f"Загрузка данных из: {data_path}")

	try:
		with open(data_path, "rb") as f:
			processed_data = pickle.load(f)
	except FileNotFoundError:
		error(f"Файл данных не найден: {data_path}")
		sys.exit(1)
	except pickle.PickleError as e:
		error(f"Ошибка загрузки pickle-файла: {e}")
		sys.exit(1)

	info(f"Ключи в данных: {list(processed_data.keys())}")

	# Проверка обязательных ключей
	required_keys = ["train", "val", "vectorizer", "label2id", "id2label"]
	missing_keys = [k for k in required_keys if k not in processed_data]

	if missing_keys:
		warning(f"Отсутствующие ключи: {missing_keys}")
		if "label2id" not in processed_data:
			label2id = {name: idx for idx, name in enumerate(rulables().keys())}
			processed_data["label2id"] = label2id
		if "id2label" not in processed_data:
			processed_data["id2label"] = {v: k for k, v in processed_data["label2id"].items()}

	# ========== ОБРАБОТКА ДАННЫХ ==========
	train_texts = processed_data["train"]["texts"]
	train_labels = processed_data["train"]["labels"]
	val_texts = processed_data["val"]["texts"]
	val_labels = processed_data["val"]["labels"]

	# Создаем one-hot векторы для 28 классов
	mlb = MultiLabelBinarizer(classes=range(28))

	# Преобразуем списки меток в one-hot векторы
	train_labels_onehot = mlb.fit_transform(train_labels)
	val_labels_onehot = mlb.transform(val_labels)

	# Заменяем исходные метки
	train_labels = train_labels_onehot
	val_labels = val_labels_onehot

	# Отладочный вывод
	print(f"Пример метки до преобразования: {processed_data['train']['labels'][0]}")
	print(f"Пример метки после преобразования: {train_labels[0]}")
	print(f"Сумма в примере (сколько эмоций): {train_labels[0].sum()}")

	# Тестовый режим: ограничение данных
	if test_mode and test_sample_size:
		info(f"Ограничение данных до {test_sample_size} примеров")

		# Ограничиваем обучающие данные
		train_texts = train_texts[:test_sample_size]
		train_labels = train_labels[:test_sample_size]

		# Для валидации берем пропорционально меньше
		val_size = max(10, test_sample_size // 10)
		val_texts = val_texts[:val_size]
		val_labels = val_labels[:val_size]

		if len(train_texts) == 0:
			error("После ограничения данных обучающая выборка пуста. Проверьте test_sample_size.")
			sys.exit(1)

	# Получаем количество классов
	num_classes = len(processed_data["label2id"])
	success(f"Количество классов (эмоций): {num_classes}")
	print(f"Обучающие примеры: {len(train_texts)}")
	print(f"Валидационные примеры: {len(val_texts)}")

	# ========== ПРЕОБРАЗОВАНИЕ МЕТОК ==========
	info("Преобразование меток в one-hot формат...")
	mlb = MultiLabelBinarizer(classes=list(range(num_classes)))
	train_labels_onehot = mlb.fit_transform(train_labels)
	val_labels_onehot = mlb.transform(val_labels)

	# Диагностика меток
	print(f"\n{Fore.YELLOW}=== ДИАГНОСТИКА МЕТОК ==={Style.RESET_ALL}")
	print(f"Формат train_labels_onehot: {train_labels_onehot.shape}")
	print(f"Пример метки: {train_labels_onehot[0]}")
	print(f"Количество эмоций на текст (среднее): {train_labels_onehot.sum(axis=1).mean():.2f}")

	# ========== СОЗДАНИЕ ДАТАСЕТОВ ==========
	info("Создание датасетов...")
	train_dataset = MultiLabelEmotionsDataset(train_texts, train_labels_onehot, tokenizer, MAX_LEN)
	val_dataset = MultiLabelEmotionsDataset(val_texts, val_labels_onehot, tokenizer, MAX_LEN)

	# ========== ЗАГРУЗКА МОДЕЛИ ==========
	info(f"Загрузка модели: {MODEL_NAME}")

	# Временно подавляем предупреждения transformers
	import warnings
	from transformers import logging as transformers_logging

	old_verbosity = transformers_logging.get_verbosity()
	transformers_logging.set_verbosity_error()
	warnings.filterwarnings("ignore",
		message="Some weights of RobertaForSequenceClassification were not initialized")

	try:
		# Ключевой параметр: problem_type="multi_label_classification"
		model = AutoModelForSequenceClassification.from_pretrained(
			MODEL_NAME,
			num_labels=num_classes,
			problem_type="multi_label_classification",
			hidden_dropout_prob=0.1,
			attention_probs_dropout_prob=0.1
		)

		success(f"✓ Модель загружена (многометочная классификация)")
		model.gradient_checkpointing_enable()
		success("Gradient Checkpointing включён")
	except OSError as e:
		error(f"Ошибка загрузки модели {MODEL_NAME}: {e}")
		sys.exit(1)
	except Exception as e:
		error(f"Неожиданная ошибка при загрузке модели: {e}")
		sys.exit(1)
	finally:
		transformers_logging.set_verbosity(old_verbosity)
		warnings.resetwarnings()

	# Перемещение модели на устройство
	model.to(device)
	info(f"Модель перемещена на: {device}")

	# ========== КОМПИЛЯЦИЯ С TRITON ==========
	if USE_TRITON and torch.cuda.is_available():
		info("Компиляция с Triton...")
		torch.set_float32_matmul_precision('high')
		model = compile_model(model)
		success("✓ Triton включён (inductor + default)")
	elif USE_TRITON and not torch.cuda.is_available():
		warning("Triton включён, но CUDA не доступен. Пропускаем компиляцию.")
		USE_TRITON = False

	# ========== НАСТРОЙКА ОБУЧЕНИЯ ==========
	info("Настройка обучения...")

	# Создаем папки для выходных данных
	os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
	os.makedirs(LOG_DIR, exist_ok=True)
	os.makedirs(OUTPUT_DIR, exist_ok=True)

	# ========== СОЗДАНИЕ DATALOADER ==========
	info("Создание DataLoader...")
	train_dataset = MultiLabelEmotionsDataset(train_texts, train_labels_onehot, tokenizer, MAX_LEN)
	val_dataset = MultiLabelEmotionsDataset(val_texts, val_labels_onehot, tokenizer, MAX_LEN)

	train_loader = DataLoader(
		train_dataset,
		batch_size=test_batch_size,
		shuffle=True,
		num_workers=0,
		pin_memory=False if device.type == "cuda" else True,
		persistent_workers = False
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=test_batch_size,
		shuffle=False,
		num_workers=0,
		pin_memory=False if device.type == "cuda" else True,
		persistent_workers = False
	)

	# ========== ПОДГОТОВКА ОПТИМИЗАТОРА И SCHEDULER ==========
	info("Подготовка оптимизатора...")

	# Более консервативный learning rate для начала
	current_lr = LEARNING_RATE
	info(f"Используется learning rate: {current_lr}")

	# Группируем параметры для weight decay
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": WEIGHT_DECAY,
		},
		{
			"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
			"weight_decay": 0.0,
		},
	]

	optimizer = AdamW(
		optimizer_grouped_parameters,
		lr=current_lr,
		betas=(0.9, 0.999),
		eps=1e-8
	)

	# Постоянный scheduler с прогревом - ПРОЩЕ И СТАБИЛЬНЕЕ
	total_steps = len(train_loader) // ACCUMULATION_STEPS * test_epochs
	warmup_steps = 50  # Фиксированные 500 шагов на прогрев
	info(f"Общее шагов: {total_steps}, Прогрев: {warmup_steps} шагов")

	scheduler = get_constant_schedule_with_warmup(
		optimizer,
		num_warmup_steps=warmup_steps
	)
	# Этот scheduler: 0 → LR за warmup_steps → потом постоянный LR

	# Mixed Precision
	scaler = GradScaler(enabled=not FP32)

	# TensorBoard
	writer = SummaryWriter(LOG_DIR)

	# ========== КАСТОМНЫЙ ЦИКЛ ОБУЧЕНИЯ ==========
	info(f"Старт обучения на {test_epochs} эпох...")
	global_step = 0
	best_f1 = 0.0

	for epoch in range(test_epochs):

		# Динамический порог классификации
		threshold = 0.3  # Начинаем с более низкого порога
		if epoch > 0 and f1 < 0.1:  # Если на предыдущей эпохе F1 был очень низким
			threshold = max(0.1, threshold * 0.9)  # Понижаем порог
			info(f"Понижение порога классификации до {threshold:.2f}")
		# ------ ФАЗА ОБУЧЕНИЯ ------
		model.train()
		total_train_loss = 0
		progress_bar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{test_epochs} [Обучение]")

		for step, batch in enumerate(progress_bar):
			# Подготовка батча
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)

			# Подготовка меток (уже в one-hot формате)
			labels = batch['labels'].to(device)

			# Forward pass с mixed precision
			with autocast(device_type=device.type, enabled=not FP32):
				try:
					outputs = model(
						input_ids=input_ids,
						attention_mask=attention_mask,
						labels=labels
					)
				except RuntimeError as e:
					if "out of memory" in str(e):
						error("Недостаточно VRAM. Попробуйте уменьшить batch_size или max_len.")
						sys.exit(1)
					else:
						raise e
				loss = outputs.loss
				loss = loss / ACCUMULATION_STEPS

			# Backward pass
			scaler.scale(loss).backward()

			# --- ПРОВЕРКА NaN/Inf ---
			has_problem = False
			problem_params = []

			for name, param in model.named_parameters():
				if param.grad is not None:
					has_nan = torch.isnan(param.grad).any().item()
					has_inf = torch.isinf(param.grad).any().item()

					if has_nan or has_inf:
						problem_params.append(name)
						if has_nan:
							error(f"NaN в градиентах: {name}")
						if has_inf:
							error(f"Inf в градиентах: {name}")
						has_problem = True

			if has_problem:
				error(f"Прерывание: некорректные градиенты в {len(problem_params)} параметрах")
				sys.exit(1)

			# --- КОНЕЦ ПРОВЕРКИ ---


			if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

				# Мониторинг градиентов (только для отладки)
				if global_step % 50 == 0:
					grad_norm = 0.0
					for p in model.parameters():
						if p.grad is not None:
							param_norm = p.grad.data.norm(2)
							grad_norm += param_norm.item() ** 2
					grad_norm = grad_norm ** 0.5
					writer.add_scalar("train/grad_norm", grad_norm, global_step)

					# Если градиенты взрываются (>10) - предупреждение
					if grad_norm > 10.0:
						warning(f"Большая норма градиентов: {grad_norm:.2f}. Возможно, нужен меньший LR.")

				scaler.step(optimizer)
				scaler.update()
				optimizer.zero_grad()
				scheduler.step()
				global_step += 1

			# Логирование
			total_train_loss += loss.item() * ACCUMULATION_STEPS
			if global_step % 10 == 0:
				writer.add_scalar("train/loss", loss.item() * ACCUMULATION_STEPS, global_step)
				writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

			# Логирование LR в консоль каждые 50 шагов
			if global_step % 50 == 0:
				current_lr = scheduler.get_last_lr()[0]

			# writer.add_scalar("train/learning_rate", current_lr, global_step) # Уже есть в TensorBoard
				if global_step % 200 == 0:  # Реже в консоль, чтобы не засорять
					info(f"Шаг {global_step}: LR = {current_lr:.2e}")

			progress_bar.set_postfix({"loss": f"{loss.item() * ACCUMULATION_STEPS:.4f}"})

		avg_train_loss = total_train_loss / len(train_loader)
		info(f"Эпоха {epoch+1}: Средняя тренировочная loss = {avg_train_loss:.4f}")

		# ДИАГНОСТИКА КАЖДЫЕ 100 ШАГОВ
		if step % 100 == 0:
			with torch.no_grad():
				probs = torch.sigmoid(outputs.logits)

				# 1. Статистика предсказаний
				avg_prob = probs.mean().item()
				min_prob = probs.min().item()
				max_prob = probs.max().item()

				# 2. Статистика меток
				avg_label = labels.mean().item()  # Какая доля меток = 1?
				positive_labels = (labels == 1).sum().item()
				total_labels = labels.numel()

				# 3. Проверка одного примера
				if step == 0:
					print(f"\n{Fore.YELLOW}ПЕРВЫЙ БАТЧ:{Style.RESET_ALL}")
					print(f"Метки (первые 5 классов в первом примере): {labels[0, :5].cpu().numpy()}")
					print(f"Предсказания (первые 5 классов): {probs[0, :5].cpu().numpy()}")

			print(f"Шаг {step}: loss={loss.item() * ACCUMULATION_STEPS:.6f}, avg_prob={avg_prob:.4f}, "
				  f"активные метки={positive_labels}/{total_labels} ({avg_label*100:.1f}%)")

			# КРИТИЧЕСКИЕ ПРОВЕРКИ
			if avg_prob < 0.01 or avg_prob > 0.99:
				error(f"Вероятности экстремальны! avg_prob={avg_prob:.4f}")
				warning("Проверь: 1) Инициализацию модели 2) Данные 3) Функцию потерь")

			if avg_label < 0.01:
				warning(f"Очень мало активных меток: {avg_label*100:.1f}%")
				warning("Возможно, метки состоят почти из нулей (класс 'нейтральный' доминирует)")

		# ------ ФАЗА ВАЛИДАЦИИ ------
		model.eval()
		total_val_loss = 0
		all_preds = []
		all_labels = []

		with torch.no_grad():
			# Сравниваем loss от модели и ручной расчет
			loss_fn_manual = nn.BCEWithLogitsLoss()
			manual_loss = loss_fn_manual(outputs.logits, labels)
			model_loss = outputs.loss

			if abs(model_loss.item() - manual_loss.item()) > 0.001:
				warning(f"Расхождение в loss: model={model_loss.item():.6f}, manual={manual_loss.item():.6f}")
			val_progress = tqdm(val_loader, desc=f"Эпоха {epoch+1}/{test_epochs} [Валидация]")
			for batch in val_progress:
				input_ids = batch['input_ids'].to(device)
				attention_mask = batch['attention_mask'].to(device)
				labels = batch['labels'].to(device)

				with autocast(device_type=device.type, enabled=not FP32):
					outputs = model(
						input_ids=input_ids,
						attention_mask=attention_mask,
						labels=labels
					)
					loss = outputs.loss

				total_val_loss += loss.item()

				# Собираем предсказания для метрик
				logits = outputs.logits
				probs = torch.sigmoid(logits)
				all_preds.append(probs.cpu())
				all_labels.append(labels.cpu())


		# Вычисление метрик
		avg_val_loss = total_val_loss / len(val_loader)
		all_preds = torch.cat(all_preds)
		all_labels = torch.cat(all_labels)

		# Вычисляем accuracy
		preds_binary = (all_preds > threshold).float()
		correct = (preds_binary == all_labels).sum().item()
		total = all_labels.numel()
		accuracy = correct / total

		# Вычисляем F1-score
		from sklearn.metrics import f1_score
		f1 = f1_score(all_labels.numpy(), preds_binary.numpy(), average='micro', zero_division=0)

		info(f"Валидация | Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.4f} | F1-micro: {f1:.4f}")
		writer.add_scalar("val/loss", avg_val_loss, epoch)
		writer.add_scalar("val/accuracy", accuracy, epoch)
		writer.add_scalar("val/f1", f1, epoch)

		# ------ СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ ------
		if f1 > best_f1:
			best_f1 = f1
			model.save_pretrained(OUTPUT_DIR)
			tokenizer.save_pretrained(OUTPUT_DIR)
			success(f"Модель сохранена (новый лучший F1: {f1:.4f})!")

		# Сохранение чекпоинта
		if CHECKPOINTS_DIR:
			checkpoint_dir = Path(CHECKPOINTS_DIR) / f"epoch_{epoch+1}"
			checkpoint_dir.mkdir(parents=True, exist_ok=True)
			model.save_pretrained(checkpoint_dir)
			tokenizer.save_pretrained(checkpoint_dir)

		torch.cuda.empty_cache()

	# ========== ФИНАЛИЗАЦИЯ ==========
	writer.close()

	if not test_mode:
		success(f"Обучение завершено! Лучшая модель сохранена в {OUTPUT_DIR}")
		print(f"\n{Fore.GREEN}{Style.BRIGHT}=== РЕЗУЛЬТАТЫ ==={Style.RESET_ALL}")
		print(f"Лучший F1-score: {best_f1:.4f}")
		print(f"Финальная validation loss: {avg_val_loss:.4f}")
		print(f"\n{Fore.CYAN}Следующие шаги:{Style.RESET_ALL}")
		print(f"  1. Проверьте логи TensorBoard: tensorboard --logdir={LOG_DIR}")
		print(f"  2. Финальная модель сохранена в: {OUTPUT_DIR}")
		print(f"  3. Используйте модель для предсказаний или дообучения")

		return {
			"best_f1": best_f1,
			"final_val_loss": avg_val_loss,
			"final_val_accuracy": accuracy
		}
	else:
		success("Тестовый режим завершён!")
		print(f"\n{Fore.YELLOW}Совет: Если метрики выглядят разумно, запустите полное обучение.{Style.RESET_ALL}")

		return {
			"test_f1": f1,
			"test_loss": avg_val_loss,
			"test_accuracy": accuracy
		}

def start_train():
	try:
		# Показать заголовок
		#clear_screen()
		header("ОБУЧЕНИЕ МОДЕЛИ КЛАССИФИКАЦИИ ЭМОЦИЙ")

		# Проверить наличие данных
		data_path = Path(config['data_dir']) / "ru_goemotions_metadata.pkl"
		if not data_path.exists():
			error(f"Файл с данными не найден: {data_path}")
			print(f"\n{Fore.YELLOW}Сначала подготовьте данные командой:{Style.RESET_ALL}")
			print(f"  python data.py")
			sys.exit(1)

		# Показать меню выбора режима
		test_mode, test_size = training_menu()

		if test_mode is None:
			print("Выход...")
			sys.exit(0)

		# Запуск обучения
		start_time = time.time()
		metrics = train(test_mode=test_mode, test_sample_size=test_size)

		# Итоговое сообщение
		total_time = time.time() - start_time
		print(f"\n{Fore.GREEN}Общее время выполнения: {total_time:.1f} сек{Style.RESET_ALL}")

	except KeyboardInterrupt:
		error("\nОбучение прервано пользователем")
		sys.exit(1)
	except Exception as e:
		error(f"Критическая ошибка: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)

if __name__ == "__main__":
	start_train()
