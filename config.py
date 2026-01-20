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

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, get_linear_schedule_with_warmup
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
if isinstance(FP32, str):
	FP32 = FP32.lower() == 'true'
USE_TRITON = config['use_triton']
if isinstance(USE_TRITON, str):
	USE_TRITON = USE_TRITON.lower() == 'true'




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
		# Метка уже в формате многометочного бинарного вектора
		label = torch.tensor(self.labels[item], dtype=torch.float)

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
			'labels': label  # Отправляем вектор, а не индекс
		}

# ========== ФУНКЦИИ ДЛЯ ВЫЧИСЛЕНИЯ МЕТРИК ==========
def multi_label_metrics(predictions, labels, threshold=0.5):
	"""
	Вычисление метрик для многометочной классификации.

	Args:
		predictions: логиты модели (numpy array)
		labels: истинные метки (numpy array)
		threshold: порог для бинаризации предсказаний

	Returns:
		dict: словарь с метриками
	"""
	# Преобразуем логиты в вероятности с помощью сигмоиды
	sigmoid = torch.nn.Sigmoid()
	probs = sigmoid(torch.tensor(predictions))

	# Бинаризируем предсказания по порогу
	preds = (probs > threshold).float()

	# Преобразуем labels в тензор
	labels_tensor = torch.tensor(labels)

	# Вычисляем accuracy (доля правильных предсказаний по всем меткам)
	correct = (preds == labels_tensor).sum().item()
	total = labels_tensor.numel()
	accuracy = correct / total

	# Вычисляем precision, recall, f1 для каждого класса (macro averaging)
	from sklearn.metrics import precision_score, recall_score, f1_score
	# Нужно преобразовать в формат для sklearn (убираем batch dimension)
	preds_np = preds.numpy()
	labels_np = labels

	# Для многометочной классификации используем 'samples' или 'micro'
	try:
		precision = precision_score(labels_np, preds_np, average='micro', zero_division=0)
		recall = recall_score(labels_np, preds_np, average='micro', zero_division=0)
		f1 = f1_score(labels_np, preds_np, average='micro', zero_division=0)
	except:
		precision = recall = f1 = 0.0

	return {
		"accuracy": accuracy,
		"precision": precision,
		"recall": recall,
		"f1": f1
	}

def compute_metrics(eval_pred):
	"""
	Функция для Hugging Face Trainer.
	Преобразует вывод модели в метрики.
	"""
	logits, labels = eval_pred
	return multi_label_metrics(logits, labels)

# ========== ГЛАВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ ==========
def train(test_mode=False, test_sample_size=100):
	"""
	Основная функция обучения для многометочной классификации эмоций
	с поддержкой тестового режима и параметров из config.yaml

	Args:
		test_mode (bool): Если True, запускается тестовый режим
		test_sample_size (int): Количество примеров для теста
	"""

	# ========== НАСТРОЙКА ПАРАМЕТРОВ ==========
	print(f"{Fore.CYAN}{Style.BRIGHT}=== {'ТЕСТОВЫЙ РЕЖИМ' if test_mode else 'ОБУЧЕНИЕ'} ==={Style.RESET_ALL}")

	# Загрузка параметров из конфига (уже сделано глобально)
	# Используем глобальные константы, прочитанные из config
	if test_mode:
		# В тестовом режиме меняем некоторые параметры
		test_epochs = 1
		test_batch_size = min(2, BATCH_SIZE)
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

	with open(data_path, "rb") as f:
		processed_data = pickle.load(f)

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

	# ========== НАСТРОЙКА ОБУЧЕНИЯ ==========
	info("Настройка обучения...")

	# Создаем папки для выходных данных
	os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
	os.makedirs(LOG_DIR, exist_ok=True)
	os.makedirs(OUTPUT_DIR, exist_ok=True)

	# Настройки TrainingArguments
	training_args = TrainingArguments(
		output_dir=str(CHECKPOINTS_DIR),
		overwrite_output_dir=True,
		num_train_epochs=test_epochs,
		per_device_train_batch_size=test_batch_size,
		per_device_eval_batch_size=test_batch_size * 2,
		warmup_steps=WARMUP_STEPS,
		weight_decay=WEIGHT_DECAY,
		learning_rate=LEARNING_RATE,
		logging_dir=str(LOG_DIR),
		logging_strategy=test_logging_steps,
		eval_strategy="epoch",
		save_strategy="epoch" if not test_mode else "no",
		save_total_limit=2,
		load_best_model_at_end=not test_mode,
		metric_for_best_model="f1",
		greater_is_better=True,
		fp16=not FP32,
		gradient_accumulation_steps=ACCUMULATION_STEPS,
		report_to="tensorboard",
		dataloader_pin_memory=True if device.type == "cuda" else False,
		dataloader_num_workers=0,
		remove_unused_columns=False,
	)

	# ========== СОЗДАНИЕ TRAINER ==========
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=val_dataset,
		compute_metrics=compute_metrics,
		tokenizer=tokenizer,
	)

	# ========== ЗАПУСК ОБУЧЕНИЯ ==========
	success("Начало обучения...")

	print(f"{Fore.CYAN}Конфигурация:{Style.RESET_ALL}")
	print(f"  Режим: {'ТЕСТ' if test_mode else 'ОБУЧЕНИЕ'}")
	print(f"  Модель: {MODEL_NAME}")
	print(f"  Эпохи: {test_epochs}")
	print(f"  Примеров: {len(train_dataset)} обучающих, {len(val_dataset)} валидационных")
	print(f"  Batch size: {test_batch_size} × {ACCUMULATION_STEPS} = {test_batch_size * ACCUMULATION_STEPS}")
	print(f"  Learning rate: {LEARNING_RATE}")
	print(f"  Max length: {MAX_LEN}")
	print(f"  Mixed precision: {'выключен' if FP32 else 'включен'}")
	print(f"  Устройство: {device}")

	# Запуск обучения
	train_result = trainer.train()

	# ========== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ==========
	if not test_mode:
		info("Сохранение результатов...")
		trainer.save_model(str(OUTPUT_DIR))
		tokenizer.save_pretrained(str(OUTPUT_DIR))
		success(f"✓ Лучшая модель сохранена в: {OUTPUT_DIR}")

		# Сохраняем метрики
		metrics = train_result.metrics
		trainer.log_metrics("train", metrics)
		trainer.save_metrics("train", metrics)
		trainer.save_state()
	else:
		info("Тестовый режим: модель не сохраняется")

	# ========== ФИНАЛЬНАЯ ОЦЕНКА ==========
	info("Финальная оценка на валидационном наборе...")
	eval_metrics = trainer.evaluate()

	# ========== ВЫВОД РЕЗУЛЬТАТОВ ==========
	print(f"\n{Fore.GREEN}{Style.BRIGHT}=== РЕЗУЛЬТАТЫ ==={Style.RESET_ALL}")
	print(f"Accuracy: {eval_metrics.get('eval_accuracy', 0):.4f}")
	print(f"Precision: {eval_metrics.get('eval_precision', 0):.4f}")
	print(f"Recall: {eval_metrics.get('eval_recall', 0):.4f}")
	print(f"F1-score: {eval_metrics.get('eval_f1', 0):.4f}")
	print(f"Loss: {eval_metrics.get('eval_loss', 0):.4f}")
	print(f"Время обучения: {train_result.metrics.get('train_runtime', 0):.1f} сек")

	if test_mode:
		info("Тестовый режим завершён. Проверьте логи и метрики.")
		print(f"{Fore.YELLOW}Совет: Если метрики выглядят разумно, запустите полное обучение.{Style.RESET_ALL}")
	else:
		success("Обучение завершено успешно!")
		print(f"\n{Fore.CYAN}Следующие шаги:{Style.RESET_ALL}")
		print(f"  1. Проверьте логи TensorBoard: tensorboard --logdir={LOG_DIR}")
		print(f"  2. Финальная модель сохранена в: {OUTPUT_DIR}")
		print(f"  3. Используйте модель для предсказаний или дообучения")

	return eval_metrics

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
