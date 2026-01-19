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

from sklearn.preprocessing import MultiLabelBinarizer
if ma: print(ts(st, "sklearn"))

import torch
if ma: print(ts(st, "torch"))

import torch.nn as nn
if ma: print(ts(st, "torch.nn"))

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

MODEL_NAME = config['source_model_dir']
MAX_LEN = 512
BATCH_SIZE = 2
ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-05
EPOCHS = 3
WARMUP_STEPS = 89
WEIGHT_DECAY = 0.03
FP32 = True  # False для AMP
USE_TRITON = True
CHECKPOINTS_DIR = config['checks_dir']
LOG_DIR = config['logs_dir']
OUTPUT_DIR = config['final_model_dir']

# Функция для подавления вывода
def suppress_output(func):
	def wrapper(*args, **kwargs):
		null_device = os.devnull  # Автоматически выберет 'NUL' или '/dev/null'
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

class EmotionsDataset(Dataset):
	def __init__(self, texts, labels, tokenizer, max_len):
		self.texts = texts
		self.labels = labels
		self.tokenizer = tokenizer
		self.max_len = max_len

	def __len__(self):
		return len(self.texts)


	def __getitem__(self, item):
		text = str(self.texts[item])
		label = self.labels[item]

		encoding = self.tokenizer(
			text,
			truncation=True,
			padding='max_length',
			max_length=self.max_len,
			return_tensors='pt',
			return_attention_mask=True
		)

		return {
			'input_ids': encoding['input_ids'][0],
			'attention_mask': encoding['attention_mask'][0],
			'labels': torch.tensor(label, dtype=torch.float)
		}


def train():
	# Устройство
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device:
		success(f"Будет использоваться: {device}")
	else:
		warning(f"Будет использоваться: {device}")

	# Токенизатор и модель
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

	# Загрузка данных из .pkl
	data_path = Path(config['data_dir']) / "ru_goemotions_metadata.pkl"
	with open(data_path, "rb") as f:
		processed_data = pickle.load(f)

	info(f"Ключи в данных: {processed_data.keys()}")

	# Проверка обязательных ключей
	required_keys = ["train", "val", "vectorizer", "label2id", "id2label"]
	missing_keys = [k for k in required_keys if k not in processed_data]
	if missing_keys:
		print(f"⚠ Отсутствующие ключи: {missing_keys}")
		# Создаем недостающие ключи на лету
		if "label2id" not in processed_data:
		   label2id = {name: idx for idx, name in enumerate(rulables().keys())}
		   processed_data["label2id"] = label2id
		if "id2label" not in processed_data:
		   processed_data["id2label"] = {v: k for k, v in processed_data["label2id"].items()}
	assert all(k in processed_data for k in required_keys), "Не все ключи присутствуют в .pkl"


	# Извлечение текстов и меток
	train_texts = processed_data["train"]["texts"]
	train_labels = processed_data["train"]["labels"]
	val_texts = processed_data["val"]["texts"]
	val_labels = processed_data["val"]["labels"]

	# Получаем количество классов из label2id
	num_classes = len(processed_data["label2id"])
	print(f"Количество классов: {num_classes}")

	print(f"Обучающие примеры: {len(train_texts)}")
	print(f"Валидационные примеры: {len(val_texts)}")

	print("Загрузка модели RoBERTa...", end="", flush=True)

	# Временно подавляем предупреждения transformers
	import warnings
	from transformers import logging as transformers_logging

	# Сохраняем текущий уровень
	old_verbosity = transformers_logging.get_verbosity()
	transformers_logging.set_verbosity_error()

	# Игнорируем конкретное предупреждение
	warnings.filterwarnings("ignore",
		message="Some weights of RobertaForSequenceClassification were not initialized")

	try:
		model = AutoModelForSequenceClassification.from_pretrained(
			MODEL_NAME,
			num_labels=num_classes
		)

		print(f"\r{Fore.GREEN}✓ Модель загружена{Style.RESET_ALL}")

		# Показываем информацию о модели
		success(f"Классификатор настроен на {num_classes} классов")

	finally:
		# Восстанавливаем настройки
		transformers_logging.set_verbosity(old_verbosity)
		warnings.resetwarnings()
	model.to(device, non_blocking=True)

	# Компиляция с Triton (опционально)
	if USE_TRITON and torch.cuda.is_available():
		torch.set_float32_matmul_precision('high')
		model = compile_model(model)
		print("Triton включён (inductor + default)")

	# Используем правильное количество классов
	mlb = MultiLabelBinarizer(classes=list(range(num_classes)))
	train_labels_onehot = mlb.fit_transform(train_labels)
	val_labels_onehot = mlb.transform(val_labels)

	# Создание датасетов
	train_dataset = EmotionsDataset(train_texts, train_labels_onehot, tokenizer, MAX_LEN)
	val_dataset = EmotionsDataset(val_texts, val_labels_onehot, tokenizer, MAX_LEN)

	# Создание даталоадеров
	train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

	 # Создание оптимизатора
	optimizer = AdamW(
		model.parameters(),
		lr=LEARNING_RATE,
		weight_decay=WEIGHT_DECAY
	)

	# 7. Расчёт шагов для scheduler
	total_batches = len(train_dataloader)
	total_optimizer_steps = math.ceil(total_batches / ACCUMULATION_STEPS) * EPOCHS

	# Если нужно учесть неполные шаги (опционально)
	if total_batches % ACCUMULATION_STEPS != 0:
		total_optimizer_steps += EPOCHS  # +1 шаг на эпоху для остатка

	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=WARMUP_STEPS,
		num_training_steps=total_optimizer_steps
	)

	# Скалер для AMP
	scaler = GradScaler()

	# Логирование
	os.makedirs(LOG_DIR, exist_ok=True)
	writer = SummaryWriter(LOG_DIR)
	log_file = os.path.join(LOG_DIR, "train_log.txt")
	with open(log_file, "w", encoding="utf-8") as f:
		f.write("epoch,step,train_loss,val_loss,val_acc,lr\n")


	best_val_acc = 0.0


	# Цикл обучения
	for epoch in range(EPOCHS):
		model.train()
		epoch_loss = 0.0
		optimizer.zero_grad()

		for step, batch in enumerate(tqdm(train_dataloader, desc=f"Эпоха {epoch+1}/{EPOCHS}")):
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)

			# Прямой проход
			with autocast(device_type='cuda', enabled=not FP32):
				outputs = model(
					input_ids,
					attention_mask=attention_mask,
					labels=labels
				)
				loss = outputs.loss / ACCUMULATION_STEPS

			# Обратное распространение
			scaler.scale(loss).backward()

			if (step + 1) % ACCUMULATION_STEPS == 0:
				scaler.step(optimizer)
				scaler.update()
				scheduler.step()
				optimizer.zero_grad()

			epoch_loss += loss.item() * ACCUMULATION_STEPS

			# Логи в TensorBoard
			writer.add_scalar(
				'Loss/train',
				loss.item() * ACCUMULATION_STEPS,
				epoch * len(train_dataloader) + step
			)
			writer.add_scalar(
				'LR',
				scheduler.get_last_lr()[0],
				epoch * len(train_dataloader) + step
			)

		# Валидация
		model.eval()
		val_loss = 0.0
		correct = 0
		total = 0

		with torch.no_grad():
			for batch in val_dataloader:
				input_ids = batch['input_ids'].to(device)
				attention_mask = batch['attention_mask'].to(device)
				labels = batch['labels'].to(device)

				with autocast(device_type='cuda', enabled=not FP32):
					outputs = model(
						input_ids,
						attention_mask=attention_mask,
						labels=labels
					)
					val_loss += outputs.loss.item()

				preds = torch.argmax(outputs.logits, dim=1)
				true_labels = torch.argmax(labels, dim=1)
				correct += (preds == labels).sum().item()
				total += labels.size(0)

		val_acc = correct / total
		val_loss /= len(val_dataloader)

		# Сохранение лучшей модели
		if val_acc > best_val_acc:
			best_val_acc = val_acc
			model.save_pretrained(f"{CHECKPOINTS_DIR}")
			tokenizer.save_pretrained(f"{CHECKPOINTS_DIR}")
			print(f"Сохранён лучший чекпоинт с acc={val_acc:.4f}")

		# Логирование
		writer.add_scalar('Loss/val', val_loss, epoch)
		writer.add_scalar('Accuracy/val', val_acc, epoch)


		with open(log_file, "a", encoding="utf-8") as f:
			f.write(
				f"{epoch},{step},"
				f"{epoch_loss/len(train_dataloader):.4f},"
				f"{val_loss:.4f},{val_acc:.4f},"
				f"{scheduler.get_last_lr()[0]:.2e}\n"
			)

		print(
			f"Эпоха {epoch + 1}: "
			f"Train Loss = {epoch_loss / len(train_dataloader):.4f}, "
			f"Val Loss = {val_loss:.4f}, "
			f"Val Acc = {val_acc:.4f}, "
			f"LR = {scheduler.get_last_lr()[0]:.2e}"
		)

	# Закрытие writer после завершения обучения
	writer.close()
	print(f"Обучение завершено. Лучшая точность на валидации: {best_val_acc:.4f}")

	# Сохранение финальной модели
	model.save_pretrained(OUTPUT_DIR)
	tokenizer.save_pretrained(OUTPUT_DIR)
	print(f"Финальная модель сохранена в {OUTPUT_DIR}")

if __name__ == "__main__":
	train()
