# train.py
import math
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from colorama import Back, Fore, Style, init
from sklearn.preprocessing import MultiLabelBinarizer
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
						  Trainer, TrainingArguments,
						  get_linear_schedule_with_warmup)

from config import config

init()

#✅⚠️❌ℹ
from config import (clear_screen, error, header, info, progress_bar, success,
					title, warning)

MODEL_NAME = config['source_model_dir']
MAX_LEN = 512
BATCH_SIZE = 8
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
TASK_TYPE = "emotional_reactions"  # или "interpretations", "explorations"


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
			'attention_mask': encoding['attention_mask'],
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
	model = AutoModelForSequenceClassification.from_pretrained(
		MODEL_NAME,
		num_labels=28
	)
	model.to(device, non_blocking=True)

	# Компиляция с Triton (опционально)
	if USE_TRITON and torch.cuda.is_available():
		model = torch.compile(model, backend="inductor", mode="default")
		print("Triton включён (inductor + default)")

	# Загрузка данных из .pkl
	data_path = Path(config['data_dir']) / "ru_goemotions_metadata.pkl"
	with open(data_path, "rb") as f:
		processed_data = pickle.load(f)

	# Проверка обязательных ключей
	required_keys = ["train", "val", "vectorizer", "label2id", "id2label"]
	assert all(k in processed_data for k in required_keys), "Не все ключи присутствуют в .pkl"


	# Извлечение текстов и меток
	train_texts = processed_data["train"]["texts"]
	train_labels = processed_data["train"]["labels"]
	val_texts = processed_data["val"]["texts"]
	val_labels = processed_data["val"]["labels"]

	print(f"Обучающие примеры: {len(train_texts)}")
	print(f"Валидационные примеры: {len(val_texts)}")

	mlb = MultiLabelBinarizer(classes=list(range(27)))  # 27 классов
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
