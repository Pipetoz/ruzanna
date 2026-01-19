# data.py
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

import pickle
if ma: print(ts(st, "pickle"))

import random
if ma: print(ts(st, "random"))

from collections import Counter
if ma: print(ts(st, "collections"))

from typing import Any, Dict, List, Tuple
if ma: print(ts(st, "typing"))

from pathlib import Path
if ma: print(ts(st, "pathlib"))

import yaml
if ma: print(ts(st, "yaml"))

from config import clear_screen, config, error, header, info, progress_bar, success, title, warning, rulables
if ma: print(ts(st, "config"))

import pandas as pd
if ma: print(ts(st, "pandas"))

if ma: print("\n" + ts(tot, "Общее время импортов") + "\n")


def load_data(data_dir: str, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Загружает train/validation для simplified."""
	train_file = Path(data_dir) / f"ru-go-emotions-{split}-train.csv"
	val_file = Path(data_dir) / f"ru-go-emotions-validation.csv"
	test_file = Path(data_dir) / f"ru-go-emotions-test.csv"

	for f in [train_file, val_file, test_file]:
		if not f.exists():
			raise FileNotFoundError(f"Файл не найден: {f}")

	return pd.read_csv(train_file), pd.read_csv(val_file), pd.read_csv(test_file)

def load_raw_data(data_dir: str, split: str) -> pd.DataFrame:
	"""Загружает raw‑датасет из одного CSV."""
	raw_file = Path(data_dir) / f"ru-go-emotions-{split}-train.csv"
	if not raw_file.exists():
		raise FileNotFoundError(f"Файл не найден: {raw_file}")
	return pd.read_csv(raw_file)

def load_simplified_data(data_dir: str, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Загружает train/validation для simplified."""
	train_file = Path(data_dir) / f"ru-go-emotions-{split}-train.csv"
	val_file = Path(data_dir) / f"ru-go-emotions-validation.csv"
	test_file = Path(data_dir) / f"ru-go-emotions-test.csv"

	for f in [train_file, val_file, test_file]:
		if not f.exists():
			raise FileNotFoundError(f"Файл не найден: {f}")

	return pd.read_csv(train_file), pd.read_csv(val_file), pd.read_csv(test_file)

def preprocess_texts(texts: List[str], min_len: int, max_len: int) -> List[str]:
	"""Очищает тексты и фильтрует по длине."""
	cleaned = [clean_text(t) for t in texts]
	filtered = [t for t in cleaned if min_len <= len(t) <= max_len]
	return filtered

def split_data(
	texts: List[str],
	labels: List[List[int]],
	ids: List[str],
	stratify_labels: List[str]
) -> Dict[str, Dict[str, List]]:
	"""Разбивает данные на train/val/test со стратификацией."""
	# Проверка на пустые данные
	from sklearn.model_selection import train_test_split

	if not texts:
		print(f"{Fore.RED}❌ Нет данных для разбиения{Style.RESET_ALL}")
		return {}

	# Проверка достаточности меток для стратификации
	if stratify_labels:
		unique_labels = set(stratify_labels)
		if len(unique_labels) < 2:
			print(f"{Fore.YELLOW}⚠ Недостаточно уникальных меток ({len(unique_labels)}). Отключаю стратификацию.{Style.RESET_ALL}")
			stratify_labels = None

	try:
		# Первый сплит: train vs temp
		X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
			texts, labels, ids,
			test_size=0.3,
			random_state=42,
			stratify=stratify_labels
		)
	except (ValueError, ZeroDivisionError) as e:
		error(f"Ошибка стратификации (1-й этап): {e}")
		warning(f"Переключаюсь на случайное разбиение")
		X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
			texts, labels, ids,
			test_size=0.3,
			random_state=42
		)

	try:
		# Второй сплит: val vs test
		stratify_temp = [str(l[0]) if l else "no_label" for l in y_temp] if stratify_labels else None
		if stratify_temp:
			unique_temp = set(stratify_temp)
			if len(unique_temp) < 2:
				stratify_temp = None

		X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
			X_temp, y_temp, ids_temp,
			test_size=0.5,
			random_state=42,
			stratify=stratify_temp
		)
	except (ValueError, ZeroDivisionError):
		warning(f"Ошибка стратификации (2-й этап). Отключаю стратификацию...")
		X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
			X_temp, y_temp, ids_temp,
			test_size=0.5,
			random_state=42
		)

	return {
		"train": {"texts": X_train, "labels": y_train, "ids": ids_train},
		"val": {"texts": X_val, "labels": y_val, "ids": ids_val},
		"test": {"texts": X_test, "labels": y_test, "ids": ids_test}
	}

def validate_processed_data(processed_data: Dict[str, Dict[str, List]], label2id: Dict[str, int]) -> bool:
	"""Валидирует структуру processed_data."""
	progress_bar(3, "ℹ  Валидация разбиения")

	# 1. Жёсткая проверка входных данных
	if not isinstance(processed_data, dict):
		print(f"{Fore.RED}❌ data не является словарём (тип: {type(data)}){Style.RESET_ALL}")
		return False

	if len(processed_data) == 0:
		print(f"{Fore.RED}❌ data пуст!{Style.RESET_ALL}")
		return False

	num_classes = len(label2id)

	try:
		# 2. Проходим по всем сплитам
		for split_name, split in processed_data.items():
			# 2.1. Проверка ключей
			expected_keys = ["texts", "labels", "ids"]
			missing = [k for k in expected_keys if k not in split]
			if missing:
				print(
					f"{Fore.RED}❌ {split_name}: "
					f"Отсутствуют ключи: {missing}{Style.RESET_ALL}"
				)
				return False

			texts, labels, ids = split["texts"], split["labels"], split["ids"]

			# 2.2. Проверка длин
			if len(texts) != len(labels) or len(texts) != len(ids):
				print(
					f"{Fore.RED}❌ {split_name}: "
					f"Несоответствие длин: texts={len(texts)}, "
					f"labels={len(labels)}, ids={len(ids)}{Style.RESET_ALL}"
				)
				return False

			# 2.3. Проверка текстов
			for i, text in enumerate(texts):
				if not isinstance(text, str) or not text.strip():
					print(
						f"{Fore.RED}❌ {split_name}: "
						f"Некорректный текст в позиции {i}: {text!r}{Style.RESET_ALL}"
					)
					return False

			# 2.4. Проверка меток
			for i, label_list in enumerate(labels):
				if not isinstance(label_list, list):
					print(
						f"{Fore.RED}❌ {split_name}: "
						f"Метка не является списком в позиции {i}: {label_list!r}{Style.RESET_ALL}"
					)
					return False
				for label in label_list:
					if not isinstance(label, int):
						print(
							f"{Fore.RED}❌ {split_name}: "
							f"Нецелочисленная метка в позиции {i}: {label!r}{Style.RESET_ALL}"
						)
						return False
					if label < 0 or label >= num_classes:
						print(
							f"{Fore.RED}❌ {split_name}: "
							f"Метка вне диапазона [0, {num_classes-1}] "
							f"в позиции {i}: {label}{Style.RESET_ALL}"
						)
						return False

	except Exception as e:
		# Если ошибка возникла до определения split_name (например, data.items() вызвал исключение)
		if 'split_name' not in locals():
			print(f"{Fore.RED}❌ Критическая ошибка валидации: {e}{Style.RESET_ALL}")
		else:
			print(f"{Fore.RED}❌ Ошибка валидации для {split_name}: {e}{Style.RESET_ALL}")
		return False

	success("Все разбиения валидны.")
	return True

def save_metadata(meta_data: Dict, output_path: str):
	"""Сохраняет метаданные в YAML."""
	with open(output_path, 'w', encoding='utf-8') as f:
		yaml.dump(meta_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

def save_processed_data(processed_data: Dict, pkl_path: str):
	"""Сохраняет processed_data в .pkl."""
	if processed_data:
		with open(pkl_path, 'wb') as f:
			pickle.dump(processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)

def parse_labels(label_str: str) -> List[int]:
	cleaned = label_str.strip('[]').strip()
	if not cleaned:
		return []
	return [int(x) for x in cleaned.split() if x.strip()]


def clean_text(text: Any) -> str:
	"""Базовая очистка текста. Гарантирует возврат непустой строки."""
	if pd.isna(text) or text is None:
		return ""
	text = str(text).strip()
	if not text:
		return ""
	import re
	text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
	text = re.sub(r'[^\w\s]', ' ', text)
	text = re.sub(r'\s+', ' ', text).strip()
	return text.lower()



def data_menu():
	header("Система создания метаданных!")
	isTestMode = False
	test_sample_size = None

	choice = input(f"{Style.BRIGHT}Введите 1, {Fore.YELLOW}тестовый режим{Fore.RESET}\n"
				  f"\nВведите 2, {Fore.YELLOW}полный режим обучения{Fore.RESET}\n\n"
				  f"{Fore.MAGENTA}или Enter возврата в меню: {Style.RESET_ALL}")
	print()
	if not choice:
		print()
		warning("Выход...")
		isTestMode = None
		return isTestMode, test_sample_size

	if choice.isdigit() and 1 <= int(choice) <= 2:
		if choice == "1":
			isTestMode = True
			while True:
				try:
					test_sample_size = int(input("Выберите процент от нейтральных эмоций: "))
					if test_sample_size > 0:
						break
					else:
						print("Число должно быть > 0")
				except ValueError:
					print("Введите целое число!")
			print()
			info(f"Выбран тестовый датасет с {test_sample_size}% нейтральных эмоций")
			return isTestMode, test_sample_size
		elif choice == "2":
			isTestMode = False
			while True:
				try:
					test_sample_size = int(input("Выберите процент от нейтральных эмоций: "))
					if test_sample_size > 0:
						break
					else:
						print("Число должно быть > 0")
				except ValueError:
					print("Введите целое число!")
			print()
			info(f"Выбран полный датасет с {test_sample_size}% нейтральных эмоций")
			return isTestMode, test_sample_size
	else:
		error("Неверный ввод!")
		data_menu()

def data_start():
	isTestMode, test_sample_size = data_menu()
	if isTestMode == None:
		from cli import cli
		cli()
		return
	elif isTestMode:
		split = "simplified"
	elif isTestMode == False:
		split = "raw"

	dataset_dir = Path(config['dataset_dir']) / "dataset"
	output_path = Path(config['data_dir']) / "ru_goemotions_metadata.yaml"
	pkl_path = Path(config['data_dir']) / "ru_goemotions_metadata.pkl"

	info("Проверка параметров сплит-разбиения")

	try:
		progress_bar(3, f"✓  Загрузка данных")

		train_df, val_df, test_df = load_data(dataset_dir, split)

		texts = []
		labels = []
		ids = []

		if train_df is not None:
			texts += train_df['ru_text'].tolist()
			labels += train_df['labels'].apply(parse_labels).tolist()
			ids += train_df['id'].tolist()

		if val_df is not None:
			texts += val_df['ru_text'].tolist()
			labels += val_df['labels'].apply(parse_labels).tolist()
			ids += val_df['id'].tolist()

		if test_df is not None:
			texts += test_df['ru_text'].tolist()
			labels += test_df['labels'].apply(parse_labels).tolist()
			ids += test_df['id'].tolist()

		# Предобработка текстов
		cleaned_texts = preprocess_texts(texts, min_len=5, max_len=500)

		# Прогресс‑бар: очистка завершена
		progress_bar(3, "✓  Очистка текстов")

		# Фильтрация по длине
		filtered_data = [
		(t, l, i) for t, l, i in zip(cleaned_texts, labels, ids)
		if 5 <= len(t) <= 500
		]

		if not filtered_data:
			print(f"{Fore.RED}❌ Все тексты отфильтрованы по длине (осталось 0 примеров){Style.RESET_ALL}")
			return

		texts_filtered, labels_filtered, ids_filtered = zip(*filtered_data)

		#print(labels_filtered)

		# До ограничения
		print(f"{Fore.YELLOW}До ограничения...{Fore.RESET}")
		print(f"Общее число: {len(labels_filtered)}")
		print(f"Число нейтральных: {Fore.YELLOW}{sum(1 for i in labels_filtered if i == [27])}{Fore.RESET}\n")

		# Ограничение нейтральных
		neutral_indices = [i for i, id_value in enumerate(labels_filtered) if id_value == [27]]
		total_neutral = len(neutral_indices)
		neutral_sample_count = int(total_neutral * test_sample_size / 100)
		if neutral_sample_count > total_neutral:
			neutral_sample_count = total_neutral
		selected_neutral_indices = set(random.sample(neutral_indices, neutral_sample_count))

		texts_new = []
		labels_new = []
		ids_new = []

		for i in range(len(texts_filtered)):
			if labels_filtered[i] == [27]:
				if i in selected_neutral_indices:
					texts_new.append(texts_filtered[i])
					labels_new.append(labels_filtered[i])
					ids_new.append(ids_filtered[i])
				# иначе пропускаем
			else:
				texts_new.append(texts_filtered[i])
				labels_new.append(labels_filtered[i])
				ids_new.append(ids_filtered[i])

		texts_filtered, labels_filtered, ids_filtered = texts_new, labels_new, ids_new

		progress_bar(3, f"✓  Ограничение нейтральных")
		# После ограничения
		print(f"{Fore.YELLOW}После ограничения...{Fore.RESET}")
		print(f"Общее число: {len(labels_filtered)}")
		print(f"Число нейтральных: {Fore.YELLOW}{sum(1 for i in labels_filtered if i == [27])}{Fore.RESET}\n")

		# Подготовка меток для стратификации
		stratify_labels = [str(l[0]) if l else "no_label" for l in labels_filtered]

		unique_stratify = set(stratify_labels)

		# Проверка на достаточность меток для стратификации
		if len(unique_stratify) < 2:
			print(f"{Fore.RED}❌ Недостаточно уникальных меток для стратификации: {unique_stratify}{Style.RESET_ALL}")
			print(f"{Fore.YELLOW}⚠ Переключаюсь на случайное разбиение без стратификации{Style.RESET_ALL}")
			stratify_labels = None  # Отключаем стратификацию

		# Разбиение на train/val/test
		progress_bar(3, "ℹ  Разбиение данных")
		processed_data = split_data(
			texts_filtered,
			labels_filtered,
			ids_filtered,
			stratify_labels
		)

		# Проверка результата разбиения
		if not processed_data:
			print(f"{Fore.RED}❌ Разбиение не выполнено{Style.RESET_ALL}")
			return

		# Валидация
		label2id = {name: idx for idx, name in enumerate(rulables().keys())}
		if not validate_processed_data(processed_data, label2id):
			return
		id2label = {idx: name for name, idx in label2id.items()}
		info("Загрузка токенизатора")
		from transformers import AutoTokenizer
		tokenizer = AutoTokenizer.from_pretrained(config['source_model_dir'])

		# Получаем только сплиты с данными
		data_splits_only = {k: v for k, v in processed_data.items()
							if isinstance(v, dict) and "texts" in v}

		# Обновляем processed_data
		processed_data.update({
			"vectorizer": tokenizer,
			"label2id": label2id,
			"id2label": id2label
		})

		# Сохранение метаданных и данных
		meta_data = {
			"class_names": list(rulables().keys()),
			"label2id": label2id,
			"splits": {
				split_name: {"size": len(split["texts"])}
				for split_name, split in data_splits_only.items()
			},
			"split_type": split,
			"total_examples": sum(len(split["texts"]) for split in data_splits_only.values())
		}

		print(f"Уникальные метки в данных: {set([l for sublist in labels_filtered for l in sublist])}\n")
		print(f"Максимальная метка: {max([l for sublist in labels_filtered for l in sublist])}\n")
		print(f"Количество классов: {len(rulables().keys())}\n")

		progress_bar(3, "ℹ  Сохранение результатов")
		save_metadata(meta_data, output_path)
		save_processed_data(processed_data, pkl_path)
		print(output_path)
		print(pkl_path)


		# Финальные сообщения
		print()
		success("Успешно!")
		print(f"  - Сплит: {split}")
		print(f"  - Всего примеров: {meta_data['total_examples']}\n")

		# Проверка на достаточность меток
		if stratify_labels and len(unique_stratify) < 2:
			print(f"{Fore.RED}❌ Недостаточно уникальных меток для стратификации: {unique_stratify}{Style.RESET_ALL}")
			print(f"{Fore.YELLOW}⚠ Переключаюсь на случайное разбиение без стратификации{Style.RESET_ALL}")
			stratify_labels = None

	except FileNotFoundError as e:
		error(f"Ошибка загрузки файла: {e}" + "\n")
	#except Exception as e:
	#	error(f"Непредвиденная ошибка: {e}" + "\n")
	finally:
		info("Работа завершена.")
	from cli import ill_be_back
	ill_be_back()

if __name__ == "__main__":
	data_start()

