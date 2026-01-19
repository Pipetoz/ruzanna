#cli.py
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

from config import (clear_screen, config, create_folders, error, header, info,
progress_bar, rainbow_text, success, title, warning, find_yaml_files, rulables)
if ma: print(ts(st, "config"))

import pickle
if ma: print(ts(st, "pickle"))

from typing import Any, Dict, List, Tuple
if ma: print(ts(st, "typing"))

import traceback
if ma: print(ts(st, "config"))

import sys
sys.stdout.reconfigure(encoding='utf-8')
if ma: print(ts(st, "sys"))

import yaml
if ma: print(ts(st, "yaml"))

from pathlib import Path
if ma: print(ts(st, "pathlib"))

from typing import Any, List
if ma: print(ts(st, "typing"))

if ma: print("\n" + ts(tot, "Общее время импортов") + "\n")

def ill_be_back():
	try:
		if not input(f"\n{Fore.MAGENTA} Enter, чтобы вернуться: {Style.RESET_ALL}\n"):
			cli()
	except Exception as e:
		error(e)
		traceback.print_exc()
		cli()

def colors():
	clear_screen()
	header(rainbow_text("З Д Е С Ь  Б Ы Л А  D E E P  S E E K 🌈"))
	print(f"{Style.BRIGHT}{Fore.RED}     <3 {Fore.CYAN}АлХиМиК КоДа"
		f"{Fore.YELLOW} + {Fore.GREEN}ПоЭт дАнНыХ{Fore.RED} <3{Style.RESET_ALL}\n")
	print(Fore.RED + "Красный текст")
	print(Fore.GREEN + "Зелёный текст")
	print(Fore.YELLOW + "Жёлтый текст")
	print(Fore.BLUE + "Синий текст")
	print(Fore.MAGENTA + "Пурпурный текст")
	print(Fore.CYAN + "Голубой текст")
	print(Fore.WHITE + "Белый текст")
	print(Fore.RESET)  # Сброс цвета

	print(Back.RED + "На красном фоне")
	print(Back.GREEN + "На зелёном фоне")
	print(Back.YELLOW + "На жёлтом фоне")
	print(Back.RESET)  # Сброс фона

	print(Style.DIM + "Тусклый текст")
	print(Style.NORMAL + "Обычный текст")
	print(Style.BRIGHT + "Яркий текст")
	print(Style.RESET_ALL)  # Полный сброс (цвет + стиль)
	ill_be_back()

def normalize_key(key):
	corrections = {
		"curiosity": "curiosity",
		"realization": "realization"
	}
	return corrections.get(key, key)

def extract_tables_from_meta(meta_data):
	tables = {}
	class_names = meta_data.get("class_names", [])
	label2id = meta_data.get("label2id", {})

	if class_names:
		headers = ["Эмоция", "ID"]
		data = []
		for name in class_names:
			rus_name = rulables().get(normalize_key(name), name)
			cid = label2id.get(name, "N/A")
			data.append([rus_name, cid])
		tables["classes"] = {"headers": headers, "data": data}

	# Таблица разбиений
	splits = meta_data.get("splits", {})
	if splits:
		# Заголовки: "Разбиение", "Размер", затем 0, 1, ..., 27
		headers = ["Разбиение", "Размер"] + [str(i) for i in range(28)]  # 30 столбцов

		data = []

		# Собираем строки для каждого разбиения
		for split_name, split_info in splits.items():
			size = split_info.get("size", 0)
			label_dist = {str(k): v for k, v in split_info.get("label_dist", {}).items()}  # к str


			row = [split_name, size]
			for lid in range(28):
				count = label_dist.get(str(lid), 0)
				row.append(count if count > 0 else "·")  # 0 → "·", иначе число
			data.append(row)

	# Таблица предобработки
	preproc = meta_data.get("preprocessing", {})
	if preproc:
		preproc_rows = [[k, str(v)] for k, v in preproc.items()]
		tables["preprocessing"] = {
			"headers": ["Параметр", "Значение"],
			"data": preproc_rows
		}

	return tables

def print_table(
	headers: List[str],
	data: List[List[Any]],
	title: str = None,
	align: str = "left",
	padding: int = 1,
	border: bool = True
):
	"""
	Форматированно выводит таблицу в консоль с цветовыми акцентами.

	Args:
		headers: список заголовков столбцов
		data: список строк (каждая строка — список значений)
		title: заголовок таблицы (опционально)
		align: выравнивание ('left', 'right', 'center')
		padding: отступы между колонками
		border: рисовать ли разделительную линию
	"""
	# 1. Подготовка: вычисляем максимальную ширину для каждого столбца
	col_widths = []
	for i, header in enumerate(headers):
		max_width = len(str(header))
		for row in data:
			max_width = max(max_width, len(str(row[i])))
		col_widths.append(max_width + padding)

	# 2. Формируем строку заголовка
	if title:
		print(Fore.BLUE + Style.BRIGHT + f"\n{title.upper()}")
		print(Style.RESET_ALL)

	header_parts = []
	for i, header in enumerate(headers):
		if align == "right":
			header_parts.append(f"{header:>{col_widths[i]}}")
		elif align == "center":
			header_parts.append(f"{header:^{col_widths[i]}}")
		else:
			header_parts.append(f"{header:<{col_widths[i]}}")

	header_row = "".join(header_parts)
	print(Fore.CYAN + Style.BRIGHT + header_row)
	Style.RESET_ALL

	# 3. Разделительная линия
	if border:
		separator = "-" * len(header_row)
		print(Fore.YELLOW + separator)
		Style.RESET_ALL

	# 4. Строки данных
	for row in data:
		row_parts = []
		for i, item in enumerate(row):
			item_str = str(item)
			if align == "right":
				row_parts.append(f"{item_str:>{col_widths[i]}}")
			elif align == "center":
				row_parts.append(f"{item_str:^{col_widths[i]}}")
			else:
				row_parts.append(f"{item_str:<{col_widths[i]}}")
		row_str = "".join(row_parts)
		print(Fore.WHITE + row_str)
		Style.RESET_ALL

	# 5. Завершение (если нужно)
	print(Style.RESET_ALL)  # Сброс всех стилей

def plot_split_distributions(processed_data: Dict[str, Dict[str, List]], label2id: Dict[str, int], id2label: Dict[int, str]):
	import matplotlib.pyplot as plt
	from collections import Counter

	fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
	splits = ['train', 'val', 'test']

	for i, split_name in enumerate(splits):
		split = processed_data.get(split_name)
		if not split:
			continue

		labels = split['labels']
		first_labels = [l[0] for l in labels if l]
		label_counts = Counter(first_labels)
		sorted_items = sorted(label_counts.items())
		label_ids, counts = zip(*sorted_items) if sorted_items else ([], [])
		# Используем id2label
		label_names = [id2label.get(i, str(i)) for i in label_ids]

		ax = axes[i]
		color_map = ['skyblue', 'lightcoral', 'lightgreen']
		bars = ax.barh(label_names, counts, color=color_map[i], edgecolor='black')
		ax.set_title(f"{split_name.upper()}", fontsize=14, fontweight='bold')

		for bar, count in zip(bars, counts):
			ax.text(count + 0.5, bar.get_y() + bar.get_height()/2, str(count),
					ha='left', va='center', fontsize=10)

	plt.tight_layout()
	plt.show()

def check_meta():
	try:
		files = find_yaml_files(config['data_dir'])
		if not files:
			print("❌ Не найдено YAML-файлов в указанном каталоге.")
			return

		yaml_path = files[0]

		# 1. Парсим YAML
		meta_data = parse_meta_yaml(yaml_path)

		# 2. Преобразуем в таблицы
		tables = extract_tables_from_meta(meta_data)

		# 3. Выводим каждую таблицу
		print(f"\n✅ Метаданные найдены в {Fore.GREEN}{yaml_path}{Fore.RESET}\n")

		for table_name, table_data in tables.items():
			if "headers" not in table_data or "data" not in table_data:
				print(Fore.RED + f"Ошибка в таблице '{table_name}': отсутствуют 'headers' или 'data'")
				print(Style.RESET_ALL)
				continue

			title = f"{table_name}" # Раздел (CLASSES, SPLITS, PREPROCESSING)
			print_table(
				headers=table_data["headers"],
				data=table_data["data"],
				title=title,
				align="left",
				padding=1,
				border=True
			)
			print()  # Пустая строка между таблицами

		print(f"Общий размер датасета: {sum(split_info['size'] for split_info in meta_data['splits'].values())}")

		pkl_path = Path(config['data_dir']) / "ru_goemotions_metadata.pkl"
		with open(pkl_path, 'rb') as f:
			processed_data = pickle.load(f)
		label2id = {name: idx for idx, name in enumerate(rulables().keys())}
		id2label = {v: k for k, v in label2id.items()}
		plot_split_distributions(processed_data, id2label, label2id)

	except Exception as e:
		print(f"❌ Критическая ошибка: {e}")
		traceback.print_exc()
	ill_be_back()

def start_learning():
	try:
		progress_bar(3, "Запускаю процессы обучения...")
		success("С Любовью!")
		from train import train
		train()
	except Exception as e:
		error(f"Критическая ошибка: {e}")
		traceback.print_exc()
	ill_be_back()

def change_parameters():
	try:
		progress_bar(3, "Запускаю настройку параметров...")
		#python /py param.py
		success("Параметры установлены!")
	except Exception as e:
		error(f"Критическая ошибка: {e}")
		traceback.print_exc()
	ill_be_back()

def parse_meta_yaml(file_path):
	file_path = Path(file_path)
	if not file_path.exists():
		raise FileNotFoundError(f"Файл не найден: {file_path}")
	try:
		with open(file_path, 'r', encoding='utf-8') as f:
			data = yaml.safe_load(f)
			return data
	except yaml.YAMLError as e:
		raise ValueError(f"Ошибка при чтении YAML: {e}")

def check_logs():
	try:
		progress_bar(3, "Ищем логи...")
		folder = Path(config['logs_dir'])

		# Проверяем, есть ли файлы в папке
		if any(folder.iterdir()):
			success(f"Логи найдены: {folder}")

			# Получаем список всех элементов в папке
			items = list(folder.iterdir())

			# Фильтруем только файлы (не папки)
			files = [item.name for item in items if item.is_file()]
			if files:
				print("Файлы логов:\n")
				for file_name in files:
					success(f"  {file_name}")
			else:
				warning("В папке нет файлов (только подпапки).")

				# Опционально: показать подпапки
				dirs = [item.name for item in items if item.is_dir()]
				if dirs:
					print("Подпапки:")
					for dir_name in dirs:
						warning(f"  {dir_name}")
		else:
			warning(f"Папка пуста: {Fore.YELLOW}{folder}")
			error(f"Логи отсутствуют: {Fore.YELLOW}{folder}")
	except Exception as e:
		error(f"Ошибка: {e}")
		traceback.print_exc()
	ill_be_back()

def menu(options):
	for i, option in enumerate(options, 1):
		print(f"{Style.BRIGHT} \n{i}. {option}")

	choice = input(f"\n{Fore.MAGENTA} Или Enter для выхода: {Style.RESET_ALL}")
	print()
	if choice.lower() == 'c':
		colors()
		return
	if not choice:
		warning("Выход...")
		return None
	if choice.isdigit() and 1 <= int(choice) <= len(options):
		return int(choice)
	else:
		error("Неверный ввод!")
		return menu(options)  # Повторный вызов при ошибке

def cli():
	try:
		header("Добро пожаловать в систему обучения!")
		create_folders()
		options = [
		"Создать метаданные",
		"Посмотреть метаданные",
		"Логи и мониторинг",
		"Информацию о системе",
		"Тест матриц",
		"Параметры обучения",
		"Запустить обучение",
		]

		choice = menu(options)

		if choice:
			choice -= 1
			print()
			info(f"Выбрано: {options[choice]}{Style.RESET_ALL}")
			if choice == 0:
				from data import data_start
				data_start()
			if choice == 1: check_meta()
			if choice == 2: check_logs()
			if choice == 3:
				from test import system_info
				system_info()
			if choice == 4:
				from test import test
				test()
			if choice == 5: change_parameters()
			if choice == 6: start_learning()
	except Exception as e:
		error(f"Ошибка: {e}")
		traceback.print_exc()
		#return

if __name__ == "__main__":
	#clear_screen()
	cli()
