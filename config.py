#config.py
# ✓ ⚠️ ❌ ℹ
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

import re
if ma: print(ts(st, "re"))

import sys
if ma: print(ts(st, "sys"))

from pathlib import Path
if ma: print(ts(st, "pathlib") )

import yaml
if ma: print(ts(st, "yaml"))

if ma: print("\n" + ts(tot, "Общее время импортов") + "\n")

def clear_screen():
	"""Очистка экрана"""
	os.system('cls' if os.name == 'nt' else 'clear')
	#print("Тут была функция очистки экрана")

def success(msg):
	print(f"{Style.BRIGHT}{Fore.GREEN}✓  {msg}{Style.RESET_ALL}\n")

def warning(msg):
	print(f"{Style.BRIGHT}{Fore.YELLOW}⚠️  {msg}{Style.RESET_ALL}\n")

def error(msg):
	print(f"{Style.BRIGHT}{Fore.RED}❌  {msg}{Style.RESET_ALL}\n")

def info(msg):
	print(f"{Style.BRIGHT}{Fore.CYAN}ℹ  {msg}{Style.RESET_ALL}\n")

def title(text):
	print(f"\n{Style.BRIGHT}{Fore.MAGENTA}{'='*45}{Style.RESET_ALL}")
	print(f"   {text.upper()}")
	print(f"{Style.BRIGHT}{Fore.MAGENTA}{'='*45}{Style.RESET_ALL}\n")

def header(text):
	print(f"\n{Style.BRIGHT}{Fore.CYAN}{'='*45}{Style.RESET_ALL}")
	print(f"   {text}")
	print(f"{Style.BRIGHT}{Fore.CYAN}{'='*45}{Style.RESET_ALL}\n")

def rainbow_text(text):
	colors = [Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.CYAN, Fore.BLUE, Fore.MAGENTA]
	result = ""
	for i, char in enumerate(text):
		result += colors[i % len(colors)] + char
	return result + Style.RESET_ALL

def progress_bar(total, label="Прогресс", on_cancel=None):
	"""
	Прогресс‑бар с обработкой Ctrl+C.

	Args:
		total (int): общее число шагов.
		label (str): подпись к бару.
		on_cancel (func): функция, вызываемая при прерывании (опционально).
	"""
	try:
		for i in range(total + 1):
			percent = (i / total) * 100
			filled = int(30 * i // total)
			bar = "█" * filled + "░" * (30 - filled)

			# Цвет в зависимости от прогресса
			if percent < 30:
				color = Fore.RED
			elif percent < 70:
				color = Fore.YELLOW
			else:
				color = Fore.GREEN

			# Вывод в одну строку (\r — возврат в начало строки)
			print(f"{Style.BRIGHT}"
				f"\r{color}{label}: |{bar}| {percent:6.1f}%{Style.RESET_ALL}",
				end="",
				flush=True
			)
			time.sleep(0.1)  # имитация работы; уберите/измените в реальном коде

		print()  # новая строка после бара

	except KeyboardInterrupt:
		error(f"Прервано пользователем {Fore.YELLOW}(Ctrl+C)")
		if on_cancel:
			on_cancel()  # вызов коллбэка (например, очистка, лог, выход)
		print()  # дополнительная пустая строка для читаемости
		return False  # признак прерывания
	print()  # новая строка после бара
	return True  # признак успешного завершения

def create_folders():
	isSource = isDataset = isData = False

	# Берем только ключи, которые оканчиваются на '_dir' - это пути
	for key in [k for k in config.keys() if k.endswith('_dir')]:
		folder_path = config[key]

		# Преобразуем в Path
		if isinstance(folder_path, str):
			folder = Path(folder_path)
		elif isinstance(folder_path, Path):
			folder = folder_path
		else:
			print(f"\n{Fore.RED}Ошибка: {Fore.YELLOW}значение для '{key}' "
				  f"не является строкой или Path:{Fore.RESET} {folder_path}")
			continue

		# Проверяем существование папки
		if not folder.exists():
			warning(f"Создана папка: {folder}")
			folder.mkdir(parents=True, exist_ok=True)
		else:
			# Только здесь проверяем содержимое для нужных ключей
			if key == "source_model_dir":
				if any(folder.iterdir()):
					success(f"Исходная модель: {folder}")
					isSource = True
				else:
					error(f"Исходная модель отсутствует: {Fore.YELLOW}{folder}")

			elif key == "dataset_dir":
				if any(folder.iterdir()):
					success(f"Датасет: {folder}")
					isDataset = True
				else:
					error(f"Датасет отсутствует: {Fore.YELLOW}{folder}")

			elif key == "data_dir":
				if any(folder.iterdir()):
					success(f"Метаданные: {folder}")
					isData = True
				else:
					error(f"Метаданные отсутствуют: {Fore.YELLOW}{folder}")

	# Итоговые сообщения — ПОСЛЕ цикла
	if isSource and isDataset and isData:
		success(f"Данные успешно загружены!")
	else:
		print(rainbow_text("Отсутствуют важные компоненты проекта!"))

	return isSource and isDataset and isData

def rulables():
	try:
		with open(Path(config['dataset_dir']) / "dataset" / "label_to_russian_label.yaml", 'r', encoding='utf-8') as f:
			rulables = yaml.safe_load(f)
		if not isinstance(rulables, dict):
			raise ValueError("Загруженные данные не являются словарем.")
	except FileNotFoundError:
		error("Файл 'label_to_russian_label.yaml' не найден. Проверьте путь к файлу.")
		rulables = {}  # Или можно завершить выполнение или задать значение по умолчанию
	except yaml.YAMLError as e:
		error(f"Ошибка при парсинге YAML: {e}")
		rulables = {}
	except ValueError as e:
		warning(f"Некорректный формат данных: {e}")
		rulables = {}
	return rulables

def find_yaml_files(directory):
	yaml_extensions = ['.yaml', '.yml']
	yaml_files = []

	for root, dirs, files in os.walk(directory):
		for file in files:
			if any(file.lower().endswith(ext) for ext in yaml_extensions):
				yaml_files.append(Path(root) / f"{file}")
	return yaml_files

# Загрузка конфига
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
	raw_config = yaml.safe_load(f)

def _replace_placeholders_in_string(s, context):
	"""Заменяет плейсхолдеры ${...} в строке на значения из контекста"""
	pattern = re.compile(r'\$\{([^}]+)\}')
	def replace(match):
		key = match.group(1)
		if key in context:
			return str(context[key])
		else:
			raise KeyError(f"Неизвестный плейсхолдер: {key}")
	return pattern.sub(replace, s)

def _resolve_structure(obj, context):
	"""Рекурсивно разрешает плейсхолдеры в структуре данных"""
	if isinstance(obj, dict):
		return {k: _resolve_structure(v, context) for k, v in obj.items()}
	elif isinstance(obj, list):
		return [_resolve_structure(v, context) for v in obj]
	elif isinstance(obj, str):
		return _replace_placeholders_in_string(obj, context)
	else:
		return obj

def resolve_flat_dict(d):
	"""Разрешает взаимные ссылки в плоском словаре (для paths)"""
	changed = True
	while changed:
		changed = False
		for key, value in d.items():
			if isinstance(value, str):
				new_value = _replace_placeholders_in_string(value, d)
				if new_value != value:
					d[key] = new_value
					changed = True
	return d

# Обработка конфига в зависимости от формата
if 'paths' in raw_config or 'training' in raw_config:
	# Новый формат с разделами
	# 1. Обрабатываем paths (разрешаем взаимные ссылки)
	if 'paths' in raw_config:
		paths = raw_config['paths']
		paths = resolve_flat_dict(paths)
		raw_config['paths'] = paths
	else:
		paths = {}

	# 2. Используем paths как контекст для остальных разделов
	context = paths

	# 3. Обрабатываем остальные разделы
	for key in raw_config:
		if key != 'paths':
			raw_config[key] = _resolve_structure(raw_config[key], context)

	# 4. Объединяем все разделы в один плоский словарь для обратной совместимости
	config = {}
	for key, value in raw_config.items():
		if isinstance(value, dict):
			config.update(value)  # добавляем все ключи из словаря
		else:
			config[key] = value
else:
	# Старый формат (плоский)
	config = resolve_flat_dict(raw_config)
