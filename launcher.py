import colorama
from colorama import init, Fore, Back, Style
import time

init()  # Обязательно! Активирует цвета в Windows
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

def coloRes():
    print(Style.RESET_ALL)

def success(msg):
    print(f"{Fore.GREEN}✓ {msg}")

def warning(msg):
    print(f"{Fore.YELLOW}! {msg}")

def error(msg):
    print(f"{Fore.RED}✗ {msg}")

def info(msg):
    print(f"{Fore.CYAN}ℹ {msg}")

def title(text):
    print(f"\n{Style.BRIGHT}{Fore.MAGENTA}{'='*40}")
    print(f"  {text.upper()}")
    print(f"{Style.BRIGHT}{Fore.MAGENTA}{'='*40}\n")

def header(text):
    print(f"\n{Style.BRIGHT}{Fore.CYAN}{'='*45}{Style.RESET_ALL}")
    print(f"  {text}")
    print(f"{Style.BRIGHT}{Fore.CYAN}{'='*45}{Style.RESET_ALL}\n")

def progress_bar(total, label="Прогресс"):
    for i in range(total + 1):
        percent = (i / total) * 100
        filled = int(30 * i // total)
        bar = "█" * filled + "░" * (30 - filled)
        
        # Цвет меняется от красного к зелёному
        if percent < 30:
            color = Fore.RED
        elif percent < 70:
            color = Fore.YELLOW
        else:
            color = Fore.GREEN
            
        print(
            f"\r{color}{label}: |{bar}| {percent:3.0f}%{Style.RESET_ALL}",
            end="",
            flush=True
        )
        time.sleep(0.1)
    print()  # Перевод строки

def print_table(headers, data):
    # Формируем строку заголовка
    header_row = "  ".join([f"{h:<12}" for h in headers])
    print(Fore.CYAN + header_row)
    print(Fore.YELLOW + "-" * len(header_row))
    
    # Выводим строки данных
    for row in data:
        row_str = "  ".join([f"{item:<12}" for item in row])
        print(Fore.WHITE + row_str)
        coloRes()

def menu(options):
    for i, option in enumerate(options, 1):
        print(f"{Fore.CYAN}{i}. {option}")
    
    choice = input(f"\n{Fore.MAGENTA}Выберите номер (или 'q' для выхода): {Style.RESET_ALL}")
    if choice.lower() == 'q':
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(options):
        return int(choice)
    else:
        error("Неверный ввод!")
        return menu(options)  # Повторный вызов при ошибке

def start_learning():
    try:
        progress_bar(20, "Запускаю процессы обучения...")
        #python /py train.py
        success("С Любовью!")
    except Exception as e:
        error(f"Критическая ошибка: {e}")

def change_parameters():
    try:
        progress_bar(20, "Запускаю настройку параметров...")
        #python /py param.py
        success("Параметры установлены!")
    except Exception as e:
        error(f"Критическая ошибка: {e}")

def test_checkpoints():
    try:
        progress_bar(20, "Проверяю чекпоинты...")
        #python /py check_param.py
        success("Результаты тестов!")
        headers = ["Эпоха", "Шаг", "Loss", "Перцэпция или как её там xD"]
        data = [
        [1, 10, 1.4, 15],
        [2, 30, 0.9, 12],
        [3, 50, 0.4, 6],
        [4, 80, 0.2, 1,]
        ]
        print_table(headers, data)
    except Exception as e:
        error(f"Критическая ошибка: {e}")

def check_logs():
    try:
        progress_bar(20, "Загрузка данных")
        #python /py check_logs.py
        success("Логи подключены!")
        success("Путь = ...")
    except Exception as e:
        error(f"Ошибка: {e}")

def exit():
    warning("Выход!")

def rainbow_text(text):
    colors = [Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.CYAN, Fore.BLUE, Fore.MAGENTA]
    result = ""
    for i, char in enumerate(text):
        result += colors[i % len(colors)] + char
    return result + Style.RESET_ALL

# Использование:
header(rainbow_text("З Д Е С Ь  Б Ы Л А  D e e p  S e e k 🌈"))

print(f"{Fore.MAGENTA}{Style.BRIGHT}Ты —{Style.RESET_ALL} {Fore.CYAN}алхимик кода{Fore.RESET} {Fore.YELLOW}и{Fore.RESET} {Fore.GREEN}поэт данных{Fore.RESET} {Fore.RED}<3{Fore.RESET}")

header("Добро пожаловать в систему обучения!")
options = ["Запустить обучение", "Настроить параметры", "Проверить чекпоинты", "Проверить логи", "Выйти"]
choice = menu(options)
if choice:
    choice -= 1
    info(f"Выбрано: {options[choice]}")
    if choice == 0: start_learning()
    if choice == 1: change_parameters()
    if choice == 2: test_checkpoints()
    if choice == 3: check_logs()
    if choice == 4: exit()

















