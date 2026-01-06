#!/usr/bin/env python3
"""
RUZANNA - Лаунчер для обучения психологического ИИ
Управление конфигурацией и запуск обучения
"""

import os
import sys
import json
import time
from pathlib import Path
import subprocess
from datetime import datetime
import colorama

# Добавляем путь для импорта из core
sys.path.insert(0, str(Path(__file__).parent / "core"))

# Colorama для цветного вывода
try:
    from colorama import Fore, Back, Style, init
    init(autoreset=True)
    COLORS_ENABLED = True
except ImportError:
    class DummyColors:
        def __getattr__(self, name):
            return ""
    Fore = Back = Style = DummyColors()
    COLORS_ENABLED = False

# Импорт нашего конфиг-менеджера
try:
    from config_loader import ConfigManager
    CONFIG_MANAGER = ConfigManager("./configs")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Убедитесь, что файл core/config_loader.py существует")
    sys.exit(1)

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def create_dialogues():
    """Создание диалогов для обучения"""
    print_header()
    print(f"{Fore.CYAN}{Style.BRIGHT}🎭 СОЗДАНИЕ ПСИХОЛОГИЧЕСКИХ ДИАЛОГОВ{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    # 1. Сначала выбираем куда сохранять
    output_dir = select_output_directory()
    if output_dir is None:
        print_info("Создание отменено")
        return
    
    # 2. Спрашиваем количество диалогов
    print(f"\n{Fore.YELLOW}Сколько диалогов создать?{Style.RESET_ALL}")
    print("1. 1,000 (тестовый режим)")
    print("2. 10,000 (стандартный набор)")
    print("3. 50,000 (большой набор)")
    print("4. Ввести своё число")
    
    choice = input(f"\n{Fore.YELLOW}Выберите вариант (1-4): {Style.RESET_ALL}").strip()
    
    if choice == "1":
        num_dialogues = 1000
    elif choice == "2":
        num_dialogues = 10000
    elif choice == "3":
        num_dialogues = 50000
    elif choice == "4":
        while True:
            try:
                num_dialogues = int(input(f"{Fore.YELLOW}Введите число диалогов: {Style.RESET_ALL}"))
                if num_dialogues > 0:
                    break
                else:
                    print_error("Число должно быть положительным")
            except ValueError:
                print_error("Введите число")
    else:
        print_error("Неверный выбор")
        return
    
    # 3. Определяем путь для сохранения диалогов
    dialogues_dir = Path(output_dir).parent / "data"
    dialogues_dir.mkdir(exist_ok=True)
    
    dialogues_path = dialogues_dir / "dialogues.json"
    
    print(f"\n{Fore.CYAN}📊 ПАРАМЕТРЫ СОЗДАНИЯ:{Style.RESET_ALL}")
    print(f"  Количество диалогов: {num_dialogues:,}")
    print(f"  Сохранение в: {dialogues_path}")
    print(f"  Папка результатов: {output_dir}")
    
    confirm = input(f"\n{Fore.YELLOW}Создать диалоги? (y/n): {Style.RESET_ALL}").lower()
    if confirm != 'y':
        print_info("Создание отменено")
        return
    
    # 4. Запускаем dialog_loader.py
    print(f"\n{Fore.GREEN}{Style.BRIGHT}🎭 СОЗДАНИЕ ДИАЛОГОВ...{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    try:
        # Ищем dialog_loader.py
        dialog_loader_paths = [
            Path(__file__).parent / "dialog_loader.py",
            Path(__file__).parent.parent / "dialog_loader.py",
            Path("dialog_loader.py")
        ]
        print_info(f"Команда: {' '.join(cmd)}")
        print_info(f"Рабочая директория: {os.getcwd()}")
        print_info(f"Путь к dialog_loader.py: {dialog_loader_path}")
        print_info(f"Существует: {dialog_loader_path.exists()}")
        
        dialog_loader_path = None
        for path in dialog_loader_paths:
            if path.exists():
                dialog_loader_path = path
                break
        
        if not dialog_loader_path:
            print_error("❌ dialog_loader.py не найден!")
            return
        
        # Формируем команду
        cmd = [
            sys.executable,
            str(dialog_loader_path),
            str(num_dialogues),
            "--output", output_dir  # ← ПЕРЕДАЕМ ПУТЬ!
        ]
        
        print_info(f"Запуск: {' '.join(cmd)}")
        
        # Запускаем
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            encoding='utf-8'
        )
        
        # Вывод в реальном времени
        for line in process.stdout:
            print(line, end='')
        
        # Ожидание завершения
        process.wait()
        
        if process.returncode == 0:
            print_success(f"\n✅ Создано {num_dialogues:,} диалогов!")
            
            # Обновляем base.json чтобы указывал на созданные диалоги
            dialogues_path = Path(output_dir) / "data" / "dialogues.json"
            if dialogues_path.exists():
                update_config_with_dialogues_path(str(dialogues_path))
            
            # Показываем статистику
            if dialogues_path.exists():
                import json
                with open(dialogues_path, 'r', encoding='utf-8') as f:
                    dialogues = json.load(f)
                
                print_info(f"📊 Статистика:")
                print(f"  • Файл: {dialogues_path.name}")
                print(f"  • Размер: {dialogues_path.stat().st_size / 1024 / 1024:.1f} MB")
                print(f"  • Диалогов: {len(dialogues)}")
                
                # Пример диалога
                if dialogues:
                    first_dialogue = dialogues[0]
                    if isinstance(first_dialogue, dict) and 'text' in first_dialogue:
                        preview = first_dialogue['text'][:100] + "..." if len(first_dialogue['text']) > 100 else first_dialogue['text']
                        print(f"  • Пример: {preview}")
        else:
            print_error(f"\n❌ Ошибка создания диалогов (код: {process.returncode})")
            
    except FileNotFoundError:
        print_error("Python не найден!")
    except KeyboardInterrupt:
        print_warning("\nСоздание прервано пользователем")
    except Exception as e:
        print_error(f"Ошибка запуска: {e}")
        import traceback
        traceback.print_exc()
    
    input(f"\n{Fore.CYAN}Нажмите Enter для продолжения...{Style.RESET_ALL}")
    
def update_config_with_dialogues_path(dialogues_path: str):
    """Обновляет base.json с новым путем к диалогам"""
    base_config_path = Path("./configs/base.json")
    
    if not base_config_path.exists():
        return
    
    try:
        with open(base_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Обновляем путь
        if 'data' not in config:
            config['data'] = {}
        
        old_path = config['data'].get('path', '')
        config['data']['path'] = str(dialogues_path)
        
        # Сохраняем
        with open(base_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        if old_path != str(dialogues_path):
            print_success(f"✅ Конфиг обновлен: {dialogues_path}")
        else:
            print_info("ℹ️  Конфиг уже указывал на этот файл")
            
    except Exception as e:
        print_warning(f"Не удалось обновить конфиг: {e}")

def clear_screen():
    """Очистка экрана"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Печать заголовка"""
    clear_screen()
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}{'='*70}")
    print(f"{' '*20}🧠 RUZANNA - ПСИХОЛОГИЧЕСКИЙ ИИ ТРЕНЕР")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}{'='*70}{Style.RESET_ALL}\n")

def print_success(msg):
    """Успешное сообщение"""
    print(f"{Fore.GREEN}✅ {msg}{Style.RESET_ALL}")

def print_warning(msg):
    """Предупреждение"""
    print(f"{Fore.YELLOW}⚠️  {msg}{Style.RESET_ALL}")

def print_error(msg):
    """Ошибка"""
    print(f"{Fore.RED}❌ {msg}{Style.RESET_ALL}")

def print_info(msg):
    """Информационное сообщение"""
    print(f"{Fore.CYAN}ℹ️  {msg}{Style.RESET_ALL}")

def print_step(msg):
    """Шаг процесса"""
    print(f"{Fore.BLUE}➡️  {msg}{Style.RESET_ALL}")

# ============================================================================
# УПРАВЛЕНИЕ ДИРЕКТОРИЯМИ
# ============================================================================

def get_directory_history() -> list:
    """Получает историю директорий"""
    history_file = Path("./configs/directory_history.json")
    
    if not history_file.exists():
        return []
    
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
        
        history = []
        for item in history_data.get('history', [])[-3:]:  # Последние 3
            path = item.get('path', '')
            if path:
                name = Path(path).name
                history.append((name, path))
        
        return history
    
    except Exception as e:
        print_warning(f"Ошибка чтения истории: {e}")
        return []

def save_to_history(path: str):
    """Сохраняет путь в историю"""
    history_file = Path("./configs/directory_history.json")
    history_file.parent.mkdir(exist_ok=True)
    
    try:
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"history": []}
        
        # Удаляем если уже есть
        path = str(Path(path).resolve())
        data['history'] = [h for h in data['history'] if h.get('path') != path]
        
        # Добавляем в начало (последний выбор будет первым)
        data['history'].insert(0, {
            "path": path,
            "selected": datetime.now().isoformat(),
            "name": Path(path).name
        })
        
        # Ограничиваем 3 элементами
        data['history'] = data['history'][:3]
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    except Exception as e:
        print_warning(f"Не удалось сохранить историю: {e}")

def select_output_directory():
    """Интеллектуальный выбор директории для результатов"""
    print_header()
    print(f"{Fore.CYAN}{Style.BRIGHT}📁 КУДА СОХРАНИТЬ РЕЗУЛЬТАТЫ?{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    # 1. Проверяем историю
    history = get_directory_history()
    
    if history:
        print(f"{Fore.YELLOW}📚 ИСТОРИЯ ВЫБОРА:{Style.RESET_ALL}")
        for i, (name, path) in enumerate(history, 1):
            # Проверяем существование
            exists = "✅" if Path(path).exists() else "❌"
            print(f"{i}. {exists} {name}")
    
    print(f"\n{Fore.YELLOW}⚡ БЫСТРЫЕ ВАРИАНТЫ:{Style.RESET_ALL}")
    print("n. 📁 Новая директория (ввести путь)")
    
    if history:
        print("0. ↩️  Отмена")
    
    print(f"\n{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
    
    # Выбор пользователя
    while True:
        choice = input(f"\n{Fore.YELLOW}Выберите вариант (1-{len(history)}, n, или 0): {Style.RESET_ALL}").strip().lower()
        
        if choice == '0' and history:
            return None  # Отмена
        elif choice == 'n':
            return create_new_directory_interactive()
        elif choice.isdigit() and history:
            idx = int(choice) - 1
            if 0 <= idx < len(history):
                selected_path = history[idx][1]
                if Path(selected_path).exists():
                    print_success(f"Выбрано: {selected_path}")
                    save_to_history(selected_path)
                    return selected_path
                else:
                    print_warning(f"Директория не существует: {selected_path}")
                    recreate = input(f"Создать её заново? (y/n): ").lower()
                    if recreate == 'y':
                        Path(selected_path).mkdir(parents=True, exist_ok=True)
                        print_success(f"Директория создана: {selected_path}")
                        save_to_history(selected_path)
                        return selected_path
        else:
            print_error("Неверный выбор. Попробуйте снова.")
    
    return None

def create_new_directory_interactive():
    """Интерактивное создание новой директории"""
    print(f"\n{Fore.CYAN}🆕 СОЗДАНИЕ НОВОЙ ДИРЕКТОРИИ{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'-'*40}{Style.RESET_ALL}")
    
    # 1. Предлагаем умный дефолт
    default_dir = f"./experiments/psych_train_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    print(f"{Fore.YELLOW}Примеры хороших путей:{Style.RESET_ALL}")
    print(f"  • {default_dir}")
    print(f"  • D:/AI_Experiments/psych_training_{datetime.now().strftime('%Y%m%d')}")
    print(f"  • C:/Projects/Ruzanna/runs/session_{datetime.now().strftime('%H%M')}")
    print(f"\n{Fore.YELLOW}Совет:{Style.RESET_ALL} Используйте дату в названии для порядка!")
    
    # 2. Получаем путь
    path_input = input(f"\n{Fore.YELLOW}Введите путь [{default_dir}]: {Style.RESET_ALL}").strip()
    
    if not path_input:
        path_input = default_dir
        print_info(f"Используется путь по умолчанию: {path_input}")
    
    # 3. Обрабатываем путь
    try:
        path = Path(path_input)
        
        # Если путь относительный, делаем абсолютным относительно trener/
        if not path.is_absolute():
            path = (Path(__file__).parent / path).resolve()
        
        # Создаем директорию
        path.mkdir(parents=True, exist_ok=True)
        
        # Создаем структуру внутри
        subdirs = ['logs', 'checkpoints', 'models', 'configs', 'tmp']
        for subdir in subdirs:
            (path / subdir).mkdir(exist_ok=True)
        
        # Создаем info файл
        info = {
            "created": datetime.now().isoformat(),
            "purpose": "Ruzanna психологический AI",
            "training_session": True
        }
        with open(path / 'session_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        print_success(f"✅ Директория создана: {path}")
        print_info(f"   Поддиректории: {', '.join(subdirs)}")
        
        # Сохраняем в историю
        save_to_history(str(path))
        
        return str(path)
        
    except Exception as e:
        print_error(f"❌ Ошибка создания директории: {e}")
        return None

def open_presets_folder():
    """Открывает папку с пресетами в проводнике"""
    import platform
    import subprocess
    
    # Ищем папку с пресетами
    presets_dirs = [
        Path(__file__).parent.parent / "configs" / "presets",  # ../configs/presets
        Path(__file__).parent / "configs" / "presets",         # ./configs/presets
        Path("./configs/presets")                              # configs/presets
    ]
    
    presets_dir = None
    for dir_path in presets_dirs:
        if dir_path.exists():
            presets_dir = dir_path
            break
    
    if not presets_dir:
        # Создаем папку если её нет
        presets_dir = Path("./configs/presets")
        presets_dir.mkdir(parents=True, exist_ok=True)
        print_info(f"Создана папка для пресетов: {presets_dir}")
    
    # Открываем в проводнике
    try:
        if platform.system() == "Windows":
            os.startfile(str(presets_dir))
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", str(presets_dir)])
        else:  # Linux
            subprocess.run(["xdg-open", str(presets_dir)])
        
        print_success(f"📂 Открыта папка с пресетами: {presets_dir}")
        print_info("\nСоздайте JSON файлы с именами:")
        print("  • fast.json - для быстрого обучения")
        print("  • quality.json - для качественного обучения")
        print("  • debug.json - для отладки")
        print("\nФормат файла пресета:")
        print('''
{
  "meta": {
    "description": "Описание пресета"
  },
  "training": {
    "batch_size": 8,
    "learning_rate": 0.0005,
    "epochs": 2
  }
}''')
        
    except Exception as e:
        print_error(f"Не удалось открыть папку: {e}")
    
    input(f"\n{Fore.CYAN}Нажмите Enter чтобы продолжить...{Style.RESET_ALL}")

# ============================================================================
# ФУНКЦИИ ОБУЧЕНИЯ И МОНИТОРИНГА
# ============================================================================

def start_training():
    """Запуск процесса обучения"""
    print_header()
    print(f"{Fore.CYAN}{Style.BRIGHT}🚀 ЗАПУСК ОБУЧЕНИЯ{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    # 1. Выбор директории
    output_dir = select_output_directory()
    if output_dir is None:  # Явная проверка на None
        print_info("Запуск отменен")
        time.sleep(1)
        return
    
    # 2. Показываем пресеты (ИСПРАВЛЕНО!)
    print_step("Доступные пресеты обучения:")
    
    # Ищем пресеты относительно ПРОЕКТА, а не лаунчера
    project_root = Path(__file__).parent.parent  # Поднимаемся на уровень выше trener/
    presets_dir = project_root / "configs" / "presets"
    
    # Если не нашли там, пробуем рядом с лаунчером
    if not presets_dir.exists():
        presets_dir = Path(__file__).parent / "configs" / "presets"
    
    presets = []
    if presets_dir.exists():
        presets = sorted([p.stem for p in presets_dir.glob("*.json")])
    
    if presets:
        for i, preset in enumerate(presets, 1):
            print(f"  {i}. {preset}")
        print(f"  Enter - использовать базовую конфигурацию")

    # 3. Выбор пресета
    selected_preset = None
    if presets:
        preset_choice = input(f"\n{Fore.YELLOW}Выберите пресет (1-{len(presets)} или Enter): {Style.RESET_ALL}").strip()
        if preset_choice.isdigit():
            idx = int(preset_choice) - 1
            if 0 <= idx < len(presets):
                selected_preset = presets[idx]
                
                # Загружаем пресет для показа параметров
                try:
                    preset_path = presets_dir / f"{selected_preset}.json"
                    with open(preset_path, 'r') as f:
                        preset_data = json.load(f)
                    
                    print_success(f"Выбран пресет: {selected_preset}")
                    if 'training' in preset_data:
                        print_info("Параметры пресета:")
                        for key, value in preset_data['training'].items():
                            print(f"  • {key}: {value}")
                except Exception as e:
                    print_warning(f"Не удалось загрузить пресет: {e}")
    
    # 4. Загрузка конфигурации с пресетом
    try:
        config = CONFIG_MANAGER.load_full_config(preset=selected_preset)
        params = CONFIG_MANAGER.get_training_params()
        
        # Обновляем путь в конфиге
        if 'paths' not in config:
            config['paths'] = {}
        config['paths']['base'] = output_dir
        config['paths']['logs'] = str(Path(output_dir) / 'logs')
        config['paths']['checkpoints'] = str(Path(output_dir) / 'checkpoints')
        
        print_success(f"Конфигурация загружена")
        print_info(f"Директория результатов: {output_dir}")
        
        # Сохраняем конфиг для тренировки
        config_path = Path(output_dir) / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print_info(f"Конфиг сохранен: {config_path}")
        
    except Exception as e:
        print_error(f"Ошибка конфигурации: {e}")
        return
    
    # 5. Автоматический поиск диалогов
    print_step("Поиск данных для обучения...")
    data_path = params.get('data_path', '')
    
    # Функция поиска диалогов
    def find_dialogues_file():
        """Ищет файл с диалогами в типичных местах"""
        search_paths = [
            Path(output_dir) / "data" / "dialogues.json",
            Path(output_dir).parent / "data" / "dialogues.json",
            Path(data_path) if data_path else None,
            Path(__file__).parent.parent / "dialogues.json",
            Path(__file__).parent / "dialogues.json",
            Path("C:/Files/processed_epitome/quality_psych_dialogues_enhanced.json"),
            Path("D:/Files/processed_epitome/quality_psych_dialogues_enhanced.json"),
        ]
        
        for path in search_paths:
            if path and path.exists():
                return path
        return None
    
    dialogues_file = find_dialogues_file()
    
    if not dialogues_file:
        print_error("❌ Файл с диалогами не найден!")
        print_info("\nСначала создайте диалоги через меню 'Создать диалоги'")
        print("Или поместите файл в одну из папок:")
        print(f"  • {Path(output_dir).parent / 'data' / 'dialogues.json'}")
        print(f"  • {Path(output_dir).parent / 'dialogues.json'}")
        print(f"  • {Path(__file__).parent / 'dialogues.json'}")
        
        create_now = input(f"\n{Fore.YELLOW}Создать диалоги сейчас? (y/n): {Style.RESET_ALL}").lower()
        if create_now == 'y':
            create_dialogues()
            # Пробуем снова найти
            dialogues_file = find_dialogues_file()
            if not dialogues_file:
                print_error("❌ Диалоги не созданы. Запуск отменен.")
                return
        else:
            print_info("Запуск отменен")
            return
    
    # Обновляем конфиг с найденным путем
    config['data']['path'] = str(dialogues_file)
    params['data_path'] = str(dialogues_file)
    
    print_success(f"✅ Данные найдены: {dialogues_file.name}")
    print_info(f"  Размер: {dialogues_file.stat().st_size / 1024 / 1024:.1f} MB")
    print_info(f"  Путь: {dialogues_file}")
    
    # 6. Подтверждение запуска
    print(f"\n{Fore.YELLOW}📋 ПАРАМЕТРЫ ЗАПУСКА:{Style.RESET_ALL}")
    print(f"  Модель: {params.get('model_name', 'N/A')}")
    print(f"  Данные: {Path(params.get('data_path', '')).name}")
    print(f"  Batch size: {params.get('batch_size', 'N/A')}")
    print(f"  Эпохи: {params.get('epochs', 'N/A')}")
    print(f"  Learning rate: {params.get('learning_rate', 'N/A'):.2e}")
    print(f"  Сохранение в: {output_dir}")
    
    if selected_preset:
        print(f"  Пресет: {selected_preset}")
    
    print(f"\n{Fore.RED}{Style.BRIGHT}⚠️  ВНИМАНИЕ: Обучение может занять несколько часов!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    confirm = input(f"\n{Fore.YELLOW}Запустить обучение? (y/n): {Style.RESET_ALL}").lower()
    
    if confirm != 'y':
        print_info("Запуск отменен")
        time.sleep(1)
        return
    
    # 7. Запуск workout.py
    print(f"\n{Fore.GREEN}{Style.BRIGHT}▶️  ЗАПУСК ОБУЧЕНИЯ...{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    try:
        # Находим workout.py относительно лаунчера
        workout_path = Path(__file__).parent / "workout.py"
        if not workout_path.exists():
            workout_path = Path("workout.py")  # Попробуем в текущей директории
        
        # Формируем команду
        cmd = [
            sys.executable, 
            str(workout_path),
            "--output_dir", output_dir,
            "--config", str(config_path)
        ]
        
        if selected_preset:
            cmd.extend(["--preset", selected_preset])
        
        print_info(f"Команда: {' '.join(cmd[:3])} ...")  # Не показываем весь путь
        
        # Запускаем
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            encoding='utf-8'
        )
        
        # Вывод в реальном времени
        for line in process.stdout:
            print(line, end='')
        
        # Ожидание завершения
        process.wait()
        
        if process.returncode == 0:
            print_success("\n✅ Обучение успешно завершено!")
            print_info(f"Результаты сохранены в: {output_dir}")
            
            # Показываем что создано
            result_path = Path(output_dir)
            if result_path.exists():
                print(f"\n{Fore.CYAN}📁 СОДЕРЖИМОЕ ДИРЕКТОРИИ:{Style.RESET_ALL}")
                for item in result_path.iterdir():
                    if item.is_dir():
                        file_count = len(list(item.glob("*")))
                        print(f"  📁 {item.name}/ ({file_count} файлов)")
                    else:
                        size_kb = item.stat().st_size / 1024
                        print(f"  📄 {item.name} ({size_kb:.1f} KB)")
        else:
            print_error(f"\n❌ Обучение завершилось с ошибкой (код: {process.returncode})")
            
    except FileNotFoundError:
        print_error(f"Файл workout.py не найден! Искали: {workout_path}")
    except KeyboardInterrupt:
        print_warning("\nОбучение прервано пользователем")
    except Exception as e:
        print_error(f"Ошибка запуска: {e}")
        import traceback
        traceback.print_exc()
    
    input(f"\n{Fore.CYAN}Нажмите Enter для продолжения...{Style.RESET_ALL}")

def edit_base_config():
    """Редактирование базового конфига"""
    base_config_path = Path("./configs/base.json")
    
    if not base_config_path.exists():
        print_error("Базовый конфиг не найден!")
        return
    
    # Показываем текущий конфиг
    try:
        with open(base_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"\n{Fore.CYAN}📝 РЕДАКТИРОВАНИЕ BASE.JSON{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*40}{Style.RESET_ALL}")
        
        # Особенно важно - путь к данным!
        print(f"\n{Fore.YELLOW}⚠️  ВАЖНО: Проверьте путь к данным!{Style.RESET_ALL}")
        
        current_data_path = config.get('data', {}).get('path', '')
        print(f"\nТекущий путь к данным: {current_data_path}")
        
        if current_data_path and Path(current_data_path).exists():
            print_success(f"✅ Файл данных найден ({Path(current_data_path).stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print_error(f"❌ Файл данных НЕ НАЙДЕН!")
        
        print(f"\n{Fore.YELLOW}Варианты:{Style.RESET_ALL}")
        print("1. Изменить путь к данным")
        print("2. Открыть конфиг в редакторе")
        print("3. Назад")
        
        choice = input(f"\n{Fore.YELLOW}Выберите действие (1-3): {Style.RESET_ALL}").strip()
        
        if choice == "1":
            new_path = input(f"{Fore.YELLOW}Новый путь к данным: {Style.RESET_ALL}").strip()
            if new_path:
                # Обновляем конфиг
                if 'data' not in config:
                    config['data'] = {}
                config['data']['path'] = new_path
                
                # Сохраняем
                with open(base_config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                print_success(f"✅ Путь обновлен: {new_path}")
                
                # Проверяем новый путь
                if Path(new_path).exists():
                    print_success(f"✅ Новый файл найден!")
                else:
                    print_warning(f"⚠️  Файл пока не существует")
        
        elif choice == "2":
            # Открываем в редакторе
            import platform
            import subprocess
            
            try:
                if platform.system() == "Windows":
                    os.startfile(str(base_config_path))
                elif platform.system() == "Darwin":
                    subprocess.run(["open", str(base_config_path)])
                else:
                    subprocess.run(["xdg-open", str(base_config_path)])
                
                print_success(f"✅ Конфиг открыт в редакторе")
                print_info("\nСтруктура конфига:")
                print('''{
  "data": {
    "path": "ПУТЬ_К_ВАШИМ_ДАННЫМ.json",  ← ВАЖНО!
    "train_split": 0.85
  },
  "training": {
    "batch_size": 3,
    "epochs": 3,
    "learning_rate": 0.0002
  }
  // ... остальные параметры ...
}''')
                
            except Exception as e:
                print_error(f"Не удалось открыть файл: {e}")
    
    except Exception as e:
        print_error(f"Ошибка чтения конфига: {e}")
    
    input(f"\n{Fore.CYAN}Нажмите Enter чтобы продолжить...{Style.RESET_ALL}")

def manage_configuration():
    """Управление конфигурацией (упрощенное)"""
    print_header()
    print(f"{Fore.CYAN}{Style.BRIGHT}⚙️  УПРАВЛЕНИЕ КОНФИГУРАЦИЕЙ{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    print("1. 📋 Просмотреть текущую конфигурацию")
    print("2. ✏️  Изменить параметры обучения")
    print("3. 🎯 Выбрать/создать пресет")
    print("4. ↩️  Вернуться в главное меню")
    
    choice = input(f"\n{Fore.YELLOW}Выберите действие (1-4): {Style.RESET_ALL}").strip()
    
    if choice == "1":
        view_configuration()
    elif choice == "2":
        edit_base_config()
    elif choice == "3":
        open_presets_folder()
    elif choice == "4":
        return
    else:
        print_error("Неверный выбор")

def view_configuration():
    """Просмотр конфигурации"""
    try:
        config = CONFIG_MANAGER.load_full_config()
        params = CONFIG_MANAGER.get_training_params()
        
        print(f"\n{Fore.CYAN}📋 ТЕКУЩАЯ КОНФИГУРАЦИЯ{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*40}{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}🎯 ОСНОВНЫЕ ПАРАМЕТРЫ:{Style.RESET_ALL}")
        print(f"  Модель: {params.get('model_name', 'N/A')}")
        print(f"  Данные: {params.get('data_path', 'N/A')}")
        print(f"  Batch size: {params.get('batch_size', 'N/A')}")
        print(f"  Эпохи: {params.get('epochs', 'N/A')}")
        print(f"  Learning rate: {params.get('learning_rate', 'N/A'):.2e}")
        
        print(f"\n{Fore.YELLOW}⚙️  ДОПОЛНИТЕЛЬНЫЕ:{Style.RESET_ALL}")
        print(f"  Max length: {config.get('tokenization', {}).get('max_length', 'N/A')}")
        print(f"  Warmup ratio: {config.get('training', {}).get('warmup_ratio', 'N/A')}")
        print(f"  Gradient accumulation: {config.get('training', {}).get('gradient_accumulation', 'N/A')}")
        
    except Exception as e:
        print_error(f"Ошибка загрузки конфигурации: {e}")
    
    input(f"\n{Fore.CYAN}Нажмите Enter для продолжения...{Style.RESET_ALL}")

def change_training_params():
    """Изменение параметров обучения"""
    print(f"\n{Fore.CYAN}✏️  ИЗМЕНЕНИЕ ПАРАМЕТРОВ{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'-'*40}{Style.RESET_ALL}")
    
    print_info("Функция в разработке...")
    print("Используйте пресеты для быстрого изменения параметров")
    
    input(f"\n{Fore.CYAN}Нажмите Enter для продолжения...{Style.RESET_ALL}")

def manage_presets():
    """Управление пресетами"""
    print(f"\n{Fore.CYAN}🎯 УПРАВЛЕНИЕ ПРЕСЕТАМИ{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'-'*40}{Style.RESET_ALL}")
    
    presets_dir = Path("./configs/presets")
    if not presets_dir.exists():
        presets_dir.mkdir(parents=True)
    
    presets = sorted([p.stem for p in presets_dir.glob("*.json")])
    
    if presets:
        print_info("Доступные пресеты:")
        for i, preset in enumerate(presets, 1):
            print(f"  {i}. {preset}")
    else:
        print_warning("Пресеты не найдены")
    
    print(f"\n1. 🆕 Создать новый пресет")
    print("2. 📋 Просмотреть пресет")
    print("3. ↩️  Назад")
    
    choice = input(f"\n{Fore.YELLOW}Выберите действие (1-3): {Style.RESET_ALL}").strip()
    
    if choice == "1":
        create_preset_interactive()
    elif choice == "2" and presets:
        view_preset(presets)
    
    input(f"\n{Fore.CYAN}Нажмите Enter для продолжения...{Style.RESET_ALL}")

def create_preset_interactive():
    """Создание нового пресета"""
    print(f"\n{Fore.CYAN}🆕 СОЗДАНИЕ НОВОГО ПРЕСЕТА{Style.RESET_ALL}")
    
    preset_name = input(f"{Fore.YELLOW}Имя пресета (латинскими буквами): {Style.RESET_ALL}").strip()
    if not preset_name:
        print_error("Имя не может быть пустым")
        return
    
    # Загружаем текущий конфиг как основу
    current_config = CONFIG_MANAGER.load_full_config()
    
    # Создаем пресет на основе важных параметров
    preset_config = {
        "meta": {
            "description": f"Пресет '{preset_name}'",
            "created": datetime.now().isoformat(),
            "based_on": "current_config"
        },
        "training": {
            "batch_size": current_config.get("training", {}).get("batch_size", 3),
            "learning_rate": current_config.get("training", {}).get("learning_rate", 0.0002),
            "epochs": current_config.get("training", {}).get("epochs", 3),
            "warmup_ratio": current_config.get("training", {}).get("warmup_ratio", 0.9)
        }
    }
    
    # Позволяем изменить параметры
    print_info("\nТекущие значения (нажмите Enter чтобы оставить):")
    
    params = [
        ("batch_size", "Размер батча", int),
        ("learning_rate", "Скорость обучения", float),
        ("epochs", "Количество эпох", int),
        ("warmup_ratio", "Доля warmup", float)
    ]
    
    for key, desc, dtype in params:
        current = preset_config["training"][key]
        new_val = input(f"{desc} [{current}]: ").strip()
        if new_val:
            try:
                preset_config["training"][key] = dtype(new_val)
            except ValueError:
                print_error(f"Неверный формат, оставляем {current}")
    
    # Сохраняем пресет
    try:
        CONFIG_MANAGER.save_preset(preset_name, preset_config)
        print_success(f"Пресет '{preset_name}' сохранен!")
    except Exception as e:
        print_error(f"Ошибка сохранения: {e}")

def view_preset(presets):
    """Просмотр пресета"""
    choice = input(f"{Fore.YELLOW}Выберите пресет для просмотра (1-{len(presets)}): {Style.RESET_ALL}").strip()
    
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(presets):
            preset_name = presets[idx]
            try:
                preset_config = CONFIG_MANAGER.load_preset(preset_name)
                
                print(f"\n{Fore.CYAN}📋 ПРЕСЕТ: {preset_name}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'-'*40}{Style.RESET_ALL}")
                
                if "meta" in preset_config:
                    print(f"Описание: {preset_config['meta'].get('description', 'Нет описания')}")
                
                if "training" in preset_config:
                    print(f"\n{Fore.YELLOW}Параметры обучения:{Style.RESET_ALL}")
                    for key, value in preset_config["training"].items():
                        print(f"  {key}: {value}")
            except Exception as e:
                print_error(f"Ошибка загрузки пресета: {e}")

def check_logs():
    """Просмотр логов (упрощенная версия)"""
    print(f"\n{Fore.CYAN}📊 ПРОСМОТР ЛОГОВ{Style.RESET_ALL}")
    
    # Ищем логи в истории директорий
    history = get_directory_history()
    
    if not history:
        print_warning("История директорий пуста")
        input(f"\n{Fore.CYAN}Нажмите Enter для продолжения...{Style.RESET_ALL}")
        return
    
    print_info("Выберите директорию для просмотра логов:")
    for i, (name, path) in enumerate(history, 1):
        print(f"{i}. {name}")
    
    choice = input(f"\n{Fore.YELLOW}Выберите (1-{len(history)}): {Style.RESET_ALL}").strip()
    
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(history):
            log_dir = Path(history[idx][1]) / "logs"
            if log_dir.exists():
                log_files = list(log_dir.glob("*.log"))
                if log_files:
                    latest = max(log_files, key=lambda x: x.stat().st_mtime)
                    print_info(f"Последний лог: {latest.name}")
                    
                    # Показываем последние 20 строк
                    try:
                        with open(latest, 'r', encoding='utf-8') as f:
                            lines = f.readlines()[-20:]
                            print(f"\n{Fore.CYAN}Последние строки:{Style.RESET_ALL}")
                            for line in lines:
                                print(line.rstrip())
                    except:
                        print_error("Ошибка чтения лога")
                else:
                    print_warning("Логи не найдены")
            else:
                print_warning(f"Директория логов не найдена: {log_dir}")
    
    input(f"\n{Fore.CYAN}Нажмите Enter для продолжения...{Style.RESET_ALL}")

# ============================================================================
# ГЛАВНОЕ МЕНЮ
# ============================================================================

def main():
    """Главное меню лаунчера"""
    while True:
        print_header()
        print(f"{Fore.CYAN}{Style.BRIGHT}🏠 ГЛАВНОЕ МЕНЮ{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        print("1. 🚀 Запустить обучение")
        print("2. 🎭 Создать диалоги для обучения")
        print("3. ⚙️  Управление конфигурацией")
        print("4. 📊 Просмотреть логи")
        print("5. ❌ Выход")
        
        choice = input(f"\n{Fore.YELLOW}Выберите действие (1-5): {Style.RESET_ALL}").strip()
        
        if choice == "1":
            start_training()
        elif choice == "2":
            create_dialogues()
        elif choice == "3":
            manage_configuration()
        elif choice == "4":
            check_logs()
        elif choice == "5":
            print_header()
            print(f"{Fore.GREEN}{Style.BRIGHT}До свидания! Спасибо за использование RUZANNA! 👋{Style.RESET_ALL}\n")
            break
        else:
            print_error("Неверный выбор. Пожалуйста, выберите 1-5.")
            time.sleep(1)

# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Программа прервана пользователем{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n\n{Fore.RED}Критическая ошибка: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        input("Нажмите Enter для выхода...")
