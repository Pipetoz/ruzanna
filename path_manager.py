"""
Управление путями с историей и валидацией
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

class PathManager:
    """Менеджер путей с историей и интеллектуальным созданием"""
    
    def __init__(self, history_file: str = "./configs/path_history.json"):
        self.history_file = Path(history_file)
        self.history = self._load_history()
        
        # Шаблоны имен для автоматического создания
        self.name_templates = {
            'experiment': 'exp_{date}_{time}_{name}',
            'session': 'session_{timestamp}',
            'version': 'v{version}_{name}'
        }
    
    def _load_history(self) -> Dict:
        """Загружает историю путей"""
        default = {
            "experiments": [],  # Последние эксперименты
            "sessions": [],     # Последние сессии обучения
            "max_history": 3,
            "last_experiment": None,
            "last_session": None
        }
        
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return {**default, **json.load(f)}
            except:
                pass
        
        return default
    
    def _save_history(self):
        """Сохраняет историю"""
        self.history_file.parent.mkdir(exist_ok=True)
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def create_experiment_dir(self, 
                            base_path: str, 
                            experiment_name: str = None,
                            template: str = 'experiment') -> Path:
        """Создает директорию для эксперимента с интеллектуальным именем"""
        base = Path(base_path)
        
        # Генерируем имя если не указано
        if not experiment_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f"exp_{timestamp}"
        
        # Создаем путь
        exp_dir = base / experiment_name
        exp_dir.mkdir(exist_ok=True, parents=True)
        
        # Создаем структуру внутри
        subdirs = ['logs', 'checkpoints', 'models', 'configs', 'results', 'tmp']
        for subdir in subdirs:
            (exp_dir / subdir).mkdir(exist_ok=True)
        
        # Добавляем в историю
        self._add_to_history('experiments', str(exp_dir), 'last_experiment')
        
        # Создаем info файл
        info = {
            'name': experiment_name,
            'created': datetime.now().isoformat(),
            'path': str(exp_dir),
            'structure': subdirs
        }
        
        with open(exp_dir / 'experiment_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"🎯 Создан эксперимент: {experiment_name}")
        print(f"   Путь: {exp_dir}")
        
        return exp_dir
    
    def create_session_dir(self, 
                          experiment_dir: Path,
                          session_name: str = None) -> Path:
        """Создает директорию для сессии обучения внутри эксперимента"""
        # Директория сессий
        sessions_dir = experiment_dir / 'sessions'
        sessions_dir.mkdir(exist_ok=True)
        
        # Имя сессии
        if not session_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_name = f"session_{timestamp}"
        
        session_dir = sessions_dir / session_name
        session_dir.mkdir(exist_ok=True)
        
        # Добавляем в историю
        self._add_to_history('sessions', str(session_dir), 'last_session')
        
        return session_dir
    
    def get_all_paths(self, base_dir: Path) -> Dict[str, Path]:
        """Возвращает все пути для данного эксперимента"""
        return {
            'base': base_dir,
            'logs': base_dir / 'logs',
            'checkpoints': base_dir / 'checkpoints',
            'models': base_dir / 'models',
            'configs': base_dir / 'configs',
            'results': base_dir / 'results',
            'tmp': base_dir / 'tmp',
            'sessions': base_dir / 'sessions'
        }
    
    def _add_to_history(self, category: str, path: str, last_key: str):
        """Добавляет путь в историю"""
        if category not in self.history:
            self.history[category] = []
        
        # Удаляем если уже есть
        if path in self.history[category]:
            self.history[category].remove(path)
        
        # Добавляем в конец
        self.history[category].append(path)
        
        # Ограничиваем историю
        max_history = self.history.get('max_history', 3)
        self.history[category] = self.history[category][-max_history:]
        
        # Обновляем последний
        self.history[last_key] = path
        
        self._save_history()
    
    def get_history_menu(self) -> List[Tuple[str, str]]:
        """Возвращает историю для меню выбора"""
        menu_items = []
        
        # Последние эксперименты
        if self.history.get('experiments'):
            menu_items.append(("📁 Последние эксперименты:", ""))
            for i, path in enumerate(self.history['experiments'][-3:], 1):
                exp_name = Path(path).name
                menu_items.append((f"  {i}. {exp_name}", path))
        
        # Последние сессии
        if self.history.get('sessions'):
            menu_items.append(("📊 Последние сессии:", ""))
            for i, path in enumerate(self.history['sessions'][-3:], 1):
                session_name = Path(path).name
                menu_items.append((f"  {i+3}. {session_name}", path))
        
        return menu_items
    
    def cleanup_old_files(self, dir_path: Path, keep_last: int = 5):
        """Очищает старые файлы, оставляя только последние keep_last"""
        if not dir_path.exists():
            return
        
        # Для чекпоинтов: удаляем старые, оставляем последние N
        if (dir_path / 'checkpoints').exists():
            checkpoints = sorted((dir_path / 'checkpoints').glob('*'))
            for checkpoint in checkpoints[:-keep_last]:
                try:
                    if checkpoint.is_file():
                        checkpoint.unlink()
                    else:
                        shutil.rmtree(checkpoint)
                except:
                    pass
        
        # Для логов: удаляем старые логи, оставляем последние N
        if (dir_path / 'logs').exists():
            logs = sorted((dir_path / 'logs').glob('*.log'))
            for log in logs[:-keep_last]:
                try:
                    log.unlink()
                except:
                    pass
