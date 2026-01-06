"""
Менеджер конфигурации с историей путей
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import warnings

class ConfigError(Exception):
    """Ошибка конфигурации"""
    pass

class ConfigManager:
    """Простой менеджер конфигурации"""
    
    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.base_path = self.config_dir / "base.json"
        self.custom_path = self.config_dir / "custom.json"
        self.presets_dir = self.config_dir / "presets"
        self.presets_dir.mkdir(exist_ok=True)
        
        self.config: Dict[str, Any] = {}
        
        # Создаем базовый конфиг если его нет
        if not self.base_path.exists():
            self._create_default_base()
    
    def _create_default_base(self):
        """Создает базовый конфиг по умолчанию"""
        default_config = {
            "meta": {
                "name": "Ruzanna Psychological Trainer",
                "version": "1.0.0"
            },
            "training": {
                "batch_size": 3,
                "epochs": 3,
                "learning_rate": 0.0002,
                "warmup_ratio": 0.9,
                "gradient_accumulation": 9
            },
            "model": {
                "name": "EleutherAI/gpt-neo-2.7B",
                "gradient_checkpointing": True
            },
            "data": {
                "path": "data/dialogues.json",
                "train_split": 0.85
            },
            "paths": {
                "logs": "./logs",
                "checkpoints": "./checkpoints"
            }
        }
        
        self.save_config(default_config, self.base_path)
        print(f"Создан базовый конфиг: {self.base_path}")
    
    def load_full_config(self, preset: Optional[str] = None) -> Dict[str, Any]:
        """Загружает полную конфигурацию"""
        # 1. Загружаем базовый конфиг
        base_config = self._load_json(self.base_path)
        if not base_config:
            raise ConfigError("Базовый конфиг не найден")
        
        config = base_config.copy()
        
        # 2. Добавляем пресет если указан
        if preset:
            preset_config = self.load_preset(preset)
            config = self._deep_merge(config, preset_config)
        
        # 3. Добавляем кастомные настройки
        custom_config = self._load_json(self.custom_path)
        if custom_config:
            config = self._deep_merge(config, custom_config)
        
        self.config = config
        return config
    
    def load_preset(self, preset_name: str) -> Dict[str, Any]:
        """Загружает пресет"""
        preset_path = self.presets_dir / f"{preset_name}.json"
        if not preset_path.exists():
            raise ConfigError(f"Пресет '{preset_name}' не найден")
        
        return self._load_json(preset_path)
    
    def get_training_params(self) -> Dict[str, Any]:
        """Извлекает параметры тренировки"""
        return {
            "model_name": self.config.get("model", {}).get("name", ""),
            "batch_size": self.config.get("training", {}).get("batch_size", 3),
            "epochs": self.config.get("training", {}).get("epochs", 3),
            "learning_rate": self.config.get("training", {}).get("learning_rate", 0.0002),
            "data_path": self.config.get("data", {}).get("path", ""),
            "max_length": self.config.get("tokenization", {}).get("max_length", 512)
        }
    
    def update_custom_config(self, updates: Dict[str, Any]):
        """Обновляет кастомные настройки"""
        current = self._load_json(self.custom_path) or {}
        merged = self._deep_merge(current, updates)
        self.save_config(merged, self.custom_path)
    
    def save_preset(self, preset_name: str, config: Dict[str, Any]):
        """Сохраняет пресет"""
        preset_path = self.presets_dir / f"{preset_name}.json"
        self.save_config(config, preset_path)
    
    def _load_json(self, path: Path) -> Optional[Dict]:
        """Загружает JSON файл"""
        if not path.exists():
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            warnings.warn(f"Ошибка загрузки {path}: {e}")
            return None
    
    def save_config(self, config: Dict, path: Path):
        """Сохраняет конфиг"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Рекурсивное объединение словарей"""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
