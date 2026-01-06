#!/usr/bin/env python3
"""
RUZANNA - Основной модуль обучения психологического ИИ
Интеграция с конфигурационной системой
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from path_manager import PathManager

# ============================================================================
# НАСТРОЙКА ПУТЕЙ И ИМПОРТОВ
# ============================================================================

# Добавляем пути для импорта
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "core"))
sys.path.insert(0, str(current_dir / "data"))

# Подавление предупреждений
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="torch.utils.checkpoint")

# Импорт наших модулей
try:
    from config_loader import ConfigManager
except ImportError as e:
    print(f"❌ Ошибка импорта config_loader: {e}")
    print("Убедитесь, что файл core/config_loader.py существует")
    sys.exit(1)

# Импорт библиотек для обучения
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import transformers
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        get_linear_schedule_with_warmup,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling
    )
    import numpy as np
    import pandas as pd
    from tqdm.auto import tqdm
    import psutil
    import GPUtil
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
except ImportError as e:
    print(f"❌ Ошибка импорта библиотек: {e}")
    print("Установите необходимые библиотеки:")
    print("pip install torch transformers numpy pandas tqdm psutil gputil colorama")
    sys.exit(1)

# ============================================================================
# КЛАСС ДАТАСЕТА
# ============================================================================

class PsychDialogueDataset(Dataset):
    """Датасет психологических диалогов"""
    
    def __init__(self, dialogues: List, tokenizer, max_length: int = 512):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.dialogues)
    
    def __getitem__(self, idx: int) -> Dict:
        dialogue = self.dialogues[idx]
        
        # Форматируем диалог
        formatted = self.format_dialogue(dialogue)
        
        # Токенизируем
        encoding = self.tokenizer(
            formatted,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Для language modeling используем те же токены как labels
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze().clone()  # Для language modeling
        }
    
    def format_dialogue(self, dialogue) -> str:
        """Форматирует диалог для обучения"""
        if isinstance(dialogue, dict):
            if 'text' in dialogue:
                text = dialogue['text']
            elif 'dialogue' in dialogue:
                text = dialogue['dialogue']
            else:
                # Пробуем получить первый строковый ключ
                for key, value in dialogue.items():
                    if isinstance(value, str) and len(value) > 10:
                        text = value
                        break
                else:
                    text = str(dialogue)
        elif isinstance(dialogue, str):
            text = dialogue
        else:
            text = str(dialogue)
        
        # Очищаем и добавляем специальные токены
        text = text.strip()
        return f"[DIALOGUE_START]\n{text}\n[DIALOGUE_END]"

# ============================================================================
# УТИЛИТЫ ДЛЯ МОНИТОРИНГА
# ============================================================================

class TrainingMonitor:
    """Мониторинг процесса обучения"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f"training_{timestamp}.log"
        self.csv_file = self.log_dir / f"metrics_{timestamp}.csv"
        
        self.metrics = []
        self.start_time = time.time()
        
    def log_step(self, step: int, loss: float, lr: float, phase: str = "train", **kwargs) -> Dict:
        """Логирует шаг обучения"""
        timestamp = datetime.now().isoformat()
        elapsed = time.time() - self.start_time
        
        # Собираем метрики
        metric = {
            'timestamp': timestamp,
            'step': step,
            'loss': float(loss),
            'lr': float(lr),
            'phase': phase,
            'elapsed_seconds': elapsed,
            **kwargs
        }
        
        # Мониторинг ресурсов
        try:
            metric['cpu_percent'] = psutil.cpu_percent()
            metric['ram_gb'] = psutil.virtual_memory().used / (1024**3)
            
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metric['gpu_memory_gb'] = gpu.memoryUsed
                metric['gpu_load'] = gpu.load * 100
                metric['gpu_temp'] = gpu.temperature
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️  Ошибка мониторинга ресурсов: {e}{Style.RESET_ALL}")
        
        self.metrics.append(metric)
        
        # Записываем в лог
        with open(self.log_file, 'a', encoding='utf-8') as f:
            log_line = f"{timestamp} | Step {step:5d} | Loss: {loss:.6f} | LR: {lr:.2e} | Phase: {phase}"
            if 'speed' in kwargs:
                log_line += f" | Speed: {kwargs['speed']:.1f} samples/s"
            f.write(log_line + "\n")
        
        # Периодически сохраняем в CSV
        if step % 10 == 0:
            self.save_metrics()
        
        return metric
    
    def save_metrics(self):
        """Сохраняет метрики в CSV"""
        if self.metrics:
            df = pd.DataFrame(self.metrics)
            df.to_csv(self.csv_file, index=False, encoding='utf-8')
    
    def print_status(self, step: int, total_steps: int, loss: float, lr: float, speed: float = None):
        """Печатает статус обучения"""
        percent = (step / total_steps) * 100 if total_steps > 0 else 0
        
        # Прогресс-бар
        bar_length = 30
        filled = int(bar_length * step // total_steps) if total_steps > 0 else 0
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Цвет в зависимости от прогресса
        if percent < 33:
            color = Fore.RED
        elif percent < 66:
            color = Fore.YELLOW
        else:
            color = Fore.GREEN
        
        # Время
        elapsed = time.time() - self.start_time
        if step > 0 and speed:
            remaining = (total_steps - step) / speed if speed > 0 else 0
            time_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d} | ETA: {int(remaining//60):02d}:{int(remaining%60):02d}"
        else:
            time_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        
        # Статус
        status = f"\r{color}{bar}{Style.RESET_ALL} {percent:5.1f}% | "
        status += f"Step {step:4d}/{total_steps} | "
        status += f"Loss: {loss:.4f} | "
        status += f"LR: {lr:.2e} | "
        status += f"Time: {time_str}"
        
        if speed:
            status += f" | Speed: {speed:.1f} samp/s"
        
        print(status, end='', flush=True)
    
    def final_report(self):
        """Финальный отчет"""
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}🏁 ОБУЧЕНИЕ ЗАВЕРШЕНО{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"Общее время: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"Логи: {self.log_file}")
        print(f"Метрики: {self.csv_file}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")

# ============================================================================
# ОСНОВНОЙ КЛАСС ТРЕНЕРА
# ============================================================================

class PsychAITrainer:
    """Тренер психологического ИИ"""
    
    def __init__(self, config: Dict):
        self.config = config

         # Инициализируем менеджер путей
        self.path_manager = PathManager()
        
        # Определяем базовую директорию
        if output_base_dir:
            self.base_dir = Path(output_base_dir)
        else:
            # Берем из конфига или создаем новую
            base_from_config = config.get('paths', {}).get('base')
            if base_from_config:
                self.base_dir = Path(base_from_config)
            else:
                # Автоматическое создание эксперимента
                exp_name = f"psych_train_{datetime.now().strftime('%Y%m%d_%H%M')}"
                self.base_dir = self.path_manager.create_experiment_dir(
                    base_path="./experiments",
                    experiment_name=exp_name
                )
        
        # Создаем сессию
        self.session_dir = self.path_manager.create_session_dir(self.base_dir)
        
        # Обновляем конфиг с путями этой сессии
        self._update_config_paths()
        
        # Теперь настраиваем устройство
        self.device = self._setup_device()
        
        # Инициализируем мониторинг
        log_dir = self.session_dir / 'logs'
        self.monitor = TrainingMonitor(str(log_dir))
        
        # Компоненты
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
        # Статистика
        self.stats = {
            'best_loss': float('inf'),
            'current_epoch': 0,
            'total_steps': 0,
            'checkpoint_paths': []
        }
        
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}🧠 ИНИЦИАЛИЗАЦИЯ ТРЕНЕРА PSYCH AI{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    
    def _update_config_paths(self):
        """Обновляет пути в конфиге на актуальные"""
        paths = self.path_manager.get_all_paths(self.session_dir)
        
        # Обновляем конфиг
        if 'paths' not in self.config:
            self.config['paths'] = {}
        
        for key, path in paths.items():
            self.config['paths'][key] = str(path)
        
        # Сохраняем конфиг этой сессии
        config_path = self.session_dir / 'configs' / 'training_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def _setup_device(self) -> torch.device:
        """Настраивает устройство для обучения"""
        device_config = self.config.get('system', {}).get('device', 'cuda')
        
        if device_config == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"{Fore.GREEN}✅ Используется GPU: {gpu_name}{Style.RESET_ALL}")
            print(f"   Память: {gpu_memory:.1f} GB")
            print(f"   CUDA версия: {torch.version.cuda}")
        else:
            device = torch.device('cpu')
            print(f"{Fore.YELLOW}⚠️  Используется CPU (CUDA не доступна){Style.RESET_ALL}")
        
        return device
    
    def load_data(self) -> List:
        """Загружает данные для обучения"""
        print(f"\n{Fore.CYAN}📥 ЗАГРУЗКА ДАННЫХ{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*40}{Style.RESET_ALL}")
        
        data_config = self.config.get('data', {})
        data_path = data_config.get('path', '')
        
        if not data_path:
            raise ValueError("Путь к данным не указан в конфигурации")
        
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Файл данных не найден: {data_path}")
        
        # Загружаем данные
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                dialogues = json.load(f)
            
            if not isinstance(dialogues, list):
                raise ValueError("Данные должны быть списком диалогов")
            
            # Ограничиваем количество если нужно
            max_dialogues = data_config.get('max_dialogues')
            if max_dialogues and len(dialogues) > max_dialogues:
                dialogues = dialogues[:max_dialogues]
                print(f"{Fore.YELLOW}⚠️  Ограничено до {max_dialogues} диалогов{Style.RESET_ALL}")
            
            print(f"{Fore.GREEN}✅ Загружено {len(dialogues)} диалогов{Style.RESET_ALL}")
            print(f"   Путь: {data_path}")
            print(f"   Размер файла: {data_path.stat().st_size / 1024 / 1024:.1f} MB")
            
            # Пример диалога
            if dialogues:
                sample = dialogues[0]
                if isinstance(sample, dict) and 'text' in sample:
                    preview = sample['text'][:100] + "..." if len(sample['text']) > 100 else sample['text']
                else:
                    preview = str(sample)[:100] + "..."
                print(f"   Пример: {preview}")
            
            return dialogues
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка парсинга JSON: {e}")
        except Exception as e:
            raise Exception(f"Ошибка загрузки данных: {e}")
    
    def prepare_tokenizer(self):
        """Подготавливает токенизатор"""
        print(f"\n{Fore.CYAN}🔤 ПОДГОТОВКА ТОКЕНИЗАТОРА{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*40}{Style.RESET_ALL}")
        
        model_config = self.config.get('model', {})
        model_name = model_config.get('name', 'EleutherAI/gpt-neo-2.7B')
        
        try:
            print(f"Загрузка токенизатора: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Настраиваем специальные токены
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"{Fore.YELLOW}⚠️  Установлен pad_token = eos_token{Style.RESET_ALL}")
            
            # Добавляем наши специальные токены
            special_tokens = {
                'additional_special_tokens': ['[DIALOGUE_START]', '[DIALOGUE_END]']
            }
            self.tokenizer.add_special_tokens(special_tokens)
            
            print(f"{Fore.GREEN}✅ Токенизатор загружен{Style.RESET_ALL}")
            print(f"   Модель: {model_name}")
            print(f"   Размер словаря: {len(self.tokenizer):,} токенов")
            print(f"   Макс. длина: {self.tokenizer.model_max_length}")
            
            return self.tokenizer
            
        except Exception as e:
            raise Exception(f"Ошибка загрузки токенизатора: {e}")
    
    def prepare_model(self):
        """Подготавливает модель"""
        print(f"\n{Fore.CYAN}🤖 ПОДГОТОВКА МОДЕЛИ{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*40}{Style.RESET_ALL}")
        
        model_config = self.config.get('model', {})
        model_name = model_config.get('name', 'EleutherAI/gpt-neo-2.7B')
        
        try:
            print(f"Загрузка модели: {model_name}...")
            
            # Определяем тип данных
            precision = self.config.get('system', {}).get('precision', 'fp32')
            if precision == 'fp16' and self.device.type == 'cuda':
                torch_dtype = torch.float16
                print(f"   Используется половинная точность (fp16)")
            else:
                torch_dtype = torch.float32
                print(f"   Используется полная точность (fp32)")
            
            # Параметры загрузки
            load_kwargs = {
                'torch_dtype': torch_dtype,
                'device_map': 'auto' if self.device.type == 'cuda' else None,
            }
            
            # Отключаем кэш если используем gradient checkpointing
            if model_config.get('gradient_checkpointing', True):
                load_kwargs['use_cache'] = False
            
            # Загружаем модель
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            )
            
            # Переносим на устройство если не использовали device_map
            if 'device_map' not in load_kwargs or not load_kwargs['device_map']:
                self.model.to(self.device)
            
            # Включаем gradient checkpointing если нужно
            if model_config.get('gradient_checkpointing', True):
                self.model.gradient_checkpointing_enable()
                print(f"   Gradient checkpointing: {Fore.GREEN}Включен{Style.RESET_ALL}")
            
            # Изменяем размер эмбеддингов для новых токенов
            if self.tokenizer and len(self.tokenizer) != self.model.config.vocab_size:
                old_size = self.model.config.vocab_size
                self.model.resize_token_embeddings(len(self.tokenizer))
                print(f"   Размер эмбеддингов изменён: {old_size:,} → {len(self.tokenizer):,}")
            
            # Считаем параметры
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"{Fore.GREEN}✅ Модель загружена{Style.RESET_ALL}")
            print(f"   Параметры: {total_params:,} (обучаемых: {trainable_params:,})")
            print(f"   Слои: {len(list(self.model.parameters()))}")
            
            return self.model
            
        except Exception as e:
            raise Exception(f"Ошибка загрузки модели: {e}")
    
    def create_datasets(self, dialogues: List):
        """Создает тренировочный и валидационный датасеты"""
        print(f"\n{Fore.CYAN}📊 СОЗДАНИЕ ДАТАСЕТОВ{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*40}{Style.RESET_ALL}")
        
        # Параметры
        data_config = self.config.get('data', {})
        tokenization_config = self.config.get('tokenization', {})
        
        train_split = data_config.get('train_split', 0.85)
        val_split = data_config.get('val_split', 0.15)
        max_length = tokenization_config.get('max_length', 512)
        
        # Разделение данных
        n_total = len(dialogues)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        # Проверяем split
        if n_train + n_val > n_total:
            n_val = n_total - n_train
        
        train_dialogues = dialogues[:n_train]
        val_dialogues = dialogues[n_train:n_train + n_val]
        
        print(f"Всего диалогов: {n_total:,}")
        print(f"Тренировочных: {n_train:,} ({train_split*100:.0f}%)")
        print(f"Валидационных: {n_val:,} ({val_split*100:.0f}%)")
        
        # Создаем датасеты
        self.train_dataset = PsychDialogueDataset(
            train_dialogues, self.tokenizer, max_length
        )
        
        self.val_dataset = PsychDialogueDataset(
            val_dialogues, self.tokenizer, max_length
        )
        
        print(f"{Fore.GREEN}✅ Датасеты созданы{Style.RESET_ALL}")
        print(f"   Макс. длина токенов: {max_length}")
        
        return self.train_dataset, self.val_dataset
    
    def create_dataloaders(self):
        """Создает DataLoader'ы"""
        print(f"\n{Fore.CYAN}🔄 СОЗДАНИЕ DATALOADERS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*40}{Style.RESET_ALL}")
        
        training_config = self.config.get('training', {})
        batch_size = training_config.get('batch_size', 3)
        grad_accumulation = training_config.get('gradient_accumulation', 1)
        
        # Создаем DataLoader для тренировки
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # 0 для избежания проблем с Windows
            pin_memory=self.device.type == 'cuda'
        )
        
        # Создаем DataLoader для валидации
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=max(1, batch_size // 2),  # Меньший batch для валидации
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == 'cuda'
        )
        
        # Считаем общее количество шагов
        epochs = training_config.get('epochs', 3)
        steps_per_epoch = len(self.train_loader) // grad_accumulation
        if len(self.train_loader) % grad_accumulation != 0:
            steps_per_epoch += 1
        
        total_steps = steps_per_epoch * epochs
        self.stats['total_steps'] = total_steps
        
        print(f"{Fore.GREEN}✅ DataLoader'ы созданы{Style.RESET_ALL}")
        print(f"   Batch size: {batch_size}")
        print(f"   Gradient accumulation: {grad_accumulation}")
        print(f"   Шагов на эпоху: {steps_per_epoch:,}")
        print(f"   Всего шагов: {total_steps:,}")
        print(f"   Batches: train={len(self.train_loader)}, val={len(self.val_loader)}")
        
        return self.train_loader, self.val_loader
    
    def prepare_optimizer(self):
        """Подготавливает оптимизатор и шедулер"""
        print(f"\n{Fore.CYAN}⚡ ПОДГОТОВКА ОПТИМИЗАТОРА{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*40}{Style.RESET_ALL}")
        
        training_config = self.config.get('training', {})
        
        # Параметры оптимизатора
        lr = training_config.get('learning_rate', 2e-4)
        weight_decay = training_config.get('weight_decay', 0.01)
        
        # Оптимизатор
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Шедулер
        warmup_ratio = training_config.get('warmup_ratio', 0.9)
        warmup_steps = int(self.stats['total_steps'] * warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.stats['total_steps']
        )
        
        print(f"{Fore.GREEN}✅ Оптимизатор настроен{Style.RESET_ALL}")
        print(f"   Алгоритм: AdamW")
        print(f"   Learning rate: {lr:.2e}")
        print(f"   Weight decay: {weight_decay}")
        print(f"   Warmup steps: {warmup_steps:,} ({warmup_ratio*100:.0f}%)")
        
        return self.optimizer, self.scheduler
    
    def save_checkpoint(self, step: int, loss: float, is_best: bool = False):
        """Сохраняет чекпоинт"""
        checkpoint_config = self.config.get('checkpoint', {})
        checkpoint_dir = Path(checkpoint_config.get('dir', './checkpoints'))
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Имя чекпоинта
        if is_best:
            checkpoint_name = f"best_model_step_{step}_loss_{loss:.4f}"
        else:
            checkpoint_name = f"checkpoint_step_{step}_loss_{loss:.4f}"
        
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        # Сохраняем состояние
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config,
            'stats': self.stats,
            'tokenizer_config': self.tokenizer.get_vocab() if self.tokenizer else None,
        }, checkpoint_path)
        
        # Добавляем в список
        self.stats['checkpoint_paths'].append(str(checkpoint_path))
        
        # Ограничиваем количество чекпоинтов
        save_total_limit = checkpoint_config.get('save_total_limit', 3)
        if len(self.stats['checkpoint_paths']) > save_total_limit:
            # Удаляем самый старый
            oldest = self.stats['checkpoint_paths'].pop(0)
            try:
                Path(oldest).unlink()
            except:
                pass
        
        if is_best:
            print(f"{Fore.GREEN}💾 Лучший чекпоинт сохранён: {checkpoint_path.name}{Style.RESET_ALL}")
        else:
            print(f"{Fore.BLUE}💾 Чекпоинт сохранён: {checkpoint_path.name}{Style.RESET_ALL}")
        
        return checkpoint_path
    
    def train_epoch(self, epoch: int) -> float:
        """Обучает одну эпоху"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}📚 ЭПОХА {epoch}/{self.config.get('training', {}).get('epochs', 3)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        training_config = self.config.get('training', {})
        grad_accumulation = training_config.get('gradient_accumulation', 1)
        max_grad_norm = training_config.get('max_grad_norm', 1.0)
        
        # Прогресс бар
        pbar = tqdm(self.train_loader, desc=f"Эпоха {epoch}", 
                   bar_format="{l_bar}{bar:30}{r_bar}", 
                   colour="green")
        
        start_time = time.time()
        accumulation_steps = 0
        
        for batch_idx, batch in enumerate(pbar):
            # Переносим batch на устройство
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss = loss / grad_accumulation  # Нормализуем loss для accumulation
            
            # Backward pass
            loss.backward()
            
            # Накопление градиентов
            accumulation_steps += 1
            if accumulation_steps % grad_accumulation == 0:
                # Обрезаем градиенты
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                # Шаг оптимизатора
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Считаем статистику
                current_step = self.stats['current_epoch'] * len(self.train_loader) + batch_idx
                current_lr = self.scheduler.get_last_lr()[0]
                
                # Мониторинг
                speed = batch_size / (time.time() - start_time) if batch_idx > 0 else 0
                start_time = time.time()
                
                self.monitor.log_step(
                    step=current_step,
                    loss=loss.item() * grad_accumulation,  # Возвращаем оригинальное значение
                    lr=current_lr,
                    phase="train",
                    speed=speed,
                    epoch=epoch,
                    batch=batch_idx
                )
                
                # Обновляем progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item() * grad_accumulation:.4f}",
                    'lr': f"{current_lr:.2e}",
                    'speed': f"{speed:.1f}/s"
                })
            
            total_loss += loss.item() * grad_accumulation * input_ids.size(0)
            total_samples += input_ids.size(0)
        
        # Завершаем оставшиеся градиенты
        if accumulation_steps % grad_accumulation != 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        print(f"{Fore.GREEN}✅ Эпоха {epoch} завершена{Style.RESET_ALL}")
        print(f"   Средний loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self) -> float:
        """Валидация модели"""
        print(f"\n{Fore.CYAN}🧪 ВАЛИДАЦИЯ{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*40}{Style.RESET_ALL}")
        
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            val_bar = tqdm(self.val_loader, desc="Валидация", 
                          bar_format="{l_bar}{bar:30}{r_bar}", 
                          colour="yellow")
            
            for batch_idx, batch in enumerate(val_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)
                
                # Обновляем progress bar
                current_loss = total_loss / total_samples if total_samples > 0 else 0
                val_bar.set_postfix({'loss': f"{current_loss:.4f}"})
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        print(f"{Fore.GREEN}✅ Валидация завершена{Style.RESET_ALL}")
        print(f"   Val loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def train(self):
        """Основной цикл обучения"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}🚀 НАЧАЛО ОБУЧЕНИЯ{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        # Загружаем данные
        dialogues = self.load_data()
        
        # Подготавливаем компоненты
        self.prepare_tokenizer()
        self.prepare_model()
        self.create_datasets(dialogues)
        self.create_dataloaders()
        self.prepare_optimizer()
        
        # Параметры обучения
        training_config = self.config.get('training', {})
        epochs = training_config.get('epochs', 3)
        checkpoint_config = self.config.get('checkpoint', {})
        
        save_steps = checkpoint_config.get('save_steps', 100)
        load_best = checkpoint_config.get('load_best_model_at_end', True)
        patience = checkpoint_config.get('early_stopping', {}).get('patience', 3)
        
        # Статистика
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Главный цикл обучения
        for epoch in range(1, epochs + 1):
            self.stats['current_epoch'] = epoch
            
            # Тренировка
            train_loss = self.train_epoch(epoch)
            
            # Валидация
            val_loss = self.validate()
            
            # Проверяем лучшую модель
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                self.stats['best_loss'] = best_val_loss
                patience_counter = 0
                print(f"{Fore.GREEN}🎯 Новый лучший результат: {best_val_loss:.4f}{Style.RESET_ALL}")
                
                # Сохраняем лучшую модель
                self.save_checkpoint(
                    step=epoch * len(self.train_loader),
                    loss=best_val_loss,
                    is_best=True
                )
            else:
                patience_counter += 1
                print(f"{Fore.YELLOW}⚠️  Падение качества, patience: {patience_counter}/{patience}{Style.RESET_ALL}")
            
            # Ранняя остановка
            if patience_counter >= patience:
                print(f"{Fore.RED}🛑 Ранняя остановка на эпохе {epoch}{Style.RESET_ALL}")
                break
            
            # Регулярное сохранение
            if epoch % max(1, save_steps // len(self.train_loader)) == 0:
                self.save_checkpoint(
                    step=epoch * len(self.train_loader),
                    loss=val_loss,
                    is_best=False
                )
        
        # Финальный отчет
        self.monitor.final_report()
        
        # Сохраняем финальную модель если нужно
        if load_best:
            print(f"\n{Fore.CYAN}💾 Сохранение финальной модели...{Style.RESET_ALL}")
            self.save_checkpoint(
                step=epochs * len(self.train_loader),
                loss=best_val_loss,
                is_best=True
            )
        
        print(f"\n{Fore.GREEN}✨ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО ✨{Style.RESET_ALL}")
        
        return best_val_loss

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Главная функция запуска обучения"""
    parser = argparse.ArgumentParser(description="Тренер психологического ИИ")
    parser.add_argument("--config", type=str, default="./configs/base.json", 
                       help="Путь к конфигурационному файлу")
    parser.add_argument("--preset", type=str, default=None,
                       help="Пресет конфигурации (fast, quality, debug)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Путь к чекпоинту для возобновления обучения")
    args = parser.parse_args()
    
    try:
        # Инициализация
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{Style.BRIGHT}🧠 RUZANNA - PSYCHOLOGICAL AI TRAINER{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        # Загрузка конфигурации
        print(f"{Fore.BLUE}📄 Загрузка конфигурации...{Style.RESET_ALL}")
        config_manager = ConfigManager("./configs")
        
        # Загружаем с учетом пресета
        config = config_manager.load_full_config(preset=args.preset)
        
        # Получаем параметры
        params = config_manager.get_training_params()
        
        print(f"{Fore.GREEN}✅ Конфигурация загружена{Style.RESET_ALL}")
        print(f"   Модель: {params.get('model_name')}")
        print(f"   Данные: {Path(params.get('data_path', '')).name}")
        print(f"   Batch size: {params.get('batch_size')}")
        print(f"   Эпохи: {params.get('epochs')}")
        print(f"   Learning rate: {params.get('learning_rate'):.2e}")
        
        # Создаем тренер
        trainer = PsychAITrainer(config)
        
        # Запускаем обучение
        best_loss = trainer.train()
        
        print(f"\n{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}🏆 ЛУЧШИЙ РЕЗУЛЬТАТ: {best_loss:.4f}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️  Обучение прервано пользователем{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}❌ Критическая ошибка: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    main()
