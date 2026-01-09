
USE_WANDB = False  # По умолчанию выключено

try:
    import wandb
    wandb.init(project="psycho-bot")
    USE_WANDB = True
except:
    print("WandB не доступен")

class TrainingLogger:
	"""Только логирование в CSV с буферизацией"""
	
	def __init__(self, log_dir, debug_mode=1):
		self.log_dir = Path(log_dir)
		self.debug_mode = debug_mode
		self.log_buffer = []
		self.buffer_limit = 10
		
	def log(self, step: int, loss: float, lr: float, memory_gb: float, 
        phase: str, perplexity: float = None, empathy_score: float = None) -> None:
    	"""Добавление записи в буфер с типами"""
		perp_str = f"{perplexity:.2f}" if perplexity is not None else ""
		empathy_str = f"{empathy_score:.3f}" if empathy_score is not None else ""
		line = f"{datetime.now().isoformat()},{step},{loss:.6f},{lr:.6f},{memory_gb:.1f},{phase},{perp_str},{empathy_str}\n"
		self.log_buffer.append(line)
		
		if len(self.log_buffer) >= self.buffer_limit:
			self._flush()
	
	def _flush(self):
		"""Сброс буфера на диск"""
		if not self.log_buffer:
			return
		
		csv_file = self.log_dir / "training_log.csv"
		try:
			write_header = not csv_file.exists()
			with open(csv_file, 'a', encoding='utf-8', newline='') as f:
				if write_header:
					f.write("timestamp,step,loss,lr,memory_gb,phase,perplexity,empathy_score\n")
				f.writelines(self.log_buffer)
			self.log_buffer.clear()
		except Exception as e:
			if self.debug_mode >= 2:
				print(f"   ❌ Ошибка записи лога: {e}")
	
	def flush(self):
		"""Принудительный сброс (вызывать в конце эпохи)"""
		self._flush()

class QualityEvaluator:
	"""Только оценка качества ответов"""
	
	def __init__(self):
		self.empathy_words = [
			"понимаю", "чувствую", "важно", "ценю", "принимаю", 
			"спасибо", "слышу", "вижу", "замечаю", "уважаю"
		]
		self.advice_words = [
			"должен", "надо", "обязан", "рекомендую", 
			"советую", "следует", "нужно", "стоит"
		]
	
	def calculate_empathy_score(self, text: str) -> float:
    	"""Расчет оценки эмпатии"""
		if not text:
			return 0.0
		text_lower = text.lower()
		empathy_count = sum(1 for word in self.empathy_words if word in text_lower)
		max_empathy = min(len(self.empathy_words), 5)
		return min(empathy_count / max_empathy, 1.0)
	
	def evaluate_response(self, prompt: str, response: str) -> float:
    	"""Комплексная оценка качества ответа"""
		if not response:
			return 0.0
		
		score = 0.0
		words = response.split()
		
		# Длина
		if 5 <= len(words) <= 80:
			score += 1.0
		
		# Эмпатия
		score += self.calculate_empathy_score(response)
		
		# Вопросы
		if '?' in response:
			score += 1.0
		
		# Отсутствие советов
		if not any(word in response.lower() for word in self.advice_words):
			score += 1.0
		
		# Уникальность слов
		if len(words) > 5:
			unique_ratio = len(set(words)) / len(words)
			if unique_ratio > 0.6:
				score += 1.0
		
		# Релевантность
		prompt_words = set(prompt.lower().split()[:10])
		response_words = set(response.lower().split())
		if len(prompt_words.intersection(response_words)) >= 1:
			score += 1.0
		
		# Структура
		if '.' in response:
			score += 0.5
		
		return min(score / 7.5, 1.0)

class ResponseGenerator:
	"""Только генерация ответов модели"""
	
	def __init__(self, tokenizer):
		self.tokenizer = tokenizer
	
	def generate_response(self, model, prompt, temperature=0.729):
		"""Генерация ответа с адаптивными параметрами"""
		try:
			full_prompt = f"{prompt}\n\nПсихолог:"
			inputs = self.tokenizer(
				full_prompt, 
				return_tensors="pt", 
				max_length=512, 
				truncation=True
			).to(model.device)
			
			with torch.no_grad():
				outputs = model.generate(
					**inputs,
					max_new_tokens=256,
					temperature=temperature,
					do_sample=True,
					top_p=0.95 if temperature > 0.8 else 0.9,
					top_k=100,
					pad_token_id=self.tokenizer.eos_token_id,
					repetition_penalty=1.1
				)
			
			response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
			response = response[len(full_prompt):].strip()
			return self.clean_response(response)
			
		except Exception as e:
			return f"Ошибка генерации: {str(e)[:50]}"
	
	def clean_response(self, text):
		"""Очистка ответа от артефактов"""
		if not text:
			return ""
		text = text.replace('�', '').replace('\x00', '')
		stops = ['\nПациент:', '\nПсихолог:', '\n---', '\n===']
		for stop in stops:
			if stop in text:
				text = text.split(stop)[0].strip()
		return ' '.join(text.split())

# ================= ПРОДВИНУТЫЙ МОНИТОРИНГ =================
class AdvancedTrainingMonitor:
	"""
	Координатор мониторинга обучения психологической модели.
	
	Объединяет логирование, оценку качества и генерацию ответов.
	Использует отдельные компоненты для соблюдения SRP.
	
	Attributes:
		logger (TrainingLogger): Логирование в CSV
		evaluator (QualityEvaluator): Оценка качества ответов
		generator (ResponseGenerator): Генерация ответов модели
	"""
	
	def __init__(self, log_dir, tokenizer, debug_mode=1):
		"""
		Инициализация мониторинга.
		
		Args:
			log_dir (str/Path): Директория для логов
			tokenizer: Токенизатор для генерации ответов
			debug_mode (int): Уровень детализации (1-3)
		"""
		self.logger = TrainingLogger(log_dir, debug_mode)
		self.evaluator = QualityEvaluator()
		self.generator = ResponseGenerator(tokenizer)
		self.debug_mode = debug_mode
		
	# Простые делегирующие методы
	def log_batch(self, step, loss, lr, grad_norm=None, memory_gb=None, 
			  step_time=None, phase="TRAIN", perplexity=None, empathy_score=None):
		"""
	Логирование метрик одного шага обучения.
	
	Args:
		step (int): Номер шага обучения
		loss (float): Значение функции потерь
		lr (float): Текущая скорость обучения
		grad_norm (float, optional): Норма градиента
		memory_gb (float, optional): Использование памяти GPU в ГБ
		step_time (float, optional): Время выполнения шага в секундах
		phase (str): Фаза обучения ('TRAIN', 'VALID', 'WARMUP', etc.)
		perplexity (float, optional): Значение perplexity
		empathy_score (float, optional): Оценка эмпатии ответа
	"""
		if USE_WANDB:
    		import wandb
    		wandb.log({"loss": loss, "step": step})
		self.logger.log(*args, **kwargs)
	
	def calculate_empathy_score(self, text):
		return self.evaluator.calculate_empathy_score(text)
	
	def evaluate_response_comprehensive(self, prompt, response):
		return self.evaluator.evaluate_response(prompt, response)
	
	def generate_adaptive_response(self, model, prompt, temperature):
		return self.generator.generate_response(model, prompt, temperature)
	
	# Сложные методы, использующие несколько компонентов
	def advanced_quality_check(self, model, step, adaptive_temp=True):
		"""
	Выполняет комплексную проверку качества модели.
	
	Генерирует ответы на тестовые промпты и оценивает их качество.
	Использует адаптивную температуру на основе предыдущих результатов.
	
	Args:
		model: Модель для тестирования
		step (int): Текущий шаг обучения (для логов)
		adaptive_temp (bool): Использовать адаптивную температуру
	
	Returns:
		tuple: (средний_скор, средняя_эмпатия, использованная_температура)
	"""
		test_prompts = [
			"Пациент: Не могу перестать волноваться.",
			"Пациент: Чувствую себя очень одиноко.",
			"Пациент: Как найти смысл в жизни?"
		]
		
		scores = []
		for prompt in test_prompts:
			response = self.generator.generate_response(model, prompt, 0.729)
			score = self.evaluator.evaluate_response(prompt, response)
			scores.append(score)
		
		return sum(scores) / len(scores) if scores else 0
	
	# Остальные методы остаются, но используют компоненты
	def flush(self):
		self.logger.flush()
