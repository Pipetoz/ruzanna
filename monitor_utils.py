class BaseModelMode:
	"""Базовый класс для контекстных менеджеров режимов модели"""
	
	def __init__(self, model):
		self.model = model
		self.original_cache = None
		self.original_training = None
		self.original_gc = None
	
	def __enter__(self):
		# Сохраняем оригинальные состояния
		self.original_cache = self.model.config.use_cache
		self.original_training = self.model.training
		self.original_gc = self.model.is_gradient_checkpointing
		
		# Применяем целевые настройки
		self.model.config.use_cache = self.target_cache
		if self.target_training is not None:
			self.model.train(self.target_training)
		self._set_gradient_checkpointing()
		return self
	
	def _set_gradient_checkpointing(self):
		"""Установка gradient checkpointing (можно переопределить)"""
		pass
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		# Восстанавливаем оригинальные состояния
		self.model.config.use_cache = self.original_cache
		if self.original_training is not None:
			self.model.train(self.original_training)
		self._restore_gradient_checkpointing()
	
	def _restore_gradient_checkpointing(self):
		"""Восстановление gradient checkpointing"""
		if self.original_gc and not self.model.is_gradient_checkpointing:
			self.model.gradient_checkpointing_enable()
		elif not self.original_gc and self.model.is_gradient_checkpointing:
			try:
				self.model.gradient_checkpointing_disable()
			except:
				self.model.gradient_checkpointing = False

class TrainingMode(BaseModelMode):
	"""Режим обучения: cache=OFF, gradient_checkpointing=ON"""
	
	def __init__(self, model):
		super().__init__(model)
		self.target_cache = False
		self.target_training = True
	
	def _set_gradient_checkpointing(self):
		self.model.gradient_checkpointing_enable()

class ValidationMode(BaseModelMode):
	"""Режим валидации: cache=OFF, gradient_checkpointing=OFF"""
	
	def __init__(self, model):
		super().__init__(model)
		self.target_cache = False
		self.target_training = False
	
	def _set_gradient_checkpointing(self):
		if self.model.is_gradient_checkpointing:
			try:
				self.model.gradient_checkpointing_disable()
			except:
				self.model.gradient_checkpointing = False

class GenerationMode(BaseModelMode):
	"""Режим генерации: cache=ON, gradient_checkpointing=OFF"""
	
	def __init__(self, model):
		super().__init__(model)
		self.target_cache = True
		self.target_training = False
	
	def _set_gradient_checkpointing(self):
		if self.model.is_gradient_checkpointing:
			try:
				self.model.gradient_checkpointing_disable()
			except:
				self.model.gradient_checkpointing = False
