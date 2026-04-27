import cv2
import numpy as np
import importlib
import inspect
from typing import Type, Dict, List
from .Effects import BaseEffect

class ImageLoader:
    def __init__(self):
        self.image_array = None
        self.info = {}

    def load(self, file_path: str):
        self.image_array = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if self.image_array is None:
            raise FileNotFoundError(f"Не удалось загрузить: {file_path}")
        
        h, w, c = self.image_array.shape
        self.info = {'width': w, 'height': h, 'channels': c, 'path': file_path}
        return self.image_array

    def save(self, file_path: str, image: np.ndarray):
        cv2.imwrite(file_path, image)
        print(f"Файл сохранен: {file_path}")
        

class NoiseLoader:
    def __init__(self, library_name: str = "noise_library"):
        
        self.library_name = library_name
        self._cache: Dict[str, Type[BaseEffect]] = {}

    def _get_module(self):
        try:
            return importlib.import_module(self.library_name)
        except ImportError:
            raise ImportError(f"Библиотека {self.library_name} не найдена. ")

    def load_class(self, class_name: str) -> Type[BaseEffect]:
        """Динамически импортирует класс по имени"""

        module = self._get_module()
        klass_ = getattr(module, class_name, None)

        # Проверка: существует ли класс и является ли он наследником BaseEffect
        if klass_ is None:
            raise AttributeError(f"Класс {class_name} не найден в {self.library_name}")
        
        if not inspect.isclass(klass_) or not issubclass(klass_, BaseEffect):
            raise TypeError(f"Объект {class_name} не является валидным эффектом")

        return klass_

    def create(self, class_name: str, **kwargs) -> BaseEffect:
        """Создает экземпляр класса с переданными параметрами"""
        klass = self.load_class(class_name)
        return klass(**kwargs)

    def list_available(self) -> List[str]:
        """Автоматически находит все доступные эффекты в библиотеке"""
        module = self._get_module()
        return [
            name for name, obj in inspect.getmembers(module)
            if inspect.isclass(obj) and issubclass(obj, BaseEffect) and obj is not BaseEffect
        ]