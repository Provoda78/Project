from Load_Image import ImageLoad
from abc import ABC, abstractmethod
from typing import Callable
import cv2
import numpy as np

class Noises(ABC):
    def __init__(self, image_loader: Callable = None):
        self.image_loader = image_loader 
        self.original_array = None
        self.noisy_image = None
        self.noise_params = {}
        
        if image_loader is not None and image_loader.image_array is not None:
            self.original_array = image_loader.image_array.copy()
            
class Gaussian(Noises):
    def __init__(self, image_loader = None):
        super().__init__(image_loader)
        
    def add_gaussian_noise(self, mean=0, sigma=25):
        
        if self.original_array is None:
            raise ValueError("Сначала загрузите изображение") from None
        
        noise = np.random.normal(mean, sigma, self.original_array.shape)
        
        noisy = self.original_array.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        self.noisy_image = noisy
        self.noise_params['gaussian'] = {'mean': mean, 'sigma': sigma}
        
        print(f"Добавлен гауссов шум (Среднее отклонение = {sigma})")
        
        return noisy
    
    def add_guissian_blur(self, kernel_size=(5,5), sigma=0):
        if self.original_array is None:
            raise ValueError("Сначала загрузите изображение") from None
        
        blur = cv2.GaussianBlur(src=self.original_array, ksize=kernel_size, sigmaX=sigma)
        print(f"Добавлен фильтр по Гауссу (Размер ядра = {kernel_size})")
        
        return blur
    
class Salt_pepper_noise(Noises):
    def __init__(self, image_loader = None):
        super().__init__(image_loader)
        
    def add_salt_pepper_noise(self, salt_prob=0.01, pepper_prob=0.01, salt_value=255, pepper_value=0):
        if self.original_array is None:
            raise ValueError("Сначала загрузите изображение через set_image()")
        
        noisy = self.original_array.copy()
        
        random_mask = np.random.random(noisy.shape[:2])
        
        is_color = len(noisy.shape) == 3
        channels = noisy.shape[2] if is_color else 1
        
        # Добавляем "соль"
        salt_mask = random_mask < salt_prob
        if is_color:
            noisy[salt_mask] = [salt_value] * channels
        else:
            noisy[salt_mask] = salt_value
        
        # Добавляем "перец"
        pepper_mask = (random_mask >= 1 - pepper_prob) & (~salt_mask)
        if is_color:
            noisy[pepper_mask] = [pepper_value] * channels
        else:
            noisy[pepper_mask] = pepper_value
        
        self.noisy_image = noisy
        self.noise_params['salt_pepper'] = {
            'salt_prob': salt_prob,
            'pepper_prob': pepper_prob,
            'salt_value': salt_value,
            'pepper_value': pepper_value
        }
        
        total_pixels = noisy.shape[0] * noisy.shape[1]
        salt_count = np.sum(salt_mask)
        pepper_count = np.sum(pepper_mask)
        
        print(f"Добавлен шум соль/перец:")
        print(f"  - Соль: {salt_count} пикселей ({salt_count/total_pixels*100:.3f}%)")
        print(f"  - Перец: {pepper_count} пикселей ({pepper_count/total_pixels*100:.3f}%)")
        
        return noisy
    
    def add_median_blure(self, kernel_size):
        if self.original_array is None:
            raise ValueError("Сначала загрузите изображение") from None
        
        blur = cv2.medianBlur(src=self.original_array, ksize=kernel_size)
        print(f"Добавлен Медианный фильтрм(Размер ядра = {kernel_size})")
        
        return blur
    
class  Geometric_filters(ABC):
    @abstractmethod
    def generate_maps(self, width, height):
        """Возвращает x_map и y_map для cv2.remap"""
        pass

class WaveFilter(Geometric_filters):
    def __init__(self, intensity=0.5, frequency=0.1, direction='horizontal'):
        self.amplitude = intensity * 5
        self.frequency = frequency
        self.direction = direction

    def generate_maps(self, width, height):
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x_map, y_map = x.astype(np.float32), y.astype(np.float32)

        if self.direction == 'horizontal':
            x_map += self.amplitude * np.sin(2 * np.pi * y * self.frequency)
        else:
            y_map += self.amplitude * np.sin(2 * np.pi * x * self.frequency)
        return x_map, y_map

class SwirlFilter(Geometric_filters):
    def __init__(self, intensity=0.5, radius=None, center=None):
        self.strength = intensity * 5
        self.radius = radius
        self.center = center

    def generate_maps(self, width, height):
        center = self.center or (width // 2, height // 2)
        radius = self.radius or min(width, height) * 0.8
        
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        dx, dy = x - center[0], y - center[1]
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)

        swirl = np.maximum(0, self.strength * (1 - r / radius))
        x_map = center[0] + r * np.cos(theta + swirl)
        y_map = center[1] + r * np.sin(theta + swirl)
        return x_map.astype(np.float32), y_map.astype(np.float32)

#Основной класс для работы с изображением
class ImageProcessor:
    def __init__(self, image_array):
        self.original = image_array
        self.processed = None

    def apply_filter(self, geo_filter: Geometric_filters):
        h, w = self.original.shape[:2]
        x_map, y_map = geo_filter.generate_maps(w, h)
        
        x_map = np.clip(x_map, 0, w - 1)
        y_map = np.clip(y_map, 0, h - 1)
        
        self.processed = cv2.remap(self.original, x_map, y_map, cv2.INTER_LINEAR)
        return self.processed