from Load_Image import ImageLoad
from abc import ABC, abstractmethod
from typing import Callable
import cv2
import numpy as np

class Noises(ABC):
    def __init__(self, image_loader: Callable = None):
        self.image_loader = image_loader 
        self.original_array = None
        self.histogramm = None
        self.noise_params = {}
        
        if image_loader is not None and image_loader.image_array is not None:
            self.original_array = image_loader.image_array.copy()
            
    def save_noisy(self, filename):
        if self.histogramm is not None:
            cv2.imwrite(filename, self.histogramm)
            print(f"Зашумленное изображение сохранено: {filename}")
        else:
            print("Нет зашумленного изображения для сохранения")
            
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
    
class Chess_mixed(Noises):
    def __init__(self, image_loader = None):
        super().__init__(image_loader)
        
    def _create_chessboard_mask(self, height: int, width: int, cell_size: int) -> np.ndarray:
        """
        Создает бинарную маску шахматного узора
        """
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Создаем шахматный узор
        for i in range(height):
            for j in range(width):
                cell_i = i // cell_size
                cell_j = j // cell_size
                
                if (cell_i + cell_j) % 2 == 0:
                    mask[i, j] = 0.0  # Первое изображение
                else:
                    mask[i, j] = 1.0  # Второе изображение
        
        return mask
    
    def _create_smooth_mask(self, binary_mask: np.ndarray, blend_width: int) -> np.ndarray:
        """
        Создает сглаженную маску с линейным градиентом на границах ячеек
        """
        if blend_width <= 0:
            return binary_mask
        
        height, width = binary_mask.shape
        smooth_mask = binary_mask.copy()
        
        kernel_size = blend_width * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
        
        smooth_mask = cv2.GaussianBlur(smooth_mask, (kernel_size, kernel_size), blend_width)
        
        smooth_mask = np.clip(smooth_mask, 0, 1)
        
        return smooth_mask
    
    def create_chessboard_blend(self, loader1: Callable, loader2: Callable, 
                                cell_size: int = 50, blend_width: int = 10) -> np.ndarray:


        image1, image2 = loader1.image_array, loader2.image_array
        
        if image1 is None or image2 is None:
            raise ValueError("Оба изображения должны быть загружены")
        
        h, w = image1.shape[:2]
        if image2.shape[:2] != (h, w):
            image2 = cv2.resize(image2, (w, h))
        
        
        if len(image1.shape) == 3:
            result = np.zeros_like(image1, dtype=np.float32)
        else:
            result = np.zeros((h, w), dtype=np.float32)
        
        chess_mask = self._create_chessboard_mask(h, w, cell_size)
        
        if blend_width > 0:
            smooth_mask = self._create_smooth_mask(chess_mask, blend_width)
        else:
            smooth_mask = chess_mask.astype(np.float32)
        
        for i in range(3 if len(image1.shape) == 3 else 1):
            if len(image1.shape) == 3:
                result[:, :, i] = (image1[:, :, i] * (1 - smooth_mask) + 
                                   image2[:, :, i] * smooth_mask)
            else:
                result = image1 * (1 - smooth_mask) + image2 * smooth_mask
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        self.noisy_image = result
        self.noise_params['chessboard_blend'] = {
            'cell_size': cell_size,
            'blend_width': blend_width,
            'image1_shape': image1.shape,
            'image2_shape': image2.shape
        }
        
        print(f"Создан шахматный узор с линейным смешиванием:")
        print(f"  - Размер ячейки: {cell_size} px")
        print(f"  - Ширина зоны смешивания: {blend_width} px")
    
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
    
    
class Histogramm(Noises):
    def __init__(self, image_loader = None):
        super().__init__(image_loader)
        self.histogramm = None
        self.global_equalization = None
        
    def _calculate_cdf(self, hist: np.ndarray) -> np.ndarray:
        """
        Вычисляет кумулятивную функцию распределения (CDF)
        """
        cdf = hist.cumsum()
        
        # Нормализуем CDF
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        
        return cdf_normalized
    
    def _calculate_histogram(self, loader: np.ndarray) -> np.ndarray:
        """
        Вычисляет гистограмму изображения
        """
        image = loader.image_array
        
        hist = np.zeros(256, dtype=np.float32)
        
        # Заполняем гистограмму
        for pixel in image.flatten():
            hist[pixel] += 1
        
        # Нормализуем гистограмму
        hist = hist / image.size
        
        self.histogramm = hist
        return hist
    
    def _global_equalization(self, loader: Callable) -> np.ndarray:
        
        image = loader.image_array
        hist = self._calculate_histogram(image)
        
        cdf = hist.cumsum()
        
        #Нормализуем CDF к диапазону [0, 255]
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        cdf_normalized = cdf_normalized.astype(np.uint8)
        
        #Применяем преобразование
        equalized = cdf_normalized[image]
        
        self.global_equalization = equalized
    
    def save_hist(self, filename):
        if self.histogramm is not None:
            cv2.imwrite(filename, self.histogramm)
            print(f"Гистограмма изображения сохранена: {filename}")
        else:
            print("Нет гистограммы изображения")
            
    def save_equaliz(self, filename):
        if self.global_equalization is not None:
            cv2.imwrite(filename, self.global_equalization)
            print(f"Глобальная эквализация изображения сохранена: {filename}")
        else:
            print("Нет глобальной эквализации изображения")
        
    
if __name__ == '__main__':
    
    loader1 = ImageLoad()
    loader1.load("download.jpeg")
    
    loader2 = ImageLoad()
    loader2.load("download.png")
    
    hits = Histogramm()
    hits._calculate_histogram(loader1)
    
    hits.save_hist("Pictures/histogramm.png")