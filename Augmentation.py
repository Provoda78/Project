from Load_Image import ImageLoad
from typing import Callable
import cv2
import numpy as np
from enum import Enum

class GeometricFilterType(Enum):
    WAVE = "wave"
    SWIRL = "swirl"

class ImageNoiser:
    def __init__(self, image_loader: Callable = None):
        self.image_loader = image_loader 
        self.original_array = None
        self.noisy_image = None
        self.noise_params = {}
        
        if image_loader is not None and image_loader.image_array is not None:
            self.original_array = image_loader.image_array.copy()
            
    def add_gaussian_noise(self, mean=0, sigma=25):
        
        """
        Добавляет гауссов шум
        mean: среднее значение шума
        sigma:  среднее квадратичное отклонение
        """
        
        if self.original_array is None:
            raise ValueError("Сначала загрузите изображение") from None
        
        noise = np.random.normal(mean, sigma, self.original_array.shape)
        
        noisy = self.original_array.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        self.noisy_image = noisy
        self.noise_params['gaussian'] = {'mean': mean, 'sigma': sigma}
        
        print(f"Добавлен гауссов шум (Среднее отклонение = {sigma})")
        
        return noisy
    
    
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
    
    def add_geometric_filter(self, filter_type=GeometricFilterType.WAVE, intensity=0.5, center=None, **kwargs):
        if self.original_array is None:
            raise ValueError("Сначала загрузите изображение")
        
        height, width = self.original_array.shape[:2]
        
        # Центр искажения по умолчанию
        if center is None:
            center = (width // 2, height // 2)
        
        x = np.arange(width)
        y = np.arange(height)
        xv, yv = np.meshgrid(x, y)
        
        # Начальные координаты
        x_map = xv.astype(np.float32)
        y_map = yv.astype(np.float32)
        
        if filter_type == GeometricFilterType.WAVE:
            amplitude = intensity * 5
            frequency = kwargs.get('frequency', 0.1)
            direction = kwargs.get('direction', 'horizontal')
            
            if direction == 'horizontal':
                x_map = xv + amplitude * np.sin(2 * np.pi * yv * frequency)
            else:
                y_map = yv + amplitude * np.sin(2 * np.pi * xv * frequency)
                
        elif filter_type == GeometricFilterType.SWIRL:
            strength = intensity * 5
            radius = kwargs.get('radius', min(width, height) * 0.8)
            
            dx = xv - center[0]
            dy = yv - center[1]
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            
            swirl = strength * (1 - r / radius)
            swirl[r > radius] = 0
            
            x_map = center[0] + r * np.cos(theta + swirl)
            y_map = center[1] + r * np.sin(theta + swirl)
            
        x_map = np.clip(x_map, 0, width - 1)
        y_map = np.clip(y_map, 0, height - 1)
            
        result = cv2.remap(self.original_array, x_map.astype(np.float32), y_map.astype(np.float32), cv2.INTER_LINEAR)
            
        self.noisy_image = result    
        return self.noisy_image
    
    def add_wave_filter(self, intensity=1, direction='horizontal', frequency=0.1):
        return self.add_geometric_filter(
            GeometricFilterType.WAVE,
            intensity=intensity,
            direction=direction,
            frequency=frequency
        )
    
    def add_swirl_filter(self, intensity=0.5, radius=None):
        return self.add_geometric_filter(
            GeometricFilterType.SWIRL,
            intensity=intensity,
            radius=radius
        )

    def save_noisy(self, filename):
        if self.noisy_image is not None:
            cv2.imwrite(filename, self.noisy_image)
            print(f"Зашумленное изображение сохранено: {filename}")
        else:
            print("Нет зашумленного изображения для сохранения")
            
if __name__ == "__main__":
    
    loader = ImageLoad()
    test_img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    cv2.imwrite("test.jpg", test_img)
    
    loader.load("download.jpeg")
    noiser = ImageNoiser(loader)
    
    print("\n")
    noisy_gauss = noiser.add_gaussian_noise(sigma=68)
    
    noiser.save_noisy("Pictures/noisy_gauss.jpg")
    
    print("\n")
    noisy_sp = noiser.add_salt_pepper_noise(salt_prob=0.2, pepper_prob=0.2)

    noiser.save_noisy("Pictures/noisy_salt_pepper.jpg")
    
    print("\n")
    noiser.add_wave_filter()
    
    noiser.save_noisy("Pictures/noisy.geometric.jpg")