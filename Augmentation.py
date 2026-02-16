import Load_Image
#import cv2
import numpy as np

class ImageNoiser:
    def __init__(self):
        self.original_image.image_array = Load_Image() 
        self.noisy_image = None
        self.noise_params = {}
        
    def add_gaussian_noise(self, mean=0, sigma=25):
        
        """
        Добавляет гауссов шум
        mean: среднее значение шума
        sigma:  среднее квадратичное отклонение
        """
        
        if self.original_image.image_array is None:
            raise ValueError("Сначала загрузите изображение")
        
        noise = np.random.normal(mean, sigma, self.original_image.image_array.shape)
        
        noisy = self.original_image.image_array.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        self.noisy_image = noisy
        self.noise_params['gaussian'] = {'mean': mean, 'sigma': sigma}
        
        print(f"Добавлен гауссов шум (Среднее отклонение = {sigma})")
        
        return noisy