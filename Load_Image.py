import cv2
import numpy as np

class ImageLoad:
    
    def __init__(self):
        self.image_array = None
        self.info = {}
        self.color_type = None
        
    def load(self, file):
        try:
            self.image_array = cv2.imread(file, cv2.IMREAD_COLOR)
            
            if self.image_array is None:
                raise ValueError("Не удалось загрузить изображение: {file}")
            
            self.image_info = {
                'filepath': file,
                'shape': self.image_array.shape,
                'dtype': self.image_array.dtype,
                'height': self.image_array.shape[0],
                'width': self.image_array.shape[1],
                'channels': self.image_array.shape[2] if len(self.image_array.shape) > 2 else 1,
                'size_kb': self.image_array.nbytes / 1024
            }
            
            self.color_type = self.detected_color() 
            
            print("Загруженно")
            return self.image_array
        
        except Exception as ex:
            print(ex)
            print(f'Ошибка при загрузки файла: {file}')
            return None
            
    def save_image(self, file, image_array=None):
        if image_array is None:
            image_array = self.image_array
        
        if image_array is not None:
            cv2.imwrite(file, image_array)
            print(f"Изображение сохранено: {file}")
            
    def detected_color(self):

        if self.image_array is None:
            return "no_image"    
    
        if len(self.image_array.shape) == 2:
            channels = 1
        else:
            channels = self.image_array.shape[2]

        if channels == 1:
            unique_values = np.unique(self.image_array)
            
            if len(unique_values) == 2:
                ratio = unique_values.max() / max(unique_values.min(), 1)
                if ratio > 10 or (unique_values.max() == 255 and unique_values.min() == 0):
                    return "monochrome"
                
            if len(unique_values) > 2:
                middle_values = unique_values[(unique_values > 50) & (unique_values < 200)]
                if len(middle_values) > 0:
                    return "grayscale"
                
            return "grayscale"

        elif channels == 3:
            return "color"