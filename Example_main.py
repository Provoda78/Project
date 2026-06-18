from augmentations_library import (ImageLoader, GaussianNoise, MedianFilter, WaveEffect, 
                                   SaltPepperNoise, GaussianBlur, SwirlEffect, HistogramEqualization, 
                                   ChessboardBlend)

def solve():

    """Пример использования динамической загрузки пакета"""

    loader = ImageLoader()
    img = loader.load("Small.png")
    img2 = loader.load('Small2.png')

    effects = [
        ChessboardBlend(img2)
    ]

    for effect in effects:
        img = effect.apply(img)
    
    loader.save("Pictures/result.png", img)
    
if __name__ == '__main__':
    solve()