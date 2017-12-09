import numpy as np
from surface import CircularWaves


class BedLiner(object):
    def __init__(self, size=(100, 100), max_depth=1.0, min_depth = 0.5):
        """
            Конструктор получает размер генерируемого массива.
            Также конструктор Генерируется параметры нескольких плоских волн.
        """
        self._size = size
        self._max_depth = max_depth
        self._min_depth = min_depth

    def depth(self):
        """
            Эта функция возвращает массив высот водной глади в момент времени t.
            Диапазон изменения высоты от -1 до 1, значение 0 отвечает равновесному положению
        """
        x = np.linspace(self._min_depth, self._max_depth,
                        self._size[0])[:, None]
        y = np.linspace(self._min_depth, self._max_depth,
                        self._size[1])[None, :]
        d = np.zeros(self._size, dtype=np.float32)
        d[:, :] = x * y
        return d


class BedLog(object):
    def __init__(self, size=(100, 100), max_depth=3.0, min_depth=0.5):
        """
            Конструктор получает размер генерируемого массива.
            Также конструктор Генерируется параметры нескольких плоских волн.
        """
        self._size = size
        self._max_depth = max_depth
        self._min_depth = min_depth

    def depth(self):
        """
            Эта функция возвращает массив высот водной глади в момент времени t.
            Диапазон изменения высоты от -1 до 1, значение 0 отвечает равновесному положению
        """
        part = self._min_depth + np.logspace(0, 1, self._size[0] / 2) / 10.0 * (self._max_depth - self._min_depth)
        center = np.array([self._max_depth] if self._size[0] % 2 == 1 else [])
        full_array = np.concatenate((part, center, part))
        x = full_array[:, None]
        y = full_array[None, :]
        d = np.zeros(self._size, dtype=np.float32)
        d[:, :] = x * y
        return d

class BedCircular():
    def __init__(self, size=(100, 100), max_depth=0.2, min_depth=0.1):
        """
            Конструктор получает размер генерируемого массива.
            Также конструктор Генерируется параметры нескольких плоских волн.
        """
        self._size = size
        self._max_depth = max_depth
        self._min_depth = min_depth

    def depth(self):
        """
            Эта функция возвращает массив высот водной глади в момент времени t.
            Диапазон изменения высоты от -1 до 1, значение 0 отвечает равновесному положению
        """
        wave = CircularWaves(self._size, self._max_depth, wave_length=0.6)
        d, grad = wave.height_and_normal()
        return d + self._min_depth
