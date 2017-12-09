import numpy as np


class BedLiner(object):
    def __init__(self, size=(100, 100), max_depth=2.0, min_depth = 0.5):
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
        x = np.linspace(self._min_depth, self._max_depth, self._size[0])[:, None]
        y = np.linspace(self._min_depth, self._max_depth, self._size[1])[None, :]
        d = np.zeros(self._size, dtype=np.float32)
        d[:, :] = x * y
        return d
