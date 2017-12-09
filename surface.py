import numpy as np

class Surface(object):
    def __init__(self, size=(100,100), flat_wave_size=10, max_height=0.2):
        """
            Конструктор получает размер генерируемого массива.
            Также конструктор Генерируется параметры нескольких плоских волн.
        """
        self._size = size
        # randn - массив размера (n, m, ...), каждая точка нормальное распределение с центром в 0
        # rand - массив размера (n, m, ...), каждая точка равноемерное [0, 1]
        self._wave_vector = 5 * np.random.randn(flat_wave_size, 2)
        # Угловая частота
        self._angular_frequency = np.random.randn(flat_wave_size)
        self._phase = 2 * np.pi * np.random.rand(flat_wave_size)
        self._amplitude=max_height * (1 + np.random.rand(flat_wave_size)) / 2 / flat_wave_size

    def position(self):
        """
            Эта функция возвращает xy координаты точек.
            Точки образуют прямоугольную решетку в квадрате [0,1]x[0,1]
        array([[[-1., -1.],
            [-1.,  0.],
            [-1.,  1.]],

            [[ 0., -1.],
             [ 0.,  0.],
             [ 0.,  1.]],

            [[ 1., -1.],
             [ 1.,  0.],
             [ 1.,  1.]]])
        """
        xy=np.empty(self._size + (2,), dtype=np.float32)
        xy[:, :, 0]=np.linspace(-1, 1, self._size[0])[:, None]
        xy[:, :, 1]=np.linspace(-1, 1, self._size[1])[None, :]
        return xy

    def height(self, t):
        """
            Эта функция возвращает массив высот водной глади в момент времени t.
            Диапазон изменения высоты от -1 до 1, значение 0 отвечает равновесному положению
        """
        x = np.linspace(-1, 1, self._size[0])[:, None]
        y = np.linspace(-1, 1, self._size[1])[None, :]
        z = np.zeros(self._size, dtype=np.float32)
        for n in range(self._amplitude.shape[0]):
            z[:,:] += self._amplitude[n] * np.cos(
                self._phase[n]
                + x * self._wave_vector[n, 0]
                + y * self._wave_vector[n, 1]
                + t * self._angular_frequency[n]
            )
        return z

    def normal(self, t):
        x = np.linspace(-1, 1, self._size[0])[:, None]
        y = np.linspace(-1, 1, self._size[1])[None, :]
        grad = np.zeros(self._size + (2,), dtype=np.float32)
        for n in range(self._amplitude.shape[0]):
            dcos = -self._amplitude[n] * np.sin(
                self._phase[n]
                + x * self._wave_vector[n, 0]
                + y * self._wave_vector[n, 1]
                + t * self._angular_frequency[n])
            grad[:, :, 0] += self._wave_vector[n, 0] * dcos
            grad[:, :, 1] += self._wave_vector[n, 1] * dcos
        return grad

     # Возвращает массив индесов вершин треугольников.
    def triangulation(self):
        # Решетка состоит из прямоугольников с вершинами
        # A (левая нижняя), B(правая нижняя), С(правая верхняя), D(левая верхняя).
        # Посчитаем индексы всех точек A,B,C,D для каждого из прямоугольников.
        a = np.indices((self._size[0] - 1, self._size[1] - 1))
        b = a + np.array([1, 0])[:, None, None]
        c = a + np.array([1, 1])[:, None, None]
        d = a + np.array([0, 1])[:, None, None]
        # Преобразуем массив индексов в список (одномерный массив)
        a_r = a.reshape((2, -1))
        b_r = b.reshape((2, -1))
        c_r = c.reshape((2, -1))
        d_r = d.reshape((2, -1))
        # Заменяем многомерные индексы линейными индексами
        a_l = np.ravel_multi_index(a_r, self._size)
        b_l = np.ravel_multi_index(b_r, self._size)
        c_l = np.ravel_multi_index(c_r, self._size)
        d_l = np.ravel_multi_index(d_r, self._size)
        # Собираем массив индексов вершин треугольников ABC, ACD
        abc = np.concatenate((a_l[..., None], b_l[..., None],c_l [...,None]), axis=-1)
        acd = np.concatenate((a_l[..., None], c_l[..., None],d_l [...,None]), axis=-1)
        # Обьединяем треугольники ABC и ACD для всех прямоугольников
        return np.concatenate((abc, acd), axis=0).astype(np.uint32)