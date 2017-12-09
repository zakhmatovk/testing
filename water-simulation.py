import numpy as np
from vispy import gloo
from vispy import app

# Обьект, генерирующий состояния водной глади.
class Surface(object):
    # Конструктор получает размер генерируемого массива.
    # Также конструктор Генерируется параметры нескольких плоских волн.
    def __init__(self, size=(100,100), flat_wave_size=5):
        self._size = size
        # randn - массив размера (n, m, ...), каждая точка нормальное распределение с центром в 0
        # rand - массив размера (n, m, ...), каждая точка равноемерное [0, 1]
        self._wave_vector = 5 * np.random.randn(flat_wave_size, 2)
        # Угловая частота
        self._angular_frequency = np.random.randn(flat_wave_size)
        self._phase = 2 * np.pi * np.random.rand(flat_wave_size)
        self._amplitude = np.random.rand(flat_wave_size) / flat_wave_size
    # Эта функция возвращает xy координаты точек.
    # Точки образуют прямоугольную решетку в квадрате [0,1]x[0,1]
    def position(self):
        """
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
        xy[:,:,0]=np.linspace(-1,1,self._size[0])[:,None]
        xy[:,:,1]=np.linspace(-1,1,self._size[1])[None,:]
        return xy
    # Эта функция возвращает массив высот водной глади в момент времени t.
    # Диапазон изменения высоты от -1 до 1, значение 0 отвечает равновесному положению
    def height(self, t):
        x=np.linspace(-1, 1, self._size[0])[:, None]
        y=np.linspace(-1, 1, self._size[1])[None, :]
        z=np.zeros(self._size, dtype=np.float32)
        for n in range(self._amplitude.shape[0]):
            z[:,:] += self._amplitude[n] * np.cos(
                self._phase[n]
                + x * self._wave_vector[n, 0]
                + y * self._wave_vector[n, 1]
                + t * self._angular_frequency[n]
            )
        return z

vertex = ("""
#version 120

attribute vec2 a_position;

attribute float a_height;

void main (void) {
"""
    # Отображаем диапазон высот [-1,1] в интервал [1,0],
"""
    float z=(1-a_height)*0.5;
    gl_Position = vec4(a_position.xy,z,z);
"""
    # После работы шейдера вершин OpenGL отбросит все точке,
    # с координатами, выпадающими из области -1<=x,y<=1, 0<=z<=1.
    # Затем x и y координаты будут поделены на
    # координату t (последняя в gl_Position),
    # что создает перспективу: предметы ближе кажутся больше.
"""
}
""")

fragment = """
#version 120

void main() {
    gl_FragColor = vec4(0.5, 0.5, 1, 1);
}
"""

class Canvas(app.Canvas):

    def __init__(self):
        app.Canvas.__init__(self, size=(600, 600), title="Water surface simulator 2")
        # запрещаем текст глубины depth_test (все точки будут отрисовываться),
        # запрещает смещивание цветов blend - цвет пикселя на экране равен gl_fragColor.
        gloo.set_state(clear_color=(0,0,0,1), depth_test=False, blend=False)
        self.program = gloo.Program(vertex, fragment)

        self.surface=Surface()
        # xy координаты точек сразу передаем шейдеру, они не будут изменятся со временем
        self.program["a_position"]=self.surface.position()
        # Устанавливаем начальное время симуляции
        self.t=0
        # Закускаем таймер, который будет вызывать метод on_timer для
        # приращения времени симуляции
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.activate_zoom()
        self.show()

    def activate_zoom(self):
        """
            Эта функция вызывается при установке размера окна
            1.Читаем размер окна
            2.Передаем размер окна в OpenGL
        """
        self.width, self.height = self.size
        gloo.set_viewport(0, 0, *self.physical_size)

    def on_draw(self, event):
        # Все пиксели устанавливаются в значение clear_color,
        gloo.clear()
        # Читаем положение высот для текущего времени
        self.program["a_height"]=self.surface.height(self.t)
        # Запускаем шейдеры
        # Метод draw получает один аргумент, указывающий тип отрисовываемых элементов.
        self.program.draw('points')
        # В результате видим анимированную картину "буйков"
        # качающихся на волнах.

    # Метод, вызываемый таймером.
    # Используем для создания анимации.
    def on_timer(self, event):
        # Делаем приращение времени
        self.t+=0.01
        # Сообщаем OpenGL, что нужно обновить изображение,
        # в результате будет вызвано on_draw.
        self.update()

    # Обработчик изменения размера окна пользователем.
    # Нужно передать данные о новом размере окна в OpenGL,
    # в противном случае OpenGL будет рисовать только на части окна.
    def on_resize(self, event):
        self.activate_zoom()

if __name__ == '__main__':
    c = Canvas()
    app.run()
