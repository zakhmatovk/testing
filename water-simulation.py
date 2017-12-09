import numpy as np
from vispy import gloo
from vispy import app

# Обьект, генерирующий состояния водной глади.
class Surface(object):
    def __init__(self, size=(100,100), flat_wave_size=5):
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
        self._amplitude = np.random.rand(flat_wave_size) / flat_wave_size

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

vertex = ("""
#version 120

attribute vec2 a_position;
attribute float a_height;
attribute vec2 a_normal;

varying vec3 v_normal;
varying vec3 v_position;

void main (void) {
    v_normal = normalize(vec3(a_normal, -1));
    v_position = vec3(a_position.xy, a_height);

    float z = (1 - a_height) * 0.5;

    gl_Position = vec4(a_position.xy / 2, a_height * z, z);
}
""")

fragment_triangle = ("""
#version 120

uniform vec3 u_sun_direction;
uniform vec3 u_sun_color;
uniform vec3 u_ambient_color;

varying vec3 v_normal;
varying vec3 v_position;

void main (void) {
"""
    # Вычисляем яркость отраженного света, предполагая, что
    # камера находится в точке eye.
"""
    vec3 eye = vec3(0, 0, 1);
    vec3 to_eye = normalize(v_position - eye);
"""
    # Сначала считаем направляющий вектор отраженного от поверхности
    # испущенного из камеры луча.
"""
    vec3 reflected = normalize(to_eye - 2 * v_normal * dot(v_normal, to_eye) / dot(v_normal, v_normal));
"""
    # Яркость блико от Солнца.
"""
    float directed_light = pow(max(0, -dot(u_sun_direction, reflected)), 16);
    vec3 rgb = clamp(u_sun_color * directed_light + u_ambient_color, 0.0, 1.0);
    gl_FragColor = vec4(rgb, 1);
}
""")

fragment_point = """
#version 120

void main() {
    gl_FragColor = vec4(1, 0, 0, 1);
}
"""

class Canvas(app.Canvas):

    def __init__(self):
        app.Canvas.__init__(self, size=(600, 600), title="Water surface simulator 2")
        # запрещаем текст глубины depth_test (все точки будут отрисовываться),
        # запрещает смещивание цветов blend - цвет пикселя на экране равен gl_fragColor.
        gloo.set_state(clear_color=(0,0,0,1), depth_test=True, blend=False)
        self.program = gloo.Program(vertex, fragment_triangle)
        self.program_point = gloo.Program(vertex, fragment_point)

        self.surface = Surface()
        # xy координаты точек сразу передаем шейдеру, они не будут изменятся со временем
        self.program["a_position"] = self.surface.position()
        self.program_point["a_position"] = self.surface.position()
        self.program["u_sun_color"] = np.array([0.8, 0.8, 0], dtype=np.float32)
        self.program["u_ambient_color"] = np.array([0.1, 0.1, 0.5], dtype=np.float32)
         # Сохраним треугольники, которые нужно соединить отрезками, в графическую память.
        self.triangles = gloo.IndexBuffer(self.surface.triangulation())
        #self.segments = gloo.IndexBuffer(self.surface.wireframe())
        # Устанавливаем начальное время симуляции
        self.t = 0
        self.set_sun_direction()
        self.are_points_visible = False
        # Закускаем таймер, который будет вызывать метод on_timer для
        # приращения времени симуляции
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.activate_zoom()
        self.show()

    def set_sun_direction(self):
        phi = np.pi * (1 + self.t * 0.1);
        sun = np.array([np.sin(phi), np.cos(phi), -0.5], dtype=np.float32)
        sun /= np.linalg.norm(sun)
        self.program["u_sun_direction"] = sun

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
        height = self.surface.height(self.t)
        self.program["a_height"] = height
        self.program["a_normal"] = self.surface.normal(self.t)
        #gloo.set_state(depth_test=True)
        self.program.draw('triangles', self.triangles)
        if self.are_points_visible:
            self.program_point["a_height"] = height
            gloo.set_state(depth_test=False)
            self.program_point.draw('points')

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

    def on_key_press(self, event):
        if event.key == 'Escape':
            self.close()
        elif event.key == ' ':
            self.are_points_visible = not self.are_points_visible

if __name__ == '__main__':
    c = Canvas()
    app.run()
