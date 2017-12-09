import numpy as np
from vispy import gloo, app, io

from surface import Surface

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
uniform sampler2D u_sky_texture;

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

    vec2 texture_coordinate = 0.25 * reflected.xy / reflected.z + (0.5, 0.5);
    vec3 sky_color = texture2D(u_sky_texture, texture_coordinate).rgb;
    float directed_light = pow(max(0, -dot(u_sun_direction, reflected)), 16);
    vec3 rgb = clamp(u_sun_color * directed_light + sky_color, 0.0, 1.0);

    gl_FragColor = vec4(rgb, 1.0);
}
""")

fragment_point = """
#version 120

void main() {
    gl_FragColor = vec4(1, 0, 0, 1);
}
"""

class Canvas(app.Canvas):

    def __init__(self, sky_img_path="fluffy_clouds.png"):
        app.Canvas.__init__(self, size=(600, 600), title="Water surface simulator 2")
        # запрещаем текст глубины depth_test (все точки будут отрисовываться),
        # запрещает смещивание цветов blend - цвет пикселя на экране равен gl_fragColor.
        gloo.set_state(clear_color=(0,0,0,1), depth_test=True, blend=False)
        self.program = gloo.Program(vertex, fragment_triangle)
        self.program_point = gloo.Program(vertex, fragment_point)

        self.surface = Surface()
        self.sky_img = io.read_png(sky_img_path)
        # xy координаты точек сразу передаем шейдеру, они не будут изменятся со временем
        self.program["a_position"] = self.surface.position()
        self.program['u_sky_texture'] = gloo.Texture2D(self.sky_img, wrapping='repeat', interpolation='linear')
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
        self.set_sun_direction()
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
