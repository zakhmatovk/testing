import numpy as np

from vispy import gloo
from vispy import app


# Шейдеры пишутся на языке GLSL
vertex = ("""
#version 120

attribute vec2 a_position;

void main (void) {
    gl_Position = vec4(a_position.xy, 1, 1);
}
""")

fragment = ("""
#version 120

void main() {
    gl_FragColor = vec4(0.5, 0.5, 1, 1);
}
""")

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, size=(600, 600), title="Water surface simulator 1")
        # запрещаем текст глубины depth_test (все точки будут отрисовываться),
        # запрещает смещивание цветов blend - цвет пикселя на экране равен gl_fragColor.
        gloo.set_state(clear_color=(0,0,0,1), depth_test=False, blend=False)

        self.program = gloo.Program(vertex, fragment)
        self.program["a_position"]=np.array([[0,0]],dtype=np.float32)
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
        # Запускаем шейдеры
        # Метод draw получает один аргумент, указывающий тип отрисовываемых элементов.
        self.program.draw('points')

if __name__ == '__main__':
    c = Canvas()
    app.run()
