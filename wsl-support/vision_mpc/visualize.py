import cv2
import numpy as np


class Display:
    """Use the project's existing Pyglet backend, avoiding OpenCV's Qt runtime."""
    def __init__(self):
        import pyglet
        from pyglet.window import key
        self.pyglet, self.stopped = pyglet, False
        self.window = pyglet.window.Window(800, 800, caption='Vision MPC - ESC to stop')

        @self.window.event
        def on_close():
            self.stopped = True
            return pyglet.event.EVENT_HANDLED

        @self.window.event
        def on_key_press(symbol, modifiers):
            if symbol == key.ESCAPE:
                self.stopped = True

    def update(self, bgr):
        self.window.switch_to()
        self.window.dispatch_events()
        if self.stopped:
            return False
        self.window.clear()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        data = self.pyglet.image.ImageData(800, 800, 'RGB', rgb.tobytes(), pitch=-2400)
        data.blit(0, 0)
        self.window.flip()
        return True

    def close(self):
        self.window.close()


def to_pixel(points):
    points = np.asarray(points)
    return np.rint(np.stack(((points[..., 0]+1)*400-.5, (1-points[..., 1])*400-.5), axis=-1)).astype(np.int32)


def overlay(rgb, state, debug, step, method):
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for name, center in state.detection.pixels.items():
        if name == 'obstacles':
            for c in center:
                cv2.circle(frame, tuple(c.astype(int)), 23, (0, 180, 255), 1)
            continue
        x, y = center.astype(int)
        cv2.rectangle(frame, (x-14, y-14), (x+14, y+14), (150, 50, 0), 1)
        cv2.putText(frame, name, (x+15, y), cv2.FONT_HERSHEY_SIMPLEX, .45, (80, 50, 0), 1)
    for key, color in [('route', (210, 100, 0)), ('predicted', (0, 150, 240))]:
        points = np.asarray(debug.get(key, []))
        if len(points) > 1:
            cv2.polylines(frame, [to_pixel(points)], False, color, 2)
    cv2.putText(frame, f'{method}  step={step}  RGB-only control', (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (30, 30, 30), 2)
    if debug.get('recovery', False):
        cv2.putText(frame, 'RECOVERY: temporary detour, extended prediction', (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 100, 210), 1)
    return frame
