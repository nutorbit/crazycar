import pyglet
import numpy as np
import time


from gym.envs.classic_control.rendering import SimpleImageViewer
from . import CrazycarGymEnv4
from PIL import Image


class EnvInteractor(SimpleImageViewer):

    def __init__(self):
        super().__init__(maxwidth=800)
        self.keys = pyglet.window.key.KeyStateHandler()
        self._finished = False
        self.imshow(np.zeros([168, 168, 3], dtype=np.uint8))

    def imshow(self, image):
        was_none = self.window is None
        image = Image.fromarray(image)
        image = image.convert('RGB')
        image = image.resize((800, 800))
        image = np.array(image)
        super().imshow(image)
        if was_none:
            # self.window.event(self.on_key_press)
            self.window.push_handlers(self.keys)

    def get_action(self):
        event = None
        if self.keys[pyglet.window.key.UP]:
            # event = 'forward'
            event = 2
        if self.keys[pyglet.window.key.LEFT]:
            # event = 'left'
            event = 0
        if self.keys[pyglet.window.key.RIGHT]:
            # event = 'right'
            event = 5
        if self.keys[pyglet.window.key.ESCAPE]:
            # event = 'escape'
            self._finished = True
        return event

    def run_loop(self, env):

        obs = env.reset()

        last_img = obs[0]
        
        while not self._finished:
            action = self.get_action()
            
            if action is not None:
                obs, rew, done, info = env.step(action)
                last_img = obs[0]

                self.imshow(obs[0])

                print(obs[1:], rew)
            else:
                self.imshow(last_img)

            # time.sleep(1e-3)

if __name__ == '__main__':

    env = CrazycarGymEnv4(renders=False, isDiscrete=True, actionRepeat=2, selfcontrol=True)

    viewer = EnvInteractor()
    viewer.run_loop(env)

