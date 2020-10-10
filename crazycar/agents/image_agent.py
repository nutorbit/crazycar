import numpy as np
import math

from crazycar.agents import BaseAgent


class ImageAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_observation(self):
        observation = {
            "image": self.get_camera()
        }
        return observation

    def get_reward(self):
        # diff_angle = self.get_diff_angle()

        # reward = self.speed * np.cos(diff_angle) - np.abs(self.speed * np.sin(diff_angle))
        reward = 0
        # print(diff_angle, reward)
        if self.is_collision():
            reward = -50
            self.nCollision += 1

        # if math.pi - math.pi / 4 <= diff_angle * math.pi <= math.pi + math.pi / 4:
        #     reward = -50
        #     self.nCollision += 1

        return reward
