import gym

from .crazycarGymEnv import CrazycarGymEnv
from .crazycarGymEnv2 import CrazycarGymEnv2
from .crazycarGymEnv3 import CrazycarGymEnv3

#from pysim.crazycarGymEnv import CrazycarGymEnv
#from . import train_pybullet_racecar
#from . import enjoy_pybullet_racecar


from gym.envs.registration import registry, make, spec
def register(id,*args,**kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id,*args,**kvargs)

# ------------bullet-------------


register(
    id='CrazyCarEnv-v1',
    entry_point='pysim.crazycarGymEnv:CrazycarGymEnv',
    max_episode_steps=2000,
    reward_threshold=5.0,
)

register(
    id='CrazyCarEnv-v2',
    entry_point='pysim.crazycarGymEnv:CrazycarGymEnv2',
    max_episode_steps=2000,
    reward_threshold=5.0,
)

register(
    id='CrazyCarEnv-v3',
    entry_point='pysim.crazycarGymEnv:CrazycarGymEnv3',
    max_episode_steps=2000,
    reward_threshold=5.0,
)

def getList():
    btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('CrazyCar')>=0]
    return btenvs
