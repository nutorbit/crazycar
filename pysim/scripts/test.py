import click

from pysim import CrazycarGymEnv4
from pysim.constants import *

from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack, SubprocVecEnv
from stable_baselines.ppo2 import PPO2


@click.command()
@click.argument('path')
def main(path):

	# init environment
	env = CrazycarGymEnv4(renders=True, isDiscrete=DISCRETE_ACTION, actionRepeat=ACTION_REP)

	# load model
	model = PPO2.load(path)

	# loop
	while True:

		# reset 
		state = env.reset()
		done = False
		ep_reward = 0

		while not done:

			# predict action
			action = model.predict(state)[0]
			print(action)

			# step
			next_state, reward, done, info = env.step(action)
			state = next_state
			ep_reward += reward

		print(f'total reward: {ep_reward}')


if __name__ == '__main__':
	main()