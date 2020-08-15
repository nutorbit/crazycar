import click

from pysim.environment import CrazyCar
from backup.pysim.constants import *

from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy, CnnPolicy
from stable_baselines.ppo2 import PPO2
from stable_baselines.ppo1 import PPO1
from stable_baselines.sac import SAC
from stable_baselines.ddpg import DDPG
from stable_baselines.td3 import TD3



@click.command()
@click.argument('name')
@click.argument('path')
def main(name, path):

	# init environment
	env = CrazyCar(renders=True)

	# load model

	if name == 'ppo2':
		model = PPO2.load(path)

	if name == 'ppo1':
		model = PPO1.load(path)
	
	if name == 'sac':
		model = SAC.load(path)
	
	if name == 'ddpg':
		model = DDPG.load(path)

	if name == 'td3_tf':
		model = TD3.load(path)

	# loop
	while True:

		# reset 
		state = env.reset()
		done = False
		ep_reward = 0

		while not done:

			# predict action
			action = model.predict(state)[0]
			# action = (1, -1)
			print('State:', state)
			# print(max(state), min(state))
			print('Action:', action)
			

			# step
			next_state, reward, done, info = env.step(action)

			print('Reward:', reward)
			state = next_state
			ep_reward += reward

		print(f'total reward: {ep_reward}')


if __name__ == '__main__':
	main()