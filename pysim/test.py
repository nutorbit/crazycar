from . import CrazycarGymEnv4
from pprint import pprint

from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack, SubprocVecEnv
from stable_baselines.ppo2 import PPO2
from stable_baselines.gail import generate_expert_traj
from stable_baselines import A2C, ACER
		

def main():
	env = CrazycarGymEnv4(renders=False, isDiscrete=True, actionRepeat=2)
	policy = MlpPolicy
	# policy = MlpLnLstmPolicy

	# policy_kwargs = dict(net_arch=[32, 32, dict(vf=[32], pi=[32])])
	env = SubprocVecEnv([lambda: env for _ in range(4)])
	# env = VecNormalize(env, norm_reward=False)
	# env = VecFrameStack(env, 4)

	model = PPO2(policy, env,
				n_steps=512,
				nminibatches=8,
			   	lam=0.95,
			   	gamma=0.9975,
			   	noptepochs=4,
			   	ent_coef=0.01,
			   	cliprange=0.1,
				verbose=1,
				tensorboard_log='./logs/tensorboard/ppo2/'
				# policy_kwargs=policy_kwargs
	)

	iters = (512 * 4) * 300

	model = PPO2.load("./pysim/models/ppo2-4", env, 
					n_steps=512,
					nminibatches=8,
					lam=0.95,
					gamma=0.9975,
					noptepochs=4,
					ent_coef=0.01,
					cliprange=0.1,
					verbose=1,
					tensorboard_log='./logs/tensorboard/ppo2/')

	model.learn(total_timesteps=iters)

	model.save('./pysim/models/ppo2-5')
	env = CrazycarGymEnv4(renders=True, isDiscrete=True, actionRepeat=2)

	while True:
		# state = env.reset([1, 1, 0])
		state = env.reset()
		done = False
		i_step = 0
		ep_reward = 0
		while not done:
			# print(state)
			# action = env.action_space.sample()
			action = model.predict(state)[0]
			print(action)
			# action = 0 # left
			# action = env.action_space.sample()
			next_state, reward, done, info = env.step(action)
			# print(action, reward)
			state = next_state
			i_step += 1
			ep_reward += reward
		print(ep_reward)




		
if __name__ == '__main__':
	main()