import click
import os

from pysim import CrazycarGymEnv4
from pysim.constants import *
from pysim.utils import get_model

from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack, SubprocVecEnv



@click.command()
@click.option('--iters', default=1e5, help='number of update')
@click.argument('idx')
@click.option('--description', help='description of experiment')
@click.argument('name')
def main(iters, idx, description, name):

    # write description
    if description:
        if not os.path.exists(f'./pysim/models/experiment_{idx}/'):
            os.makedirs(f'./pysim/models/experiment_{idx}')
        with open(f'./pysim/models/experiment_{idx}/README.md', 'w') as f:
            f.write(
            f'''

                # Experiment{idx}
                {description}

                Action:
                &emsp;Is Discrete Action: {DISCRETE_ACTION}
                &emsp;Max Speed: {MAX_SPEED}
                &emsp;Min Speed: {MIN_SPEED}
                &emsp;Action Repeat: {ACTION_REP}

            ''')

    # init environment
    env = CrazycarGymEnv4(renders=False, isDiscrete=DISCRETE_ACTION, actionRepeat=ACTION_REP)
    env = SubprocVecEnv([lambda: env for _ in range(N_PARALLEL)])

    # get model
    model = get_model(env=env, name=name, idx_experiment=1)

    # train
    model.learn(total_timesteps=iters)

    # save
    model.save(f'./pysim/models/experiment_{idx}/{name}.pkl')


if __name__ == '__main__':
	main()