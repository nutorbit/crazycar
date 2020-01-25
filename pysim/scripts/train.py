import numpy as np
import click
import os

from datetime import datetime
from tqdm import tqdm

from pysim.environment import CrazyCar
from pysim.constants import *
from pysim.utils import get_model

from stable_baselines.results_plotter import load_results, ts2xy

best_mean_reward, n_steps = -np.inf, 0
now = datetime.now()
now = now.strftime("%b_%d_%Y_%H:%M:%S")
pbar = tqdm(total=1<<20)

@click.command()
@click.option('--iters', default=1<<20, help='number of update')
@click.argument('idx')
@click.argument('name')
def main(iters, idx, name):

    if not os.path.exists(f'./models/experiment_{idx}/'):
        os.makedirs(f'./models/experiment_{idx}')


    # get model
    model = get_model(name=name, idx_experiment=idx)

    def callback(_locals, _globals):

        global n_steps, best_mean_reward

        x, y = ts2xy(load_results(LOG_DIR), 'timesteps')
        if len(x) > 0:
            if (n_steps + 1)%10000 == 0:
                mean_reward = np.mean(y[-5:])
                if not os.path.exists(f'./models/experiment_{idx}/{name}/{now}'):
                    os.makedirs(f'./models/experiment_{idx}/{name}/{now}')
                _locals['self'].save(f'./models/experiment_{idx}/{name}/{now}/{name}_{n_steps}_{mean_reward:.3f}.pkl')
                pbar.write(f'Average reward at ep {n_steps} is {mean_reward:.3f}')
            
        pbar.set_description(f'Current reward is {y[-1] if len(x) else 0}')
        pbar.update(1)
        n_steps += 1
        return True

    # train
    model.learn(total_timesteps=iters, callback=callback)

    # save
    model.save(f'./models/experiment_{idx}/{name}/{now}/{name}_last.pkl')


if __name__ == '__main__':
	main()