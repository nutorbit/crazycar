import numpy as np
import click
import os

from pysim.environment import CrazyCar
from pysim.constants import *
from pysim.utils import get_model

from stable_baselines.results_plotter import load_results, ts2xy

best_mean_reward, n_steps = -np.inf, 0


@click.command()
@click.option('--iters', default=1<<15, help='number of update')
@click.argument('idx')
@click.argument('name')
def main(iters, idx, name):

    if not os.path.exists(f'./models/experiment_{idx}/'):
        os.makedirs(f'./models/experiment_{idx}')


    # get model
    model = get_model(name=name, idx_experiment=idx)

    def callback(_locals, _globals):

        global n_steps, best_mean_reward

        if (n_steps + 1)%1000 == 0:    
            x, y = ts2xy(load_results(LOG_DIR), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y)

                if mean_reward >= best_mean_reward:
                    best_mean_reward = mean_reward
                    _locals['self'].save(f'./models/experiment_{idx}/{name}_best.pkl')
                    print(f'reward at ep {n_steps} is {mean_reward}')

        n_steps += 1
        return True

    # train
    model.learn(total_timesteps=iters, callback=callback)

    # save
    model.save(f'./models/experiment_{idx}/{name}_last.pkl')


if __name__ == '__main__':
	main()