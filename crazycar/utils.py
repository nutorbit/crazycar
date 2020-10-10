import numpy as np

from time import time
from datetime import datetime
from absl import logging
from functools import wraps


def get_observation_shape(x):
    """
    Get observation shape for each key (recursive)

    Args:
        x: dictionary of observation

    Returns:
        dictionary of shape
    """

    d = {}
    if isinstance(x, dict):
        for k in x.keys():
            d[k] = get_observation_shape(x[k])
    else:
        return x.shape
    return d


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype='float32') / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def evaluation(env, models, n_episode=10):
    """
    Evaluation the models

    Args:
        env: environment
        models: list of model
    """

    steps = []
    rews = []

    for _ in range(n_episode):
        obs = env.reset()
        done = False
        step_runner = 0
        rew_runner = []

        while not done:

            # get action
            acts = []
            for idx in range(len(models)):
                acts.append(models[idx].predict(obs[idx]))
            acts = np.squeeze(np.array(acts), axis=1)

            # apply action
            next_obs, rew, done, info = env.step(acts)

            # save
            step_runner += 1
            rew_runner.append(np.array(rew))

            # to next state
            done = done[0]
            obs = next_obs

        # store step per episode
        rews.append(np.sum(np.array(rew_runner), axis=0))
        steps.append(step_runner)

    return np.squeeze(np.mean(np.array(rews), axis=0), axis=1), np.mean(steps)


def timing(name, debug=False):
    """
    Time countdown

    Args:
        name: name of the logs
        debug: indicate to print or not print
    """

    def wrap(f):
        @wraps(f)
        def wrap_f(*args, **kwargs):
            # time counting
            ts = time()
            res = f(*args, **kwargs)
            te = time()

            # report
            if debug:
                logging.info(f'|{name}| Started in: {datetime.now()}, Took: {te-ts} sec')
            return res
        return wrap_f
    return wrap
