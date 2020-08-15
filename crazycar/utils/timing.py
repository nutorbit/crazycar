import os

from time import time
from datetime import datetime
from absl import logging
from functools import wraps


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
            # create the logs
            # if not os.path.exists(f'./tmp/{name}'):
            #     os.makedirs(f'./tmp/{name}')
            # logging.get_absl_handler().use_absl_log_file(name, f'./tmp/{name}')
            # logging.get_absl_handler().setFormatter(None)

            # time counting
            ts = time()
            res = f(*args, **kwargs)
            te = time()

            # report
            # logging.info(f'{name}: Took: {ts-te} sec')
            if debug:
                print(f'|{name}| Started in: {datetime.now()}, Took: {te-ts} sec')
            return res
        return wrap_f
    return wrap
