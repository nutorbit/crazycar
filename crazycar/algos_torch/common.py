import torch
import random
import numpy as np
import torch.nn as nn

from collections import deque


def initial(seed=100):
    """
    Initial seed & gpu

    Args:
        seed: seed number
    """

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        if j < len(sizes)-2:
            # layers += [nn.Linear(sizes[j], sizes[j+1]), nn.BatchNorm1d(sizes[j+1]), activation()]
            layers += [nn.Linear(sizes[j], sizes[j + 1]), activation()]
        else:  # output layer
            layers += [nn.Linear(sizes[j], sizes[j+1]), output_activation()]
    return nn.Sequential(*layers)


class Replay:
    """
    Experience replay for RL
    """

    def __init__(self, maxlen=int(1e6)):
        self.data = deque(maxlen=maxlen)

    def store(self, dict_data):
        """
        Store data

        Args:
            dict_data:
            {
                "obs": xxx
                "act": xxx
                "next_obs": xxx
                "rew": xxx
                "done": xxx
            }
        """

        self.data.append(dict_data)

    def process(self, data):
        """
        Process data

        Args:
            data: list of dictionary

        Returns:
            dictionary of batch for each
        """

        batch = {
            "obs": {
                "image": torch.FloatTensor(
                    np.concatenate([el["obs"]["image"] for el in data])
                ).to('cuda') if "image" in data[0]["obs"] else None,
                "sensor": torch.FloatTensor(
                    np.concatenate([el["obs"]["sensor"] for el in data])
                ).to('cuda') if "sensor" in data[0]["obs"] else None
            },
            "act": torch.FloatTensor(
                np.concatenate([np.expand_dims(el["act"], axis=0) for el in data])
            ).to('cuda'),
            "next_obs": {
                "image": torch.FloatTensor(
                    np.concatenate([el["next_obs"]["image"] for el in data])
                ).to('cuda') if "image" in data[0]["obs"] else None,
                "sensor": torch.FloatTensor(
                    np.concatenate([el["next_obs"]["sensor"] for el in data])
                ).to('cuda') if "sensor" in data[0]["obs"] else None
            },
            "rew": torch.FloatTensor(
                np.expand_dims(np.concatenate([el["rew"] for el in data]), axis=-1)
            ).to('cuda'),
            "done": torch.FloatTensor(
                np.expand_dims(np.concatenate([el["done"] for el in data]), axis=-1)
            ).to('cuda')
        }

        # clear unwanted key
        if batch["obs"]["image"] is None:
            batch["obs"].pop("image")
        if batch["obs"]["sensor"] is None:
            batch["obs"].pop("sensor")

        if batch["next_obs"]["image"] is None:
            batch["next_obs"].pop("image")
        if batch["next_obs"]["sensor"] is None:
            batch["next_obs"].pop("sensor")

        return batch

    def sample(self, size=256):
        idx = list(range(len(self.data)))
        np.random.shuffle(idx)
        idx = idx[:size]
        tmp = np.array(self.data)
        return self.process(tmp[idx])

