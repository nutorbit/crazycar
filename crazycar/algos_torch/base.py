import torch
import torch.nn as nn
import numpy as np

from crazycar.algos_torch.common import Replay


class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def soft_update(self, other_network, tau):

        other_variables = other_network.parameters()
        current_variables = self.parameters()

        with torch.no_grad():
            for (current_var, other_var) in zip(current_variables, other_variables):
                current_var.data.copy_(tau * other_var.data + (1.0 - tau) * current_var.data)

    def hard_update(self, other_network):
        self.soft_update(other_network, tau=1.)


class BaseModel:
    """
    Base class for Actor-Critic algorithm
    """

    def __init__(self, replay_size=int(1e5)):
        super().__init__()
        self.rb = Replay(replay_size)

    def write_metric(self, metric, step):
        raise NotImplementedError

    def actor_loss(self, batch):
        raise NotImplementedError

    def critic_loss(self, batch):
        raise NotImplementedError

    @staticmethod
    def rescale(act, low=0, high=1):
        """
        Rescale action from tanh [-1, 1] to [low, high]
        """

        act = low + (0.5 * (act + 1.0) * (high - low))

        return act

    @staticmethod
    def _update(opt, loss_fn, batch):
        # calculate loss
        loss = loss_fn(batch)

        # update
        opt.zero_grad()
        loss.backward()
        opt.step()

        return loss

    def update_params(self, i, batch_size=256):
        batch = self.rb.sample(batch_size)

        critic_loss = self._update(self.critic_opt, self.critic_loss, batch)
        actor_loss = self._update(self.actor_opt, self.actor_loss, batch)

        # update target network
        if i % self.interval_target == 0:
            self.actor_target.soft_update(self.actor, self.tau)
            self.critic_target.soft_update(self.critic, self.tau)

        return {
            "actor_loss": actor_loss.numpy(),
            "critic_loss": critic_loss.numpy()
        }

    def random_action(self):
        act = np.random.uniform(-1, 1, size=(1, self.actor.act_dim))
        if self.actor.act_dim == 2:
            # apply rescale to speed
            act[0][0] = self.rescale(act[0][0])
        return act

    def predict(self, obs):
        act = self.actor(obs)
        act = act.numpy()
        if len(act[0]) == 2:
            # apply rescale to speed
            act[0][0] = self.rescale(act[0][0])
        return act