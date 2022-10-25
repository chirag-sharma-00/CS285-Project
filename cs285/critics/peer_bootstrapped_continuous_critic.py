from .base_critic import BaseCritic
from torch import nn
from torch import optim
import numpy as np
import torch

from cs285.infrastructure import pytorch_util as ptu


class PeerBootstrappedContinuousCritic(nn.Module, BaseCritic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        other_critics is a list of PeerBootstrappedContinuousCritic objects corresponding
        to the critics of the other agents in the peer system

        Note: batch self.size /n/ is defined at runtime.
        is None
    """
    def __init__(self, hparams):
        super().__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']

        # critic parameters
        self.advice_dim = hparams['advice_dim']
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']
        self.eps = hparams['epsilon']
        self.critic_network = ptu.build_mlp(
            #TODO: right now, during training, pass in the ob vector and another
            #advice_dim length vector: with prob eps, this is zero, otherwise it's a 
            #compressed representation of the outputs of the other critics in the
            #system (the advice) -- during eval time, always pass in 0 for advice
            self.ob_dim + self.advice_dim, 
            1,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.critic_network.to(ptu.device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.critic_network.parameters(),
            self.learning_rate,
        )

    def build_advice_net(self, other_critics):
        #TODO: may need to make these choices hyperparameters
        self.other_critics = other_critics
        self.advice_network = ptu.build_mlp(
            len(other_critics),
            self.advice_dim,
            n_layers=1,
            size=4
        )

    def forward(self, obs, train_mode=False):
        use_advice = np.random.choice([0, 1], p=[self.eps, 1 - self.eps])
        if train_mode and use_advice:
            outputs = [critic.forward(obs).unsqueeze(1) for critic in self.other_critics]
            outputs = torch.cat(outputs, dim=1)
            advice = self.advice_network(outputs)
        else:
            advice = torch.zeros(obs.shape[0], self.advice_dim)
        assert len(obs.shape) == len(advice.shape) and obs.shape[0] == advice.shape[0]
        obs = torch.cat([obs, advice], dim=1)
        return self.critic_network(obs).squeeze(1)

    def forward_np(self, obs):
        obs = ptu.from_numpy(obs)
        predictions = self(obs, train_mode=False)
        return ptu.to_numpy(predictions)

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                training loss
        """
        # Implement the pseudocode below: do the following (
        # self.num_grad_steps_per_target_update * self.num_target_updates)
        # times:
        # every self.num_grad_steps_per_target_update steps (which includes the
        # first step), recompute the target values by
        #     a) calculating V(s') by querying the critic with next_ob_no
        #     b) and computing the target values as r(s, a) + gamma * V(s')
        # every time, update this critic using the observations and targets
        #
        # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it
        #       to 0) when a terminal state is reached
        # HINT: make sure to squeeze the output of the critic_network to ensure
        #       that its dimensions match the reward
        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        for i in range(self.num_grad_steps_per_target_update * self.num_target_updates):
            if i % self.num_grad_steps_per_target_update == 0:
                v_tp1_values = self(next_ob_no).squeeze()
                assert reward_n.shape == v_tp1_values.shape
                target = reward_n + self.gamma * v_tp1_values * (1 - terminal_n)
                target = target.detach()
            v_t_values = self(ob_no, train_mode=True).squeeze()

            assert v_t_values.shape == target.shape
            loss = self.loss(v_t_values, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()
