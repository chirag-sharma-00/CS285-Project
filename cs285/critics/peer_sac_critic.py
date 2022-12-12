import itertools
from .base_critic import BaseCritic
from torch import nn
from torch import optim
import numpy as np
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import sac_utils
import torch

class PeerSACCritic(nn.Module, BaseCritic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """
    def __init__(self, hparams):
        super(PeerSACCritic, self).__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']

        # critic parameters
        self.eps = hparams['epsilon']
        self.advice_dim = hparams['advice_dim']
        self.adv_n_layers = hparams['advice_net_n_layers']
        self.adv_size = hparams['advice_net_size']
        self.gamma = hparams['gamma']
        #TODO: right now, during training, pass in the ob vector and another
        #advice_dim length vector into Q networks: 
        #with prob eps, this second vector is zero, otherwise it's a compressed 
        #representation of the outputs of the other critics in the
        #system (the advice) -- during eval time, always pass in 0 for advice
        self.Q1 = ptu.build_mlp(
            self.ob_dim + self.ac_dim + self.advice_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
            activation='relu'
        )
        self.Q2 = ptu.build_mlp(
            self.ob_dim + self.ac_dim + self.advice_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
            activation='relu'
        )
        self.Q1.to(ptu.device)
        self.Q2.to(ptu.device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.parameters(),
            self.learning_rate,
        )
        # self.apply(sac_utils.weight_init)

    def build_advice_net(self, other_critics):
        self.other_critics = other_critics
        self.advice_network1 = ptu.build_mlp(
            len(other_critics),
            self.advice_dim,
            n_layers=self.adv_n_layers,
            size=self.adv_size
        ).to(ptu.device)
        self.advice_network2 = ptu.build_mlp(
            len(other_critics),
            self.advice_dim,
            n_layers=self.adv_n_layers,
            size=self.adv_size
        ).to(ptu.device)
        self.optimizer = optim.Adam(
            self.parameters(),
            self.learning_rate,
        )
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor, train_mode=False):
        use_advice = np.random.choice([0, 1], p=[self.eps, 1 - self.eps])
        if train_mode and use_advice:
            outputs_zipped = [critic.forward(obs, action) for critic in self.other_critics]
            outputs1 = [output_zipped[0] for output_zipped in outputs_zipped]
            outputs2 = [output_zipped[1] for output_zipped in outputs_zipped]
            outputs1 = torch.cat(outputs1, dim=1).detach()
            outputs2 = torch.cat(outputs2, dim=1).detach()
            advice1 = self.advice_network1(outputs1)
            advice2 = self.advice_network2(outputs2)
        else:
            advice1, advice2 = (torch.zeros(obs.shape[0], self.advice_dim).to(ptu.device), 
                                torch.zeros(obs.shape[0], self.advice_dim).to(ptu.device))
        assert len(obs.shape) == len(advice1.shape) and obs.shape[0] == advice1.shape[0]
        assert len(obs.shape) == len(advice2.shape) and obs.shape[0] == advice2.shape[0]
        obs_action = torch.cat([obs, action], dim=-1)
        obs_action_advice1 = torch.cat([obs_action, advice1], dim=-1)
        obs_action_advice2 = torch.cat([obs_action, advice1], dim=-1)
        q1 = self.Q1(obs_action_advice1)
        q2 = self.Q2(obs_action_advice2)
        return [q1, q2]

    def forward_np(self, obs: np.ndarray, action: np.ndarray):
        obs = ptu.from_numpy(obs)
        predictions = self(obs)
        return ptu.to_numpy(predictions)


        
