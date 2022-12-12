import random
from typing import List
from .base_critic import BaseCritic
from torch import nn
from torch import optim
import numpy as np
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import sac_utils
import torch

class PeerSACCriticV2(nn.Module, BaseCritic):
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
        super(PeerSACCriticV2, self).__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']

        # critic parameters
        self.eps = hparams['epsilon']
        self.advice_dim = hparams['advice_dim']
        self.gamma = hparams['gamma']
        #DONE: right now, during training, pass in the ob vector and another
        #advice_dim length vector into Q networks: 
        #with prob eps, this second vector is zero, otherwise it's a compressed 
        #representation of the outputs of the other critics in the
        #system (the advice) -- during eval time, always pass in 0 for advice
        self.Q1 = ptu.build_mlp(
            self.ob_dim + self.ac_dim + self.advice_dim,
            1 + self.advice_dim,
            n_layers=self.n_layers,
            size=self.size,
            activation='relu'
        )
        self.Q2 = ptu.build_mlp(
            self.ob_dim + self.ac_dim + self.advice_dim,
            1 + self.advice_dim,
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
        self.other_critics = []
        # self.apply(sac_utils.weight_init)
    
    def build_advice_net(self, other_critics):
        # This is a misnomer. No advice network is built here. 
        # This name is kept for consistancy with the first version of PeerSACCritic
        self.other_critics = other_critics
        
        for i, critic in enumerate(other_critics):
            for j, param in enumerate(critic.parameters()):
                self.register_parameter(f"peer_{i}_param_{j}",param)
        
        
        self.optimizer = optim.Adam(
            self.parameters(),
            self.learning_rate
        )
    
    def forward_with_advice(self, obs: torch.Tensor, action: torch.Tensor, train_mode=False):
        use_advice = np.random.choice([0, 1], p=[self.eps, 1 - self.eps])
        if train_mode and len(self.other_critics) > 0 and use_advice:
            # TODO: implement the average advice version
            random_peer = random.choice(self.other_critics)
            _, _, advice = random_peer.forward_with_advice(obs, action, train_mode=False)
        else:
            advice = torch.zeros(obs.shape[0], self.advice_dim).to(ptu.device)
            
        assert len(obs.shape) == len(advice.shape) and obs.shape[0] == advice.shape[0]
        
        obs_action = torch.cat([obs, action], dim=-1)
        obs_action_advice = torch.cat([obs_action, advice], dim=-1)
        q1_advice1 = self.Q1(obs_action_advice)
        q2_advice2 = self.Q2(obs_action_advice)
        q1, advice1 = q1_advice1[:, 0], q1_advice1[:, 1:]
        q2, advice2 = q2_advice2[:, 0], q2_advice2[:, 1:]
        advice_out = (advice1 + advice2) / 2
        return [q1, q2, advice_out]
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor, train_mode=False):
        return self.forward_with_advice(obs, action, train_mode)[:-1]