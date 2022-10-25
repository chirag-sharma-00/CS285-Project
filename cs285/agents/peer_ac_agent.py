from collections import OrderedDict

from cs285.critics.peer_bootstrapped_continuous_critic import \
    PeerBootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent


class PeerACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PeerACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.agent_num = 0 #modified by rl_trainer.__init__
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
        )
        self.critic = PeerBootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def get_other_critics(self, agents):
        # given a list of the other PeerACAgent agents in the system, obtains a list
        # of their PeerBootstrappedContinuousCritic critics
        self.other_critics = [agent.critic for agent in agents]
        self.critic.build_advice_net(self.other_critics)

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # advantage = estimate_advantage(...)

        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)

        adv_n = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)

        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            actor_loss = self.actor.update(ob_no, ac_na, adv_n)

        loss = OrderedDict()
        loss['Agent{}_Critic_Loss'.format(self.agent_num)] = critic_loss
        loss['Agent{}_Actor_Loss'.format(self.agent_num)] = actor_loss

        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        # TODO Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
        v_t_values = self.critic.forward_np(ob_no)
        v_tp1_values = self.critic.forward_np(next_ob_no)
        qa_t_values = re_n + self.gamma * v_tp1_values * (1 - terminal_n)
        adv_n = qa_t_values - v_t_values

        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
