from collections import OrderedDict, defaultdict
import pickle
import os
import sys
import time
import copy
from cs285.agents.ac_agent import ACAgent
from cs285.agents.peer_ac_agent import PeerACAgent
from cs285.agents.peer_sac_agent import PeerSACAgent
from cs285.agents.sac_agent import SACAgent
from cs285.policies.mean_policy import MeanPolicy
from cs285.infrastructure.atari_wrappers import ReturnWrapper

import gym
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu

from cs285.infrastructure.utils import Path
from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

#from cs285.agents.sac_agent import SACAgent

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the code below


class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logdir'])
        self.logmetrics_sac = defaultdict(lambda: False)

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        # register_custom_envs()
        if self.params['agent_class'] in [PeerSACAgent, SACAgent]:
            self.env = gym.make(self.params['env_name'], max_episode_steps=self.params['ep_len'])
        else:
            self.env = gym.make(self.params['env_name'])
        if self.params['video_log_freq'] > 0:
            self.episode_trigger = lambda episode: episode % self.params['video_log_freq'] == 0
        else:
            self.episode_trigger = lambda episode: False
        if 'env_wrappers' in self.params:
            # These operations are currently only for Atari envs
            self.env = wrappers.RecordEpisodeStatistics(self.env, deque_size=1000)
            self.env = ReturnWrapper(self.env)
            self.env = wrappers.RecordVideo(self.env, os.path.join(self.params['logdir'], "gym"), episode_trigger=self.episode_trigger)
            self.env = params['env_wrappers'](self.env)
            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')
        if 'non_atari_colab_env' in self.params and self.params['video_log_freq'] > 0:
            self.env = wrappers.RecordVideo(self.env, os.path.join(self.params['logdir'], "gym"), episode_trigger=self.episode_trigger)
            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')

        self.env.seed(seed)

        # import plotting (locally if 'obstacles' env)
        if not(self.params['env_name']=='obstacles-cs285-v0'):
            import matplotlib
            matplotlib.use('Agg')

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2

        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes

        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1/self.env.model.opt.timestep
        elif 'env_wrappers' in self.params:
            self.fps = 30 # This is not actually used when using the Monitor wrapper
        elif 'video.frames_per_second' in self.env.env.metadata.keys():
            self.fps = self.env.env.metadata['video.frames_per_second']
        else:
            self.fps = 10


        #############
        ## AGENTS
        #############

        agent_class = self.params['agent_class']
        self.agents = []
        for i in range(self.params['num_agents']):
            self.agents.append(agent_class(self.env, self.params['agent_params']))
            self.agents[i].agent_num = i
        if agent_class in [PeerSACAgent, PeerACAgent]:
            for i in range(self.params['num_agents']):
                self.agents[i].set_peers(self.agents[:i] + self.agents[i + 1:])

    def run_training_loop(self, n_iter, collect_policies, eval_policies,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policies: policies to collect training data for each agent
        :param eval_policy: policies to collect data to evaluate each agent
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = defaultdict(int)
        self.start_time = time.time()
        agent_class = self.params['agent_class']

        print_period = 1
        #print_period = 1 if not isinstance(self.agents[0], PeerSACAgent) else 1000

        for itr in range(n_iter):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.logvideo = True
            else:
                self.logvideo = False

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            #for each agent:
            for agent in self.agents:
                agent_num = agent.agent_num
                # collect trajectories, to be used for training
                use_batchsize = self.params['batch_size']
                if itr==0:
                    use_batchsize = self.params['batch_size_initial']
                paths, envsteps_this_batch, train_video_paths = (
                    self.collect_training_trajectories(
                        itr, initial_expertdata, collect_policies[agent_num], use_batchsize)
                )

                self.total_envsteps[agent_num] += envsteps_this_batch

                # add collected data to replay buffer
                agent.add_to_replay_buffer(paths)

                # train agent (using sampled data from replay buffer)
                if itr % print_period == 0:
                    print("\nTraining agent...")
                all_logs = self.train_agent(agent_num)

                # log/save
                if self.logvideo or self.logmetrics:
                    if agent_class in [PeerACAgent, PeerSACAgent] and not self.params['ensemble']:
                        # perform logging
                        print('\nBeginning logging procedure...')
                        self.perform_logging(itr, agent_num, paths, 
                            eval_policies[agent_num], train_video_paths, all_logs)

                    if self.params['save_params']:
                        agent.save('{}/agent_{}_itr_{}.pt'.format(self.params['logdir'], 
                            agent_num, itr))
                
            #logging for ensemble case
            if self.logmetrics and (agent_class in [ACAgent, SACAgent] or self.params['ensemble']):
                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging_ensemble(itr, paths, MeanPolicy(eval_policies), 
                                                train_video_paths, all_logs)

    ####################################
    ####################################
    
    def run_sac_training_loop(self, n_iter, collect_policies, eval_policies):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = defaultdict(int)
        self.start_time = time.time()
        episode_step = 0
        episode_return = 0
        episode_stats = defaultdict(lambda:{'reward': [], 'ep_len': []})
        
        # for SAC we need to put agents in different envs
        train_envs = {}
        for agent in self.agents:
            train_envs[agent.agent_num] = copy.deepcopy(self.env)
            
        done = defaultdict(lambda: False)
        obs = {}
        episode_step = defaultdict(int)
        episode_return = defaultdict(int)
        
        print_period = 1
        
        for itr in range(n_iter):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.logvideo = True
            else:
                self.logvideo = False

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            #for each agent:
            for agent in self.agents:
                agent_num = agent.agent_num
                # collect trajectories, to be used for training
                use_batchsize = self.params['batch_size']
                if itr==0:
                    use_batchsize = self.params['batch_size_initial']
                    print("\nSampling seed steps for training...")
                    paths, envsteps_this_batch = utils.sample_random_trajectories(self.env, use_batchsize, self.params['ep_len'])
                    train_video_paths = None
                    episode_stats[agent_num]['reward'].append(np.mean([np.sum(path['reward']) for path in paths]))
                    episode_stats[agent_num]['ep_len'].append(len(paths[0]['reward']))
                    self.total_envsteps[agent_num] += envsteps_this_batch
                else:
                    if itr == 1 or done[agent_num]:
                        obs[agent_num] = train_envs[agent_num].reset()
                        episode_stats[agent_num]['reward'].append(episode_return[agent_num])
                        episode_stats[agent_num]['ep_len'].append(episode_step[agent_num])
                        episode_step[agent_num] = 0
                        episode_return[agent_num] = 0

                    action = agent.actor.get_action(obs[agent_num])[0]
                    next_obs, rew, done[agent_num], _ = train_envs[agent_num].step(action)

                    episode_return[agent_num] += rew

                    episode_step[agent_num] += 1
                    self.total_envsteps[agent_num] += 1

                    if done[agent_num]:
                        terminal = 1
                    else:
                        terminal = 0
                    paths = [Path([obs[agent_num]], [], [action], [rew], [next_obs], [terminal])]
                    obs[agent_num] = next_obs

                # add collected data to replay buffer
                agent.add_to_replay_buffer(paths)

                # train agent (using sampled data from replay buffer)
                if itr % print_period == 0:
                    print("\nTraining agent...")
                all_logs = self.train_agent(agent_num)

                # log/save
                if self.logvideo or self.logmetrics:
                    # perform logging
                    print('\nBeginning logging procedure...')
                    self.perform_sac_logging(itr, agent_num, episode_stats[agent_num], eval_policies[agent_num], train_video_paths, all_logs)
                    episode_stats[agent_num] = {'reward': [], 'ep_len': []}
                    if self.params['save_params']:
                        agent.save('{}/agent_{}_itr_{}.pt'.format(self.params['logdir'], 
                            agent_num, itr))

    def collect_training_trajectories(self, itr, initial_expertdata, collect_policy, num_transitions_to_sample, save_expert_data_to_disk=False):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param num_transitions_to_sample:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        if itr == 0 and initial_expertdata is not None:
            loaded_paths = np.load(initial_expertdata, allow_pickle=True)
            return loaded_paths, 0, None

        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = utils.sample_trajectories(self.env, 
                                                               collect_policy, 
                                                               num_transitions_to_sample, 
                                                               self.params['ep_len'])

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
        train_video_paths = None
        if self.logvideo:
            print('\nCollecting train rollouts to be used for saving videos...')
            train_video_paths = utils.sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self, agent_num):
        print('\nTraining agent {} using sampled data from replay buffer...'.format(agent_num))
        agent = self.agents[agent_num]
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = agent.sample(self.params['train_batch_size'])
            train_log = agent.train(
                ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs

    ####################################
    ####################################

    def perform_logging(self, itr, agent_num, paths, eval_policy, train_video_paths, all_logs):

        last_log = all_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])

        # save eval rollouts as videos in tensorboard event file
        if self.logvideo and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            #save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

        #######################

        # save eval metrics
        if self.logmetrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Agent{}_Eval_AverageReturn".format(agent_num)] = np.mean(eval_returns)
            logs["Agent{}_Eval_StdReturn".format(agent_num)] = np.std(eval_returns)
            logs["Agent{}_Eval_MaxReturn".format(agent_num)] = np.max(eval_returns)
            logs["Agent{}_Eval_MinReturn".format(agent_num)] = np.min(eval_returns)
            logs["Agent{}_Eval_AverageEpLen".format(agent_num)] = np.mean(eval_ep_lens)

            logs["Agent{}_Train_AverageReturn".format(agent_num)] = np.mean(train_returns)
            logs["Agent{}_Train_StdReturn".format(agent_num)] = np.std(train_returns)
            logs["Agent{}_Train_MaxReturn".format(agent_num)] = np.max(train_returns)
            logs["Agent{}_Train_MinReturn".format(agent_num)] = np.min(train_returns)
            logs["Agent{}_Train_AverageEpLen".format(agent_num)] = np.mean(train_ep_lens)

            logs["Agent{}_Train_EnvstepsSoFar".format(agent_num)] = self.total_envsteps[agent_num]
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Agent{}_Initial_DataCollection_AverageReturn".format(agent_num)] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()

    ####################################
    ####################################
    def perform_sac_logging(self, itr, agent_num, stats, eval_policy, train_video_paths, all_logs):
        last_log = all_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.eval_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])

        # save eval rollouts as videos in tensorboard event file
        if self.logvideo and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            #save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

        #######################

        # save eval metrics
        if self.logmetrics:
            # returns, for logging
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Agent{}_Eval_AverageReturn".format(agent_num)] = np.mean(eval_returns)
            logs["Agent{}_Eval_StdReturn".format(agent_num)] = np.std(eval_returns)
            logs["Agent{}_Eval_MaxReturn".format(agent_num)] = np.max(eval_returns)
            logs["Agent{}_Eval_MinReturn".format(agent_num)] = np.min(eval_returns)
            logs["Agent{}_Eval_AverageEpLen".format(agent_num)] = np.mean(eval_ep_lens)


            logs["Agent{}_Train_AverageReturn".format(agent_num)] = np.mean(stats['reward'])
            logs["Agent{}_Train_StdReturn".format(agent_num)] = np.std(stats['reward'])
            logs["Agent{}_Train_MaxReturn".format(agent_num)] = np.max(stats['reward'])
            logs["Agent{}_Train_MinReturn".format(agent_num)] = np.min(stats['reward'])
            logs["Agent{}_Train_AverageEpLen".format(agent_num)] = np.mean(stats['ep_len'])

            logs["Agent{}_Train_EnvstepsSoFar".format(agent_num)] = self.total_envsteps[agent_num]
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(stats['reward'])
            logs["Agent{}_Initial_DataCollection_AverageReturn".format(agent_num)] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                try:
                    self.logger.log_scalar(value, key, itr)
                except:
                    pdb.set_trace()
            print('Done logging...\n\n')

            self.logger.flush()

    def perform_logging_ensemble(self, itr, paths, eval_policy, train_video_paths, all_logs):

        last_log = all_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])

        # save eval rollouts as videos in tensorboard event file
        if self.logvideo and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            #save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                            video_title='eval_rollouts')

        #######################

        # save eval metrics
        if self.logmetrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps[0]
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()