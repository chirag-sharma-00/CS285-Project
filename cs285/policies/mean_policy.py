import numpy as np


class MeanPolicy(object):

    def __init__(self, policies):
        self.policies = policies

    def get_action(self, obs):
        return sum([pi.get_action(obs) for pi in self.policies]) / len(self.policies)