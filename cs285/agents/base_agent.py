class BaseAgent(object):
    def __init__(self, **kwargs):
        super(BaseAgent, self).__init__(**kwargs)

    def train(self) -> dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError

    def add_to_replay_buffer(self, paths):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError
    
    def set_peers(self, agents):
        # function for peer agents
        raise NotImplementedError