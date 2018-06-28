import numpy as np
from abc import ABC, abstractmethod
from random import choice

class Environment(ABC):
    def __init__(self, actions):
        self.actions = actions
        self.last_state = None
        self.cur_state = self._get_initial_state()

    def get_state(self):
        return self.cur_state

    def get_reward(self, agent, action):
        self.last_state = self.cur_state
        self._transition(agent, action)
        reward = self._reward(agent, action)
        return reward    

    @abstractmethod
    def _reward(self, agent, action):
        pass

    @abstractmethod
    def _transition(self, agent, action):
        pass

    @abstractmethod
    def _get_initial_state(self):
        pass

