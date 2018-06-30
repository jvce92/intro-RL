import numpy as np
from abc import ABC, abstractmethod

class Policy(ABC):
    def __init__(self, actions):
        self.actions = actions
        self.action_state = {}
        self.policy = {}
        self.is_trainable = False

    @abstractmethod
    def get_action(self, state):
        pass