import numpy as np
from abc import ABC, abstractmethod

class Policy(ABC):
    def __init__(self, actions):
        self.actions = actions

    @abstractmethod
    def get_action(self, state):
        pass