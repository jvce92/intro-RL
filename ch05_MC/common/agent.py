import numpy as np
from random import choice
from .policy import Policy
from .environment import Environment
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, policy, terminal_states, verbose=False):
        assert(isinstance(policy, Policy))

        self.policy = policy
        self.terminal_states = terminal_states
        self.value_state = {}
        self.returns = {}

    def load_policy(self, _policy):
        self.policy = _policy

    @abstractmethod
    def _internal_representation(self, state):
        pass
