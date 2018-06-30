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

    def evaluate_episode(self, episode):
        G = 0
        prev_states = []
        for i, step in enumerate(episode):
            s, a, r = step
            s = self._internal_representation(s)
            G += r
            
            if s not in prev_states:
                if s not in self.returns.keys():
                    self.returns[s] = []

                self.returns[s].append(G)
                self.value_state[s] = np.mean(self.returns[s])
                prev_states.append(s)

    @abstractmethod
    def _internal_representation(self, state):
        pass
