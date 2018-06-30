import numpy as np
from abc import ABC, abstractmethod
from random import choice
from .state import BaseState
from .action import BaseAction
from .policy import BasePolicy
from .action_state_value import ValueState, ActionValueState 

class Agent(ABC):
	def __init__(self, policy, actions, states=None):
		if not isinstance(policy, BasePolicy):
			raise ValueError("{0} is not a Policy".format(policy))

		self.policy = policy
		self.action_state = ActionValueState(states, actions)
		self.value_state = ValueState(states)
		self.actions = actions

	def get_action(self, state):
		if not isinstance(state, BaseState):
			raise ValueError("{0} is not a valid state".format(state))

		return self.policy(state)

	def update_Q(self, state, action, val):
		self.action_state.update(state, action, val)

	def update_v(self, state, val):
		self.value_state.update(state, val)

	def update_policy(self, state, action):
		policy.update(state) = action

	def _get_best_action(self, state):
		if not isinstance(state, BaseState):
			raise ValueError("{0} is not a valid state".format(state))
		
		best_value = float("-inf")
		for action in self.actions:
			cur_value = self.action_state(state, action)
			if cur_value > best_value:
				best_value = cur_value
				best_action = action

		return best_action

	def _get_random_action(self):
		return np.random.choic(self.actions)

	@abstractmethod
	def get_action(self, state):
		pass

		