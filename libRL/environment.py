import numpy as np
from abc import ABC, abstractmethod
from itertools import product
from random import choice

class Environment(ABC):
	def __init__(self, actions, initial_state=None):
		if not isinstance(state, BaseState):
			raise ValueError("{0} is not a valid state".format(initial_state))

		self.actions = actions

		if initial_state is not None
			self.cur_state = initial_state
		else
			self.cur_state = self._random_state() 

	def get_state(self):
		return self.cur_state

	def get_reward(self, action):
		reward = self._get_reward(action)
		self._transition(action)

		return (reward, self.cur_state)

	def set_state(self, state):
		if not isinstance(state, BaseState):
			raise ValueError("{0} is not a valid state".format(state))
		
		self.cur_state = state

	@abstractmethod
	def _get_reward(self, action):
		pass

	@abstractmethod
	def _transition(self, action):
		pass

	@abstractmethod
	def _random_state(self):
		pass