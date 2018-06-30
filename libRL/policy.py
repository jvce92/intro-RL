from abc import ABC, abstractmethod
from .state import BaseState
from .action import BaseAction

class BasePolicy:
	def __init__(self, name):
		if not isinstance(name, str):
			raise ValueError("{0}} is not a string".format(name))

		self.name = name
		self.pi = {}

	def update(self, state, action):
		if not isinstance(state, BaseState):
						raise ValueError("{0} is not a valid state".format(state))

		if not isinstance(action, BaseAction):
			raise ValueError("{0} is not a valid action".format(action))

		self.pi[state] = action

	def get_action(self, state):
		if not isinstance(state, BaseState):
						raise ValueError("{0} is not a valid state".format(state))

		return self.pi[state]