from collections import Iterable
from .state import BaseState
from .action import BaseAction

class ValueState:
	def __init__(self, states=None):
		self.v = {}

		if states is not None:
			if not isinstance(states, Iterable):
				raise ValueError("{0} is not iterable".format(states))

			for state in states:
				if not isinstance(state, BaseState):
					raise ValueError("{0} is not a valid state".format(state))

				self.v[state] = 0

	def update(self, state, value):
		if not isinstance(state, BaseState):
			raise ValueError("{0} is not a valid state".format(state))

		if not isinstance(value, float):
			raise ValueError("{0} is not a numeric type".format(value))

		v[state] = value

	def __call__(self, state):
		if state in self.v.keys():
			return self.v[state]

		return 0

class ActionValueState:
	def __init__(self, states=None, actions=None):
		self.Q = {}

		if states is not None and actions is not None:
			if not isinstance(states, Iterable):
				raise ValueError("{0} is not iterable".format(states))

			if not isinstance(actions, Iterable):
				raise ValueError("{0} is not iterable".format(actions))

			for state in states:
				for action in actions:
					if not isinstance(state, BaseState):
						raise ValueError("{0} is not a valid state".format(state))

					if not isinstance(action, BaseAction):
						raise ValueError("{0} is not a valid action".format(action))

					self.Q[(state, action)] = 0

	def update(self, state, action, value):
		if not isinstance(state, BaseState):
						raise ValueError("{0} is not a valid state".format(state))

		if not isinstance(action, BaseAction):
			raise ValueError("{0} is not a valid action".format(action))

		if not isinstance(value, float):
			raise ValueError("{0} is not a numeric type".format(value))

		self.Q[(state, action)] = value

	def __call__(self, state, action):
		if (state, action) in  self.Q.keys()
			return self.Q[(state, action)]

		return 0