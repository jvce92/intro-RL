from abc import ABC, abstractmethod

class BaseState(ABC):
	def __init__(self, state):
		self.state = self._internal_representation(state)

	@abstractmethod
	def _internal_representation(self, state):
		pass

	@abstractmethod
	def __hash__(self):
		pass

	@abstractmethod
	def __eq__(self, other):
		pass

	@abstractmethod
	def __repr__(self):
		pass

	def __call__(self):
		return self.state

	def __ne__(self, other):
		return not (self == other)