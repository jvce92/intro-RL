from abc import ABC, abstractmethod

class BaseState(ABC):
	def __init__(self, action):
		self.action = action

	@abstractmethod
	def _internal_representation(self, action):
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
		return self.action

	def __ne__(self, other):
		return not (self == other)