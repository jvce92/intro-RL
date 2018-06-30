import numpy as np
from abc import ABC, abstractmethod
from random import choice

class Trainer(ABC):
	def __init__(self, env):
		self.env = env
		self.agent = Agent(env)

		self.returns = {}
		for s in env.states:
			for a in env.actions:
				self.returns[(s, a)] = []

	@abstractmethod
	def train(self, num_episodes):
		return


