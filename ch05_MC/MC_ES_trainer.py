from ..common import Trainer, Agent
from random import choice
from statistics import mean

class MC_ES_Trainer(Trainer):
	def __init__(self, env):
		super().__init__(env)

	def train(self, num_episodes):
		for ep in range(num_episodes):
			s = choice(env.states)
			actions = []
			states = [s]
			rewards = [0]
			G = 0
			T = 0

			while True:
				a = self.agent.get_action(state)
				actions.append(a)
				next_s, r = self.env.interact(a)
				
				if next_s == env.final_state:
					break

				else:
					T += 1
					rewards.append(r)
					s = next_s
					states.append(s) 

			for t in range(T-1, -1, -1):
				G += rewards[t+1]
				self.returns[(states[t], actions[t])].append(G)
				self.agent.update_Q(states[t], actions[t],
									mean(self.returns[(states[t], actions[t])]))
				self.agent.update_policy(states[t], 
										 self.agent.get_max_action(states[t]))
