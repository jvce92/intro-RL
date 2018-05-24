from abc import ABC, abstractmethod
from itertools import product
from random import choice
import numpy as np

METHODS = ["sync", "value_iter"]

class MDP(ABC):
	def __init__(self, states_dim, actions, eps, verbose):
		ranges = []
		for dim in states_dim:
			ranges.append(range(dim))
		self.states = list(product(*ranges))
		self.policy = {}
		self.value_state = {}
		self.action_value = {}
		self.num_states = len(self.states)

		for i in range(self.num_states):
			self.policy[self.states[i]] = choice(actions)
			self.value_state[self.states[i]] = 0

		self.actions = actions
		self.eps = eps
		self.val_it = 0
		self.pol_it = 0
		self.verbose = verbose

	def sync_eval_policy(self):
		delta = 0
		for cur_state in self.states:
			old_val = self.value_state[cur_state]
			self.value_state[cur_state] = self.expected_returns(cur_state, self.policy[cur_state])
			delta = np.maximum(delta, np.abs(old_val - self.value_state[cur_state]))

		return delta

	def sync_improve_policy(self):
		stable = True
		for cur_state in self.states:
			old_action = self.policy[cur_state]
			best_action = old_action
			best_val = 0
			for action in self.actions:
				cur_val = self.expected_returns(cur_state, action)
				if cur_val > best_val:
					best_val = cur_val
					best_action = action

			self.policy[cur_state] = best_action
			
			if self.verbose:
				print("* {0:^16} * {1:^17} * {2:^17} * {3:^17.4f} *".format(
					str(cur_state), 
					old_action, best_action, best_val))
			
			if best_action != old_action:
				stable = False

		return stable

	def eval_policy(self, method="sync"):
		assert(method in METHODS)

		if method == "sync":
			delta = 10 * self.eps
			while delta > self.eps:
				delta = self.sync_eval_policy()

				if self.verbose:
					print("Delta = {0}".format(delta))

	def value_iteration(self, delta):
		it = 0
		for state in self.states:
			old_val = self.value_state[state]
			best_val = old_val
			best_action = self.policy[state]
			for action in self.actions:
				cur_val = self.expected_returns(state, action)
				# print(cur_val, state, action)
				if cur_val > best_val:
					best_action = action
					best_val = cur_val
			self.value_state[state] = best_val
			delta = max(delta, np.abs(old_val - best_val))
			it += 1

			if self.verbose and (it % (self.num_states // 10) == 0):
				print("Delta = {0}".format(delta))

		return delta

	def improve_policy(self, method="sync"):
		assert(method in METHODS)

		if method == "sync":
			stable = False
			while not stable:
				self.val_it += 1
				
				if self.verbose:
					print("Starting Iteration #{0} of policy evaluation".format(self.val_it))

				self.eval_policy(method)
				
				if self.verbose:
					print("="*80)
				
				self.pol_it += 1
				
				if self.verbose:
					print("Starting Iteration #{0} of policy improvement".format(self.pol_it))
					print("*"*80)
					print("* {0:^16} * {1:^17} * {2:^17} * {3:^17} *".format("State", 
																			 "Old Action",
																			 "New Action",
																			 "Best Value"))
					print("*"*80)

				stable = self.sync_improve_policy()
				
				if self.verbose:
					print("*"*80)
					print("="*80)

		if method == "value_iter":
			while True:
				delta = 0
				self.val_it += 1

				if self.verbose:
					print("Starting Iteration #{0} of value iteration".format(self.val_it))

				delta = self.value_iteration(delta)

				if delta < self.eps:
					break

			if self.verbose:
				print("="*80)
				print("Finding optimal policy")
				print("*"*80)
				print("* {0:^16} * {1:^17} * {2:^17} * {3:^17} *".format("State", 
																		 "Old Action",
																		 "New Action",
																		 "Best Value"))
				print("*"*80)

			self.sync_improve_policy()


	@abstractmethod
	def expected_returns(self, state, action):
		return
