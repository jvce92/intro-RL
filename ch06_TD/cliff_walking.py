import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from sys import argv
from gridworld import ACTIONS, ACTIONS_MAP, sarsa, eps_greedy, opt_policy
from itertools import product
from copy import deepcopy

DIMENSIONS = (12, 4)
CLIFF = [(i, 0) for i in range(1, 11)]
INITIAL_STATE = (0, 0)
TERMINAL_STATE = (11, 0)
STATES = list(product(range(DIMENSIONS[0]), range(DIMENSIONS[1])))
VAN_EPS_STEPS_COUNT = 0

def q_learn(action_value, policy, env, actions, step_size, discount, epochs):
	steps = [0]
	rewards = []
	opt_pol = lambda s, avs: opt_policy(s, avs, ACTIONS)
	for ep in range(epochs):
		total_reward = 0
		s = INITIAL_STATE
		while s != TERMINAL_STATE:
			a = policy(s, action_value)
			next_s, r = env(s, a)
			max_Q = np.max([action_value[(next_s, a)] for a in actions])
			action_value[(s, a)] += step_size * (r + discount * max_Q - action_value[(s, a)])
			s = next_s
			steps.append(ep)
			total_reward += r
		rewards.append(total_reward)

	return action_value, steps, rewards

def cliff_gridworld(state, action):
	a_val = ACTIONS_MAP[action]
	next_state = (max(min(state[0]+a_val[0], DIMENSIONS[0]-1), 0), 
				  max(min(state[1]+a_val[1], DIMENSIONS[1]-1), 0))
	if next_state in CLIFF:
		next_state = INITIAL_STATE
		reward = -100
	else:
		reward = 0 if next_state == TERMINAL_STATE else -1
	
	return next_state, reward

def vanishing_eps_greedy(state, action_value, actions, eps):
	global VAN_EPS_STEPS_COUNT
	if np.random.random() < (eps / (0.1 * VAN_EPS_STEPS_COUNT + 1)):
		return np.random.choice(actions)

	VAN_EPS_STEPS_COUNT += 1
	avs = [action_value[(state, a)] for a in actions]
	avs_idx = [i for i, a in enumerate(avs) if a == np.max(avs)]
	return actions[np.random.choice(avs_idx)]

def plot_policy(policy, action_value, title=""):
	moves = {}

	for state in STATES:
		action = policy(state, action_value)
		if action in moves.keys():
			moves[action].append(state)
		else:
			moves[action] = [state]

	fig, ax = plt.subplots()
	cmap = iter(cm.rainbow(np.linspace(0, 1, len(moves.keys()))))

	for action in moves.keys():
		x, y = zip(*moves[action])
		ax.scatter(x, y, 50, c=next(cmap), label=action)
	
	ax.annotate("Initial State", xytext=(INITIAL_STATE[0], -1), arrowprops=dict(arrowstyle='->'), xy=INITIAL_STATE)
	ax.annotate("Terminal State", xytext=(TERMINAL_STATE[0], -1), arrowprops=dict(arrowstyle='->'), xy=TERMINAL_STATE)
	ax.set_title("Optimal Policy" + title)
	ax.legend()

if __name__ == "__main__":
	eps = 0.01
	step_size = 0.5
	epochs = 500
	discount = 1
	env = lambda s, a: cliff_gridworld(s, a)
	eps_pol = lambda s, avs: eps_greedy(s, avs, ACTIONS, eps)
	opt_pol = lambda s, avs: opt_policy(s, avs, ACTIONS)
	van_eps_pol = lambda s, avs: vanishing_eps_greedy(s, avs, ACTIONS, eps)
	action_value = {}
	for row in range(DIMENSIONS[0]):
		for col in range(DIMENSIONS[1]):
			for a in ACTIONS:
				action_value[((row, col), a)] = 0

	sarsa_avs = deepcopy(action_value)
	q_learn_avs = deepcopy(action_value)
	vanishing_sarsa_avs = deepcopy(action_value)

	sarsa_avs, sarsa_steps, sarsa_rewards = sarsa(sarsa_avs, eps_pol, env, step_size, discount, epochs)
	vanishing_sarsa_avs, vanishing_sarsa_steps, vanishing_sarsa_rewards = sarsa(vanishing_sarsa_avs, van_eps_pol, env, step_size, discount, epochs)
	q_learn_avs, q_learn_steps, q_learn_rewards = q_learn(q_learn_avs, eps_pol, env, ACTIONS, step_size, discount, epochs)

	fig, ax = plt.subplots()
	ax.plot(range(10, epochs), sarsa_rewards[10:epochs], lw=2, label=r" Sarsa ($\epsilon$-greedy)")
	ax.plot(range(10, epochs), vanishing_sarsa_rewards[10:epochs], lw=2, c='g', label=r" Sarsa (Vanishing $\epsilon$-greedy)")
	ax.plot(range(10, epochs), q_learn_rewards[10:epochs], lw=2, c='r', label=r" Q Learning ($\epsilon$-greedy)")
	ax.set_xlabel("Episodes")
	ax.set_ylabel("Sum of Rewards")
	ax.legend()
	
	plot_policy(opt_pol, sarsa_avs, title=r" Sarsa ($\epsilon$-greedy)")
	plot_policy(opt_pol, vanishing_sarsa_avs, title=r" Sarsa (Vanishing $\epsilon$-greedy)")	
	plot_policy(opt_pol, q_learn_avs, title=r" Q Learning ($\epsilon$-greedy)")

	plt.show()
	