import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
from sys import argv
from itertools import product

ACTIONS = ["up", "down", "left", "right"]
ACTIONS_MAP = {"up"   		: (0,  1), 
			   "down" 		: (0, -1), 
			   "left" 		: (-1, 0), 
			   "right"		: (1,  0),
			   "up-left"	: (-1, 1), 
			   "up-right"	: (1,  1), 
			   "down-left"	: (-1,-1), 
			   "down-right"	: (1, -1)}
KING_ACTIONS = ACTIONS + ["up-left", "up-right", "down-left", "down-right"]
DIMENSIONS = (10, 7)
WIND_STRENGTH = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
STATES = list(product(range(DIMENSIONS[0]), range(DIMENSIONS[1])))
INITIAL_STATE = (0, 3)
TERMINAL_STATE = (7, 3)

def sarsa(action_value, policy, env, step_size, discount, epochs):
	steps = [0]
	rewards = []
	for ep in range(epochs):
		total_reward = 0
		s = INITIAL_STATE
		a = policy(s, action_value)
		while s != TERMINAL_STATE:
			next_s, r = env(s, a)
			next_a = policy(next_s, action_value)
			action_value[(s, a)] += step_size * (r + discount * action_value[(next_s, next_a)] - action_value[(s, a)])
			s = next_s
			a = next_a
			steps.append(ep)
			total_reward += r
		rewards.append(total_reward)

	return action_value, steps, rewards

def opt_policy(state, action_value, actions):
	avs = [action_value[(state, a)] for a in actions]
	return actions[np.argmax(avs)]

def eps_greedy(state, action_value, actions, eps):
	if np.random.random() < eps:
		return np.random.choice(actions)

	avs = [action_value[(state, a)] for a in actions]
	avs_idx = [i for i, a in enumerate(avs) if a == np.max(avs)]
	return actions[np.random.choice(avs_idx)]

def windy_gridworld(state, action):
	a_val = ACTIONS_MAP[action]
	next_state = (max(min(state[0]+a_val[0], DIMENSIONS[0]-1), 0), 
				  max(min(state[1]+a_val[1], DIMENSIONS[1]-1), 0))
	next_state = (next_state[0],
				  max(min(next_state[1] + WIND_STRENGTH[next_state[0]], DIMENSIONS[1]-1), 0))
	reward = 0 if next_state == TERMINAL_STATE else -1

	return next_state, reward

def stochastic_windy_gridworld(state, action):
	a_val = ACTIONS_MAP[action]
	noise = np.random.choice([-1, 0, 1])
	next_state = (max(min(state[0]+a_val[0], DIMENSIONS[0]-1), 0), 
				  max(min(state[1]+a_val[1], DIMENSIONS[1]-1), 0))
	next_state = (next_state[0],
				  max(min(next_state[1] + WIND_STRENGTH[next_state[0]] + noise, DIMENSIONS[1]-1), 0))
	reward = 0 if next_state == TERMINAL_STATE else -1

	return next_state, reward

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
	eps = 0.1
	step_size = 0.5
	epochs = 200
	discount = 1
	env = lambda s, a: windy_gridworld(s, a)
	policy = lambda s, avs: eps_greedy(s, avs, ACTIONS, eps)
	# regular windy gridworld
	action_value = {}
	for row in range(DIMENSIONS[0]):
		for col in range(DIMENSIONS[1]):
			for a in ACTIONS:
				action_value[((row, col), a)] = 0

	action_value, steps, _ = sarsa(action_value, policy, env, step_size, discount, epochs)
	fig, ax = plt.subplots()
	ax.plot(steps, lw=2)
	ax.set_xlabel("Time Steps")
	ax.set_ylabel("Episodes")
	ax.set_title("Traditional")
	opt_pol = lambda s, avs: opt_policy(s, avs, ACTIONS)
	plot_policy(opt_pol, action_value, " (Traditional)")

	# king's moves windy gridworld
	policy = lambda s, avs: eps_greedy(s, avs, KING_ACTIONS, eps)
	action_value = {}
	for row in range(DIMENSIONS[0]):
		for col in range(DIMENSIONS[1]):
			for a in KING_ACTIONS:
				action_value[((row, col), a)] = 0

	action_value, steps, _ = sarsa(action_value, policy, env, step_size, discount, epochs)
	fig, ax = plt.subplots()
	ax.plot(steps, lw=2)
	ax.set_xlabel("Time Steps")
	ax.set_ylabel("Episodes")
	ax.set_title("King's Moves")
	opt_pol = lambda s, avs: opt_policy(s, avs, KING_ACTIONS)
	plot_policy(opt_pol, action_value, " (King's Moves)")

	# king's moves + stochastic windy gridworld
	env = lambda s, a: stochastic_windy_gridworld(s, a)
	policy = lambda s, avs: eps_greedy(s, avs, KING_ACTIONS, eps)
	action_value = {}
	for row in range(DIMENSIONS[0]):
		for col in range(DIMENSIONS[1]):
			for a in KING_ACTIONS:
				action_value[((row, col), a)] = 0

	action_value, steps, _ = sarsa(action_value, policy, env, step_size, discount, epochs)
	fig, ax = plt.subplots()
	ax.plot(steps, lw=2)
	ax.set_xlabel("Time Steps")
	ax.set_ylabel("Episodes")
	ax.set_title("King's Moves + Stochastic Wind")
	opt_pol = lambda s, avs: opt_policy(s, avs, KING_ACTIONS)
	plot_policy(opt_pol, action_value, " (King's Moves + Stochastic Wind)")

	plt.show()