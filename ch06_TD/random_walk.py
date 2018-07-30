from random import choice, seed
import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from matplotlib import cm

EVENTS = ["Left", "Right"]
EVENTS_MAP = {"Left" : -1,
			  "Right":  1}
STATES = ["END_LEFT", "A", "B", "C", "D", "E", "END_RIGHT"]
INITIAL_STATE = 3
TERMINAL_STATES = ["END_LEFT", "END_RIGHT"]
VERBOSE = False
GRAPH = ("O" + "-"*5)*6 + "O"
seed(a=0)


def run_episode_TD(value_state, step_size, discount):
	s = INITIAL_STATE
	while STATES[s] not in TERMINAL_STATES:
		e = choice(EVENTS)
		if VERBOSE:
			g = GRAPH[:s*6] + "*" + GRAPH[s*6+1:]
			print(g)
			print("Transition: {0}".format(e))

		next_s = s + EVENTS_MAP[e]
		if STATES[next_s] == "END_RIGHT":
			reward = 1
		else:
			reward = 0

		value_state[s] += step_size * (reward + discount * value_state[next_s] - value_state[s])
		s = next_s

	return value_state

def TD0(epochs, step_size, discount):
	value_state = np.zeros(len(STATES))
	value_state[1:6] = 0.5
	
	for ep in range(epochs):
		if VERBOSE:
			print("Game #{0}".format(ep+1))

		value_state = run_episode_TD(value_state, step_size, discount)
		if VERBOSE:
			print()

	return value_state

def plot_value_state(value_state, ax, steps):
	ax.plot(range(1,6), value_state[1:6], '--ob', lw=2)
	ax.annotate(str(steps), xy=(5, value_state[5]), xytext=(5+0.3, value_state[5]), arrowprops=dict(arrowstyle='->'))

if __name__ == "__main__":
	assert(len(argv) >= 4)
	step_size = float(argv[1])
	discount = float(argv[2])
	
	epochs = []
	for i in range(len(argv) - 3):
		epochs.append(int(argv[3+i]))
	value_state_TD = []

	true_values = np.array([i/6 for i in range(1, 6)])
	fig, ax = plt.subplots(nrows=2, ncols=1)

	ax[0].plot(range(1,6), true_values, '-or', lw=2, label="True")
	ax[0].annotate("True Value", xy=(5, true_values[4]), xytext=(5+0.1, true_values[4]+0.1), arrowprops=dict(arrowstyle='->'))

	rmse_TD = []
	for i, n_steps in enumerate(epochs):
		V_TD = TD0(n_steps, step_size, discount)
		value_state_TD.append(V_TD)
		plot_value_state(value_state_TD[i], ax[0], n_steps)

	ax[0].axis([0, 6, 0, 1])
	ax[0].set_xticks(range(7))
	ax[0].set_xticklabels(STATES)

	n_samples = 50
	num_eps = np.arange(1, max(epochs)+1)
	alpha = [.05, .15, .25]
	colors = ['-b', '-r', '-g']

	for c, a in enumerate(alpha):
		rmse_TD = np.zeros(len(num_eps))
		for i, ep in enumerate(num_eps):
			for samp in range(n_samples):
				V_TD = TD0(ep, a, discount)
				rmse_TD[i] += (np.linalg.norm(V_TD[1:6] - true_values) / (np.sqrt(5) * n_samples))

		ax[1].plot(num_eps, rmse_TD, colors[c], lw=2, label=r"$\alpha = {0}$".format(a))

	ax[1].set_xlabel("Number of Episodes")
	ax[1].set_ylabel("RMS Error")
	ax[1].legend()

	plt.show()