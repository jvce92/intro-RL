from random import choice, seed
from statistics import mean
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
from sys import argv
import json

CARDS = ["Ace", "2", 
		 "3", "4", "5", 
		 "6", "7", "8", 
		 "9", "10", "Jack", 
		 "Queen", "King"]

CARD_VALUE = {"Ace":1,
			  "2":2, 
		 	  "3":3, 
		 	  "4":4, 
		 	  "5":5, 
		 	  "6":6, 
		 	  "7":7, 
		 	  "8":8, 
		 	  "9":9, 
		 	  "10":10, 
		 	  "Jack":10, 
		 	  "Queen":10, 
		 	  "King":10}

ACTIONS = ["Hit", "Stick"]

REWARD = {"Win":1,
		  "Loss":-1,
		  "Draw":0}

VERBOSE = True

seed(a=42)

def eval_hand(hand):
	hand_sum = 0;
	aces = 0
	usable_ace = False
	for card in hand:
		if card == "Ace":
			aces += 1
			continue

		hand_sum += CARD_VALUE[card]

	for i in range(aces):
		if hand_sum <= 10:
			hand_sum += 11
			usable_ace = True
		else:
			usable_ace = False
			hand_sum += 1


	return hand_sum, usable_ace

def draw_hands():
	pl_hand = [choice(CARDS), choice(CARDS)]
	dl_hand = [choice(CARDS), choice(CARDS)]

	return pl_hand, dl_hand

def check_game(pl_hand, dl_hand):
	pl_sum, _ = eval_hand(pl_hand)
	dl_sum, _ = eval_hand(dl_hand)

	if pl_sum > 21:
		return REWARD["Loss"]

	if dl_sum > 21:
		return REWARD["Win"]

	if pl_sum == dl_sum:
		return REWARD["Draw"]

	if pl_sum > dl_sum:
		return REWARD["Win"]
	else:
		return REWARD["Loss"]

	return None

def initialize(init_action_value=False):
	value_state = {}
	if init_action_value:
		action_value = {}
		policy = {}
	counts = {}
	states = []
	for usable_ace in [True, False]:
		for pl_hand in range(12, 22):
			for dl_showing in range(1,11):
				s = (usable_ace, pl_hand, dl_showing)
				states.append(s)
				value_state[s] = 0
				if init_action_value:
					policy[s] = choice(ACTIONS)
					for a in ACTIONS:
						action_value[(s, a)] = 0
						counts[(s, a)] = 1
				else:
					counts[s] = 1

	if init_action_value:
		return action_value, policy, counts, states

	return value_state, counts, states

def play_ep(policy, a0):
	pl, dl = draw_hands()
	while True:
		pl_sum, usable_ace = eval_hand(pl)
		if pl_sum >= 12:
			break
		pl.append(choice(CARDS))
		
	steps = []
	actions = []
	players_turn = True
	first_play = True
	while True:
		if players_turn:
			pl_sum, usable_ace = eval_hand(pl)
			if pl_sum > 21:
				break
			s = (usable_ace, pl_sum, CARD_VALUE[dl[0]])
			if pl_sum == 21:
				steps.append(s)
				actions.append("Stick")
				players_turn = False
				continue
			steps.append(s)
			if a0 is not None and first_play:
				first_play = False
				a = a0
			else:
				a = policy[s]
			actions.append(a)
			if a is "Hit":
				pl.append(choice(CARDS))
			elif a is "Stick":
				players_turn = False
		else:
			dl_sum, _ = eval_hand(dl)
			if dl_sum > 21:
				break

			if dl_sum < 17:
				dl.append(choice(CARDS))
			else:
				break

	reward = check_game(pl, dl)

	if VERBOSE:
		print("Steps {0}".format(steps))
		print("Actions {0}".format(actions))
		print("Player Hand {0}".format(pl))
		print("Dealer Hand {0}".format(dl))
		print("Reward {0}".format(reward))
		print()

	return steps, actions, reward

def first_visit_MC_eval(policy, epochs):
	value_state, counts, states = initialize()

	for ep in range(epochs):
		if VERBOSE:
			print("Game #{0}".format(ep+1))
		
		steps, _, r = play_ep(policy, None)
		for t, s in enumerate(steps):
			if s not in steps[:t]:
				counts[s] += 1
				value_state[s] += (1.0/counts[s]) * (r - value_state[s])

	return value_state, states

def exploring_starts_MC(epochs):
	action_value, policy, counts, states = initialize(True)

	for ep in range(epochs):
		if VERBOSE:
			print("Game #{0}".format(ep+1))
		a0 = choice(ACTIONS)
		steps, actions, r = play_ep(policy, a0)
		av_pairs = list(zip(steps, actions))
		for t, s in enumerate(steps):
			if (s, actions[t]) not in av_pairs[:t]:
				counts[(s, actions[t])] += 1
				action_value[(s, actions[t])] += (1.0/counts[(s, actions[t])]) * (r - action_value[(s, actions[t])])
				policy[s] = "Hit" if action_value[(s, "Hit")] >= action_value[(s, "Stick")] else "Stick" 

	return action_value, policy

def fixed_policy(threshold):
	policy = {}
	for usable_ace in [True, False]:
		for pl_hand in range(12, 22):
			for dl_showing in range(1,11):
				s = (usable_ace, pl_hand, dl_showing)
				if pl_hand >= threshold:
					policy[s] = "Stick"
				else:
					policy[s] = "Hit"

	return policy

def plot_value_state(value_state, states, title):
	x, y, z = [], [], []
	x_usable_ace, y_usable_ace, z_usable_ace = [], [], []

	for s in states:
		usable_ace, pl_sum, dl_showing = s
		if usable_ace:
			x_usable_ace.append(pl_sum)
			y_usable_ace.append(dl_showing)
			z_usable_ace.append(value_state[s])
		else:
			x.append(pl_sum)
			y.append(dl_showing)
			z.append(value_state[s])

	fig = plt.figure(figsize=(12,9))
	ax = fig.gca(projection='3d')
	# ax.scatter(x, y, z)
	ax.plot_trisurf(x, y, z, color="red", linewidth=0.1)
	ax.set_title(title + " (No Usable Ace)", size=20)
	ax.set_xlabel("Player Sum", size=18)
	ax.set_xlim(12, 21)
	ax.set_ylabel("Dealer Showing", size=18)
	ax.set_ylim(1, 10)
	ax.set_zlabel("Expected Profit", size=18)
	ax.set_zlim(-1, 1)
	# fig.show()

	fig_usable_ace = plt.figure(figsize=(12,9))
	ax_usable_ace = fig_usable_ace.gca(projection='3d')
	# ax_usable_ace.scatter(x_usable_ace, y_usable_ace, z_usable_ace)
	ax_usable_ace.plot_trisurf(x_usable_ace, y_usable_ace, z_usable_ace, color="red", linewidth=0.1)
	ax_usable_ace.set_title(title + " (Usable Ace)", size=20)
	ax_usable_ace.set_xlabel("Player Sum", size=18)
	ax_usable_ace.set_xlim(12, 21)
	ax_usable_ace.set_ylabel("Dealer Showing", size=18)
	ax_usable_ace.set_ylim(1, 10)
	ax_usable_ace.set_zlabel("Expected Profit", size=18)
	ax_usable_ace.set_zlim(-1, 1)
	# fig_usable_ace.show()


def plot_policy(policy, title):
	stick_us_x = []
	stick_us_y = []
	stick_x = []
	stick_y = []
	hit_us_x = []
	hit_us_y = []
	hit_x = []
	hit_y = []
	for key in policy.keys():
		if policy[key] == "Hit":
			if key[0]:
				hit_us_x.append(key[1])
				hit_us_y.append(key[2])
			else:
				hit_x.append(key[1])
				hit_y.append(key[2])
		elif policy[key] == "Stick":
			if key[0]:
				stick_us_x.append(key[1])
				stick_us_y.append(key[2])
			else:
				stick_x.append(key[1])
				stick_y.append(key[2])

	us_fig = plt.figure(figsize=(12,9))
	fig = plt.figure(figsize=(12,9))

	us_ax = us_fig.add_subplot(111)
	us_ax.set_title(title + " (Usable Ace)")
	us_ax.set_ylabel("Player Sum", size=18)
	us_ax.set_xlabel("Dealer Showing", size=18)
	us_ax.scatter(hit_us_y, hit_us_x, color='r', label="Hit")
	us_ax.scatter(stick_us_y, stick_us_x, color='b', label="Stick")
	us_ax.legend()

	ax = fig.add_subplot(111)
	ax.set_title(title + " (No Usable Ace)")
	ax.set_ylabel("Player Sum", size=18)
	ax.set_xlabel("Dealer Showing", size=18)
	ax.scatter(hit_y, hit_x, color='r', label="Hit")
	ax.scatter(stick_y, stick_x, color='b', label="Stick")
	ax.legend()

if __name__ == "__main__":
	assert(len(argv) >= 2)
	epochs = int(argv[1])
	policy = fixed_policy(20)
	value_state, states = first_visit_MC_eval(policy, epochs)
	if VERBOSE:
		value_state_json = {}
		for key in value_state.keys():
			value_state_json[str(key)] = value_state[key]

		with open("val_state.log", "w") as f:
			f.write(json.dumps(value_state_json))

	plot_value_state(value_state, states, "Figure 5.1")

	action_value, policy = exploring_starts_MC(epochs)
	value_state, states = first_visit_MC_eval(policy, epochs)
	plot_value_state(value_state, states, "Figure 5.2")
	plot_policy(policy, "Figure 5.2")

	plt.show()

