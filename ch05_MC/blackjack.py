from random import choice, seed
from statistics import mean
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
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

def first_visit_MC_eval(policy, epochs):
	# occ = {}
	value_state = {}
	returns = {}
	states = []
	for usable_ace in [True, False]:
		for pl_hand in range(12, 22):
			for dl_showing in range(1,11):
				s = (usable_ace, pl_hand, dl_showing)
				states.append(s)
				value_state[s] = 0
				returns[s] = []

	for ep in range(epochs):
		a = None
		pl, dl = draw_hands()
		r = None
		steps = []
		while True:
			if a is not "Stick":
				pl_sum, usable_ace = eval_hand(pl)
				if pl_sum > 21:
					break

				s = (usable_ace, pl_sum, CARD_VALUE[dl[0]])
				if pl_sum >= 12:
					steps.append(s)
				a = policy(s)
				if a is "Hit":
					pl.append(choice(CARDS))
				
			else:
				dl_sum, _ = eval_hand(dl)
				if dl_sum > 21:
					break

				if dl_sum < 17:
					dl.append(choice(CARDS))
				else:
					break

		# occ[(pl_sum, CARD_VALUE[dl[0]])] += 1
		r = check_game(pl, dl)
		for t, s in enumerate(steps):
			if s not in steps[:t]:
				returns[s].append(r)
				value_state[s] = mean(returns[s])

		if VERBOSE:
			print("Game #{0}".format(ep+1))
			print("State {0}".format(s))
			print("Player Hand {0}".format(pl))
			print("Dealer Hand {0}".format(dl))
			print("Reward {0}".format(r))
			print()

	return value_state, states

def fixed_policy(state, threshold):
	usable_ace, pl_sum, dl_showing = state
	if pl_sum >= threshold:
		return "Stick"

	return "Hit"

def plot_value_state(value_state, states):
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
	ax.scatter(x, y, z)
	ax.set_title("No Usable Ace", size=20)
	ax.set_xlabel("Player Sum", size=18)
	ax.set_xlim(12, 21)
	ax.set_ylabel("Dealer Showing", size=18)
	ax.set_ylim(1, 10)
	ax.set_zlabel("Expected Profit", size=18)
	ax.set_zlim(-1, 1)
	fig.show()

	fig_usable_ace = plt.figure(figsize=(12,9))
	ax_usable_ace = fig_usable_ace.gca(projection='3d')
	ax_usable_ace.scatter(x_usable_ace, y_usable_ace, z_usable_ace)
	ax_usable_ace.set_title("Usable Ace", size=20)
	ax_usable_ace.set_xlabel("Player Sum", size=18)
	ax_usable_ace.set_xlim(12, 21)
	ax_usable_ace.set_ylabel("Dealer Showing", size=18)
	ax_usable_ace.set_ylim(1, 10)
	ax_usable_ace.set_zlabel("Expected Profit", size=18)
	ax_usable_ace.set_zlim(-1, 1)
	fig_usable_ace.show()


if __name__ == "__main__":
	assert(len(argv) >= 2)
	epochs = int(argv[1])
	policy = lambda x: fixed_policy(x, 20)
	value_state, states = first_visit_MC_eval(policy, epochs)
	if VERBOSE:
		value_state_json = {}
		for key in value_state.keys():
			value_state_json[str(key)] = value_state[key]

		with open("val_state.log", "w") as f:
			f.write(json.dumps(value_state_json))

	plot_value_state(value_state, states)
	input()
	
