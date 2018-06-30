from common import Agent, Environment, Policy, CARDS, SUITS, VALUES, InfiniteDeck, card_sum
from itertools import product
import numpy as np
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

ACTIONS = ["Hit", "Stick"]
TERMINAL_STATES = ["Draw", "PlayerWin", "DealerWin"]

class House(Environment):
    def __init__(self):
        self.deck = InfiniteDeck()
        self.player_idx = 0
        self.dealer_idx = 1
        self.draw = 0
        self.pl_win = 1
        self.dl_win = 2
        self.card_sum = [None, None]
        self.last_action = [None, None]
        super().__init__(ACTIONS)

    def restart(self):
        self.card_sum = [None, None]
        self.last_action = [None, None]
        self.last_state = None
        self.cur_state = self._get_initial_state()

    def draw_hand(self):
        return [self.deck.draw_card() for i in range(2)]

    def _get_initial_state(self):
        init_state = (self.draw_hand(), self.draw_hand())
        self.card_sum[self.player_idx] = sum(init_state[self.player_idx])
        self.card_sum[self.dealer_idx] = sum(init_state[self.dealer_idx])

        return init_state 

    def _transition(self, agent, action):
        if isinstance(agent, Player):
            if action == "Hit":
                new_card = self.deck.draw_card()
                self.cur_state[self.player_idx].append(new_card)
                self.last_action[self.player_idx] = "Hit"
                self.card_sum[self.player_idx] = card_sum(self.cur_state[self.player_idx])
                self.check_results()
            
            elif action == "Stick":
                self.last_action[self.player_idx] = "Stick"
                self.check_results()

            else:
                raise ValueError("The action must be either 'Hit' or 'Stick'")

        elif isinstance(agent, Dealer):
            if action == "Hit":
                new_card = self.deck.draw_card()
                self.cur_state[self.dealer_idx].append(new_card)
                self.last_action[self.dealer_idx] = "Hit"
                self.card_sum[self.dealer_idx] = card_sum(self.cur_state[self.player_idx])
                self.check_results()
                
            elif action == "Stick":
                self.last_action[self.dealer_idx] = "Stick"
                self.check_results()

            else:
                raise ValueError("The action must be either 'Hit' or 'Stick'")

        else:
            raise ValueError("The action should be take by a Player or Dealer")

    def check_results(self):
        if self.card_sum[self.player_idx] > 21:
            self.cur_state = "DealerWin"
            
        if self.card_sum[self.dealer_idx] > 21:
            self.cur_state = "PlayerWin"
            
        if self.last_action[self.player_idx] == "Stick" and self.last_action[self.dealer_idx] == "Stick":
            if self.card_sum[self.player_idx] > self.card_sum[self.dealer_idx]:
                self.cur_state = "PlayerWin"
                
            elif self.card_sum[self.player_idx] < self.card_sum[self.dealer_idx]:
                self.cur_state = "DealerWin"
                
            else:
                self.cur_state= "Draw"

    def _reward(self, agent):
        if isinstance(agent, Player):
            if self.cur_state == "PlayerWin":
                return 1

            if self.cur_state == "DealerWin":
                return -1

            return 0
        
        if isinstance(agent, Dealer):
            if self.cur_state == "PlayerWin":
                return -1

            if self.cur_state == "DealerWin":
                return 1

            return 0

        else:
            raise ValueError("The action should be take by a Player or Dealer")

    def get_state(self, agent):
        if isinstance(agent, Player):
            if self.cur_state in TERMINAL_STATES:
                return self.cur_state

            return (self.cur_state[self.player_idx], self.cur_state[self.dealer_idx][0]) 
        
        elif isinstance(agent, Dealer):
            if self.cur_state in TERMINAL_STATES:
                return self.cur_state

            return self.cur_state

        else:
            raise ValueError("The agent should be either a Player or a Dealer")


    def __repr__(self):
        if self.cur_state == "Draw":
            hands = "Player's Hand: {0} \nDealer's Hand: {1}".format(self.last_state[self.player_idx], self.last_state[self.dealer_idx])
            return hands + "\nThe game was a draw"
        if self.cur_state == "PlayerWin":
            hands = "Player's Hand: {0} \nDealer's Hand: {1}".format(self.last_state[self.player_idx], self.last_state[self.dealer_idx])
            return hands + "\nThe Player Won"
        if self.cur_state == "DealerWin":
            hands = "Player's Hand: {0} \nDealer's Hand: {1}".format(self.last_state[self.player_idx], self.last_state[self.dealer_idx])
            return hands + "\nThe Dealer Won"

        return  "Player's Hand: {0} \nDealer's Hand: {1}".format(self.cur_state[self.player_idx], self.cur_state[self.dealer_idx])


class Player(Agent):
    def __init__(self, policy, verbose=False):
        super().__init__(policy, TERMINAL_STATES, verbose)
        self.value_state["UsableAce"] = {}
        self.value_state["NoUsableAce"] = {}
        self.returns["UsableAce"] = {}
        self.returns["NoUsableAce"] = {}

    def _is_ace_usable(self, hand):
        hand_sum = 0
        aces = False

        for card in hand:
            if card.card == "Ace":
                aces = True
                continue

            hand_sum = hand_sum + card

        return (hand_sum < 11) and aces

    def _internal_representation(self, state):
        s = ["NoUsableAce", (card_sum(state[0]), card_sum([state[1]]))]

        if self._is_ace_usable(state[0]):
            s[0] = "UsableAce"

        return s

    def evaluate_episode(self, episode):
        G = 0
        prev_states = []
        for i, step in enumerate(episode):
            s, a, r = step
            s = self._internal_representation(s)
            G += r
            
            if self.policy.is_trainable:
                if (s, a) not in prev_states:
                    if (s[1], a) not in self.returns[s[0]].keys():
                        self.returns[s[0]][(s[1], a)] = []

                    # print(s)

                    self.returns[s[0]][(s[1], a)].append(G)
                    self.value_state[s[0]][(s[1], a)] = np.mean(self.returns[s[0]][(s[1], a)])
                    
                    if s[1] in self.policy.policy[s[0]].keys():
                        best_action = a
                        best_return = self.value_state[s[0]][(s[1], a)]
                        for a in self.policy.actions:
                            if (s, a) not in self.value_state[s[0]].keys():
                                continue

                            if self.value_state[s[0]][(s[1], a)] > best_return:
                                self.policy.policy[s[0]][s[1]] = a
                                best_return = self.value_state[s[0]][(s[1], a)]
                    else:
                        self.policy.policy[s[0]][s[1]] = a

                    prev_states.append((s, a))

            else:
                if s not in prev_states:
                    if s[1] not in self.returns[s[0]].keys():
                        self.returns[s[0]][s[1]] = []

                    # print(s)

                    self.returns[s[0]][s[1]].append(G)
                    self.value_state[s[0]][s[1]] = np.mean(self.returns[s[0]][s[1]])
                    prev_states.append(s)

    def plot_value_state(self):
        fig_use = plt.figure(figsize=(12,9))
        ax_use = fig_use.gca(projection='3d')

        x_use, y_use = list(zip(*self.value_state["UsableAce"].keys()))
        vals_use = []

        for xi_use, yi_use in zip(x_use, y_use):
            vals_use.append(self.value_state["UsableAce"][(xi_use, yi_use)])

        surf_use = ax_use.scatter(x_use, y_use, vals_use)
        ax_use.view_init(30, -80)
        ax_use.set_xlabel("Player's Hand", size=18)
        ax_use.set_xlim(12, 21)
        ax_use.set_ylabel("Dealer's Hand", size=18)
        ax_use.set_ylim(0, 11)
        ax_use.set_zlabel("Expected Return", size=18)
        ax_use.set_zlim(-1, 1)
        ax_use.set_title("Usable Ace", size=20)

        fig_not_use = plt.figure(figsize=(12,9))
        ax_not_use = fig_not_use.gca(projection='3d')

        x_not_use, y_not_use = list(zip(*self.value_state["UsableAce"].keys()))
        vals_not_use = []

        for xi_not_use, yi_not_use in zip(x_not_use, y_not_use):
            vals_not_use.append(self.value_state["UsableAce"][(xi_not_use, yi_not_use)])

        surf_not_use = ax_not_use.scatter(x_not_use, y_not_use, vals_not_use)
        ax_not_use.view_init(30, -80)
        ax_not_use.set_xlabel("Player's Hand", size=18)
        ax_not_use.set_xlim(12, 21)
        ax_not_use.set_ylabel("Dealer's Hand", size=18)
        ax_not_use.set_ylim(0, 11)
        ax_not_use.set_zlabel("Expected Return", size=18)
        ax_not_use.set_zlim(-1, 1)
        ax_not_use.set_title("No Usable Ace", size=20)

        plt.show()


class Dealer(Agent):
    def __init__(self, policy, verbose=False):
        super().__init__(policy, TERMINAL_STATES, verbose)

    def _internal_representation(self, state):
        return (card_sum(state[0]), card_sum(state[1]))

class AlwaysStick(Policy):
    def __init__(self):
        self.hit = 0
        self.stick = 1
        super().__init__(ACTIONS)

    def get_action(self, state):
        return self.actions[self.stick]

class FixedPolicy(Policy):
    def __init__(self):
        self.hit = 0
        self.stick = 1
        super().__init__(ACTIONS)

    def get_action(self, state):
        if sum(state[0]) >= 20:
            return self.actions[self.stick]

        return self.actions[self.hit]

class ESPolicy(Policy):
    def __init__(self):
        self.hit = 0
        self.stick = 1
        super().__init__(ACTIONS)
        self.policy["UsableAce"] = {}
        self.policy["NoUsableAce"] = {}
        self.is_trainable = True

    def _is_ace_usable(self, hand):
        hand_sum = 0
        aces = False

        for card in hand:
            if card.card == "Ace":
                aces = True
                continue

            hand_sum = hand_sum + card

        return (hand_sum < 11) and aces

    def get_action(self, state):
        if self._is_ace_usable(state[0]):
            return self.policy["UsableAce"][]

def evaluate(player, dealer, house, epochs, verbose=False):
    assert(isinstance(house, House))
    assert(isinstance(player, Player))
    assert(isinstance(dealer, Dealer))

    for e in range(epochs):
        if verbose:
            print("="*60)
            print("Game #{0} \nPlayer's Turn".format(e))

        house.restart()
        episode = []

        dealers_turn = False
        
        while True:
            if not dealers_turn:
                s_player = deepcopy(house.get_state(player))

                if verbose:
                    print(house)

                if s_player in TERMINAL_STATES:
                    break

                a_player = player.policy.get_action(s_player)
                r_player = house.get_reward(player, a_player)

                if verbose:
                    print("Player's action : {0}".format(a_player))

                if a_player == "Stick":
                    dealers_turn = True
                    if verbose:
                        print("\n***Dealer's Turn***\n")
                    continue

                if (r_player != 0):
                    episode.append((s_player, a_player, r_player))
            else:
                s_dealer = deepcopy(house.get_state(dealer))

                if verbose:
                    print(house)

                if s_dealer in TERMINAL_STATES:
                    r_player = house.get_reward(player)
                    episode.append((s_player, a_player, r_player))
                    break

                a_dealer = dealer.policy.get_action(s_dealer)
                r_dealer = house.get_reward(dealer, a_dealer)

                if verbose:
                    print("Dealer's action : {0}".format(a_dealer))

            if verbose:
                print()

        player.evaluate_episode(episode)


