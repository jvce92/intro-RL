from common import Agent, Environment, Policy, CARDS, SUITS, VALUES, InfiniteDeck
from itertools import product
import numpy as np

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
                self.card_sum[self.player_idx] += new_card
            
            elif action == "Stick":
                self.last_action[self.player_idx] = "Stick"

            else:
                raise ValueError("The action must be either 'Hit' or 'Stick'")

        elif isinstance(agent, Dealer):
            if action == "Hit":
                new_card = self.deck.draw_card()
                self.cur_state[self.dealer_idx].append(new_card)
                self.last_action[self.dealer_idx] = "Hit"
                self.card_sum[self.dealer_idx] += new_card
                
            elif action == "Stick":
                self.last_action[self.dealer_idx] = "Stick"

            else:
                raise ValueError("The action must be either 'Hit' or 'Stick'")

        else:
            raise ValueError("The action should be take by a Player or Dealer")

    def _reward(self, agent, action):
        if isinstance(agent, Player):
            if self.card_sum[self.player_idx] > 21:
                self.cur_state = "DealerWin"
                return -1
            if self.card_sum[self.dealer_idx] > 21:
                self.cur_state = "PlayerWin"
                return 1
            if self.last_action[self.player_idx] == "Stick" and self.last_action[self.dealer_idx] == "Stick":
                if self.card_sum[self.player_idx] > self.card_sum[self.dealer_idx]:
                    self.cur_state = "PlayerWin"
                    return 1
                elif self.card_sum[self.player_idx] < self.card_sum[self.dealer_idx]:
                    self.cur_state = "DealerWin"
                    return -1
                else:
                    self.cur_state= "Draw"
                    return 0

            return 0
        
        elif isinstance(agent, Dealer):
            if self.card_sum[self.player_idx] > 21:
                self.cur_state = "DealerWin"
                return 1
            if self.card_sum[self.dealer_idx] > 21:
                self.cur_state = "PlayerWin"
                return -1
            if self.last_action[self.player_idx] == "Stick" and self.last_action[self.dealer_idx] == "Stick":
                if self.card_sum[self.player_idx] > self.card_sum[self.dealer_idx]:
                    self.cur_state = "PlayerWin"
                    return -1
                elif self.card_sum[self.player_idx] < self.card_sum[self.dealer_idx]:
                    self.cur_state = "DealerWin"
                    return 1
                else:
                    self.cur_state= "Draw"
                    return 0

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

    def _internal_representation(self, state):
        return (sum(state[0]), sum([state[1]]))


class Dealer(Agent):
    def __init__(self, policy, verbose=False):
        super().__init__(policy, TERMINAL_STATES, verbose)

    def _internal_representation(self, state):
        return (sum(state[0]), sum(state[1]))

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

def evaluate(player, dealer, house, epochs, verbose=False):
    assert(isinstance(house, House))
    assert(isinstance(player, Player))
    assert(isinstance(dealer, Dealer))

    value_state = {}
    returns = {}

    for e in range(epochs):
        if verbose:
            print("="*60)
            print("Game #{0} \nPlayer's Turn".format(e))

        house.restart()
        episode = []

        dealers_turn = False
        
        while True:
            if not dealers_turn:
                s = house.get_state(player)

                if verbose:
                    print(house)

                if s in TERMINAL_STATES:
                    break

                a = player.policy.get_action(s)

                if a == "Stick":
                    dealers_turn = True

                if verbose:
                    print("Player's action : {0}".format(a))

                r = house.get_reward(player, a)
                episode.append((s, a, r))
            else:
                if verbose:
                    print("Dealer's Turn")

                s = house.get_state(dealer)

                if verbose:
                    print(house)

                if s in TERMINAL_STATES:
                    break

                a = dealer.policy.get_action(s)
                r = house.get_reward(dealer, a)

                if verbose:
                    print("Dealer's action : {0}".format(a))

            if verbose:
                print()

        G = 0
        prev_states = []
        for i, step in enumerate(episode):
            s, a, r = step
            s = player._internal_representation(s)
            G += r
            
            if s not in prev_states:
                if s not in returns.keys():
                    returns[s] = []

                returns[s].append(G)
                value_state[s] = np.mean(returns[s])
                prev_states.append(s)

    return value_state


