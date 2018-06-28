from abc import ABC, abstractmethod
import numpy as np

CARDS = ["Ace", "2", "3", "4", 
         "5", "6", "7", "8", "9", "10",
         "Jack", "Queen", "King"]

SUITS = ["Clubs", "Hearts", "Diamonds", "Spades"]

VALUES = {"Ace"     : None,
          "2"       :   2, 
          "3"       :   3, 
          "4"       :   4, 
          "5"       :   5, 
          "6"       :   6, 
          "7"       :   7, 
          "8"       :   8, 
          "9"       :   9, 
          "10"      :   10,
          "Jack"    :   10, 
          "Queen"   :   10, 
          "King"    :   10}

class Card:
    def __init__(self, card, suit):
        assert(card in CARDS)
        assert(suit in SUITS)

        self.card = card
        self.suit = suit
        self.value = VALUES[card]

    def __add__(self, other_card):
        if self.card == "Ace" and other_card.card != "Ace":
            if 11 + other_card.value <= 21:
                return 11 + other_card.value
            else:
                return 1 + other_card.value

        if self.card != "Ace" and other_card.card == "Ace":
            if self.value + 11 <= 21:
                return self.value + 11
            else:
                return self.value + 1

        return self.value + other_card.value

    def __add__(self, int_val):
        if self.card == "Ace" :
            if 11 + int_val <= 21:
                return 11 + int_val
            else:
                return 1 + int_val
        
        return self.value + int_val

    __radd__ = __add__

    def __repr__(self):
        return self.card + " of " + self.suit

class InfiniteDeck:
    def __init__(self):
        self.prob_card = np.array([4.0/52 for i in range(13)])
        self.prob_suit = np.array([13.0/52 for i in range(4)])

    def draw_card(self):
        return Card(np.random.choice(CARDS, replace=False, p=self.prob_card), 
                    np.random.choice(SUITS, replace=False, p=self.prob_suit))

    
