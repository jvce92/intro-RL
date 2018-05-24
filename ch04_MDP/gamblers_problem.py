import numpy as np
from matplotlib import pyplot as plt
from common import MDP

class GamblersProblem(MDP):
    def __init__(self, goal=100, prob_heads=.5, eps=1e-4, gamma=1.0,
                 verbose=False):
        self.goal = goal
        self.ph = prob_heads
        self.gamma = gamma
        states_dim = [goal+1]
        actions = list(range(1, goal))
        super().__init__(states_dim, actions, eps, verbose)
        self.value_state[(self.goal,)] = 1
        self.policy[(0,)] = 0

    def expected_returns(self, state, action):
        if state[0] == 0:
            return 0

        if state[0] == self.goal:
            return 1

        if action > min(state[0], self.goal - state[0]):
            return 0
    
        next_state_win = tuple([state[0] + action])
        next_state_loss = tuple([state[0] - action])
        
        return self.ph * self.gamma * self.value_state[next_state_win] + \
               (1 - self.ph) * self.gamma * self.value_state[next_state_loss]

    def plot_value_function(self):
        x = np.zeros(self.num_states)
        y = np.zeros(self.num_states)

        for i, s in enumerate(self.states):
            x[i] = s[0]
            y[i] = self.value_state[s]

        fig = plt.figure(figsize=(12, 9))
        ax = fig.gca()

        val_plot = ax.plot(x, y, lw=2)
        ax.set_title("Gambler's Problem with Ph = {0}".format(self.ph),
                     size=22)
        ax.set_xlabel("Capital", size=18)
        ax.set_xlim(0, self.goal)
        ax.set_ylabel("Value Estimates", size=18)
        ax.set_ylim(0, 1.1)
        plt.show()

    def plot_policy(self):
        x = np.zeros(self.num_states)
        y = np.zeros(self.num_states)

        for i, s in enumerate(self.states):
            x[i] = s[0]
            y[i] = self.policy[s]

        fig = plt.figure(figsize=(12, 9))
        ax = fig.gca()

        val_plot = ax.plot(x, y, lw=2, drawstyle='steps-pre')
        ax.set_title("Gambler's Problem with Ph = {0}".format(self.ph),
                     size=22)
        ax.set_xlabel("Capital", size=18)
        ax.set_xlim(0, self.goal)
        ax.set_ylabel("Policy", size=18)
        ax.set_ylim(0, self.goal)
        plt.show()
