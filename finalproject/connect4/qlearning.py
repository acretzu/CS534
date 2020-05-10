#! /usr/bin/python3.6

import sys
import random
#from connect4 import Connect4
from decimal import Decimal

class QLearner:
    """
    This class represents each state
    """
    def __init__(self, player, epsilon_decay=0.99, alpha=0.1, gamma=0.9):
        self.player = player
        self.action_with_max = [0, 1, 2, 3, 4, 5, 6]
        self.q = {}
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma

    def get_max(self):
        """
        Returns the maximum q-value.
        """
        max_pos = int(random.choice(self.action_with_max))

        # Update maximum q-value of the state
        max_value = float(self.q[max_pos])
        self.maximum = max_value
        return max_value

    def update(self, action, value):
        """
        Update action (a) with a value (v)
        """
        self.q[int(action)] = value

        # Update maximum
        self.maximum = max(self.q)

        # Loop thru all directions and update action_with_max
        self.action_with_max = []
        for pos in range(7):
            if self.q[pos] == self.maximum:
                self.action_with_max.append(pos)

    def get_max_action(self):
        """
        Return action with max q-value
        """
        return random.choice(self.action_with_max)

    def get_action_value(self, a):
        """
        Return value from given action
        """
        return self.q[a]

    def random_action(self):
        """
        Returns a random column (0-6)
        """
        return random.randint(0,6)

    def learn(self, board, actions):
        """
        Run the Q-Learning algorithm
        """
        reward = 0
        winner = board.check_winner()
        if (winner > 0):
            if winner == 3: # Draw
                reward = 0.5
            elif winner == self.player: # Win
                reward = 1
            else: # Lose
                reward = -2
        prev_state = board.get_prev_state()
        prev = self.getQ(prev_state, chosen_action)
        result_state = board.get_state()
        maxqnew = max([self.getQ(result_state, a) for a in actions])
        self.q[(prev_state, chosen_action)] = prev + self.alpha * ((reward + self.gamma*maxqnew) - prev)        

    
