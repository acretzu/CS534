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
        self.q_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.q = {}
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = 1.0
        print("Qlearner is player", self.player)

    def getQ(self, state, action):
        """
        Return a probability for a given state and action where the greater
        the probability the better the move
        """
        # encourage exploration; "optimistic" 1.0 initial values
        if self.q.get((state, action)) is None:
            self.q[(state, action)] = 0.0
        return self.q.get((state, action))

    def choose_max_action(self, aa):
        """
        Return an action based on the best move recommendation by the current
        Q-Table with a epsilon chance of trying out a new move
        """
        qs = [self.getQ(current_state, a) for a in aa]
        maxQ = max(qs)

        if qs.count(maxQ) > 1:
            # more than 1 best option; choose among them randomly
            best_options = [i for i in range(len(aa)) if qs[i] == maxQ]
            i = random.choice(best_options)
        else:
            i = qs.index(maxQ)

        return a[i]    
    
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
        qs = [self.getQ(current_state, a) for a in actions]
        maxQ = max(qs)

        if qs.count(maxQ) > 1:
            # more than 1 best option; choose among them randomly
            best_options = [i for i in range(len(actions)) if qs[i] == maxQ]
            i = random.choice(best_options)
        else:
            i = qs.index(maxQ)

    def get_action_value(self, a):
        """
        Return value from given action
        """
        #return self.q[a]
        

    def random_action(self, aa):
        """
        Returns a random column from the available_actions (aa)
        """
        return random.choice(aa)

    def explore(self):
        """
        Returns ture if we rolled for exploration, else false (exploitation)
        """
        ret_val = True
        # Always start with 100% exploration
        if(self.explore == 1):
            ret_val = True

        else:
            # Otherwise multiply the current rate by the decay value
            roll = random.uniform(0, self.exploration)
            if(roll > self.exploration):
                ret_val = False

        self.exploration *= self.epsilon_decay
        print("explore?", ret_val, ", exploration =", self.exploration)
        return ret_val    

    def learn(self, connect4):
        """
        Run the Q-Learning algorithm
        """
        # If we are exploring, then select a random action
        if(self.explore()):
            action = self.random_action(connect4.available_columns())            
            # Otherwise, do exploitation
        else:
            action = self.choose_max_action(connect4.available_columns())

        print("decided action = ", action)

        print("qlearner before:")        
        print(connect4)

        # Save current state
        current_board = connect4.get_state()
        current_q = self.getQ(current_board, action)
        
        # Perform transition (i.e. Place the piece)
        connect4.place(action)
        print("qlearner after:")
        print(connect4)

        new_board = connect4.get_state()
        new_q = max([self.getQ(new_board, a) for a in connect4.available_columns()])
        
        reward = -0.01
        winner = connect4.has_winner()
        if (winner is not 0):
            if winner == self.player: # Win
                reward = 1
            else: # Lose
                reward = -1
        if (connect4.full()):
            reward = 0.5        
        self.q[(current_board, action)] = current_q + self.alpha * ((reward + self.gamma*new_q) - current_q)
        print("updated to:", self.q[(current_board, action)])
        
        # Update       
        # qold = self.get_action_value(connect4.prev_board, chosen_action)
        # #qnew = self.get_
        # result_state = connect4.board
        # maxqnew = max([self.getQ(result_state, a) for a in actions])
        # self.q[(prev_state, chosen_action)] = prev + self.alpha * ((reward + self.gamma*maxqnew) - prev)        
        
        # reward = 0
        # winner = connect4.has_winner()
        # if (winner > 0):
        #     if winner == 3: # Draw
        #         reward = 0.5
        #     elif winner == self.player: # Win
        #         reward = 1
        #     else: # Lose
        #         reward = -2
        

    
