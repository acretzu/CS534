#! /usr/bin/python3.6

import sys
import random
#from connect4 import Connect4
from decimal import Decimal


class QLearner:
    """
    This class represents each state
    """
    def __init__(self, player, c4, epsilon_decay=0.9, alpha=0.3, gamma=0.9):
        self.player = player
        self.q = {}
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = 1.0

        self.current_q = 0.0
        self.new_q = 0.0
        self.last_tuple = None
        self.start = False
        self.connect4 = c4
        
        #print("Qlearner is player", self.player)
        
    def getQ(self, state, action):
        """
        Return a probability for a given state and action where the greater
        the probability the better the move
        """
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
        #print("explore?", ret_val, ", exploration =", self.exploration)
        return ret_val    

    def check_if_lost(self):
        if not self.start:
            return
        
        # Check if we lost/draw
        reward = -0.01
        winner = self.connect4.has_winner()
        #print("check if lost, w = ", winner)
        if (winner is not 0 or self.connect4.full()):
            if self.connect4.full(): # Draw
                #print("QL DRAW")
                reward = 0.5
            else:
                #print("QL LOSER!")
                reward = -1 # Lose

            #print("QLearner prev q value = ", self.q[self.last_tuple])
            #print("new_q = ", self.new_q)
            #print("current_q = ", self.current_q)
            self.q[self.last_tuple] = self.current_q + self.alpha * ((reward + self.gamma*self.new_q) - self.current_q)
            #print("QLearner new q value = ", self.q[self.last_tuple])

    
    def learn(self):
        """
        Run the Q-Learning algorithm
        """
        if self.start == False:
            self.start = True
            #print("changing start to :", self.start)
            
        # If we are exploring, then select a random action
        if(self.explore()):
            action = self.random_action(self.connect4.available_columns())            
            # Otherwise, do exploitation
        else:
            action = self.choose_max_action(self.connect4.available_columns())

        #print("decided action = ", action)

        #print("qlearner before:")        
        #print(self.connect4)

        # Save current state
        current_board = self.connect4.get_state()
        current_q = self.getQ(current_board, action)
        
        # Perform transition (i.e. Place the piece)
        self.connect4.place(action)
        #print("qlearner after:")
        #print(self.connect4)

        # Check if we won or draw
        reward = -0.01
        winner = self.connect4.has_winner()
        if (winner is not 0 or self.connect4.full()):
            if winner == self.player: # Win
                #print("QL Win!")
                reward = 1
            elif self.connect4.full(): # Draw
                print("QL Draw")
                reward = 0.5
        
        new_board = self.connect4.get_state()

        if self.connect4.full():
            new_q = 0.5
        else:
            new_q = max([self.getQ(new_board, a) for a in self.connect4.available_columns()])                
        self.q[(current_board, action)] = current_q + self.alpha * ((reward + self.gamma*new_q) - current_q)

        # Save update state to update if we lose
        self.last_tuple = (current_board, action)
        self.current_q = current_q
        self.new_q = new_q
        
        #print("updated to:", self.q[(current_board, action)])
        
    
