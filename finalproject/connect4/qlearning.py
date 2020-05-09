#! /usr/bin/python3.6

import os
import sys
import random
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

    def learn(self, board, actions, chosen_action, game_over, game_logic):
        """
        Run the Q-Learning algorithm
        """
        reward = 0
        if (game_over):
            win_value = game_logic.get_winner()
            if win_value == 0:
                reward = 0.5
            elif win_value == self.coin_type:
                reward = 1
            else:
                reward = -2
        prev_state = board.get_prev_state()
        prev = self.getQ(prev_state, chosen_action)
        result_state = board.get_state()
        maxqnew = max([self.getQ(result_state, a) for a in actions])
        self.q[(prev_state, chosen_action)] = prev + self.alpha * ((reward + self.gamma*maxqnew) - prev)
        
    def random_action(self):
        """
        Returns a random column (0-6)
        """
        return random.randint(0,6)

        
    
class QLGrid:
    """
    The class that does Q-learning
    """
    def __init__(self, f, cost, prob):
        self.num_rows = 0
        self.num_cols = 0
        self.exploration = 1
        self.move_cost = cost
        self.move_prob = prob
        self.steps = 0
        self.decay = 0.92
        self.alpha = 0.5
        self.gamma = 0.9
        self.ongoing_reward = 0.0
        self.runs = 0
        self.converge_val = Decimal(0.0)
        self.converge_th = Decimal(0.000000001)
        self.converged_trials = 0
        self.visited_confidence_th = move_cost * 10
        self.converged_time = time.time()                            
        

    def explore(self):
        """
        Returns ture if we rolled for exploration, else false (exploitation)
        """
        ret_val = True
        # Always start with 100% exploration
        if(self.steps == 0):
            return ret_val
        else:
            # Otherwise multiply the current rate by the decay value
            roll = random.uniform(0, self.exploration)
            if(roll > self.exploration):
                ret_val = False
        return ret_val    

    def ql_algorithm(self, x, y, a, new_x, new_y):
        """
        Runs the Q-Learning algorithm to update the current state
        """
        qcurrent = self.grid[x][y].get_action_value(a)
        qnew = self.grid[new_x][new_y].get_max()
        new_total = qcurrent + self.alpha *(self.move_cost + self.gamma * qnew - qcurrent)

        # Update the state
        self.grid[x][y].update(a, new_total)
        


    def epsilon_greedy(self):
        """
        Implements epsilon-greedy for epsilon == 0.1
        """
        roll = random.random()

        if roll <= 0.1:
            return random.randint(0,3)
        else:
            return self.grid[self.agent_y][self.agent_x].get_max_action()  
    
    def learn(self, num_steps):
        """
        Runs the Q-Learning main loop
        """
        
        # Initial exploration
        self.exploration *= self.decay

        # Q-Learning main loop
        initial_steps = self.steps
            
        while(self.steps < num_steps+initial_steps):
            self.runs += 1

            # If we are exploring, then select a random action
            if(self.explore()):
                action = self.random_action()
                #action = self.least_visited()
            # Otherwise, do exploitation
            else:
                action = self.grid[self.agent_y][self.agent_x].get_max_action()

            # Testing random
            #action = self.random_action()

            # Testing epsilon-greedy
            #action = self.epsilon_greedy()

            # Save current state position
            current_x = self.agent_x
            current_y = self.agent_y
            
            # Run trans model to see new location            
            new_state = self.transition(action)

            # Update Q values for prior state based on best move for new location
            self.ql_algorithm(current_y, current_x, action, self.agent_y, self.agent_x)

            # if terminal state then end trial, otherwise loop back top
            if self.grid[self.agent_y][self.agent_x].terminal:
                self.steps += 1
                self.init_agent()
                self.exploration *= self.decay

            
    def print_summary(self):
        """
        Prints number of trials and average reward (performance)
        """
        print("Trials run:", self.steps)
        rwd_str = "Average reward: " + '{:.3f}'.format(self.ongoing_reward / self.runs)       
        print(rwd_str)
        if self.converged_trials == 500:
            print("Time to converge: " + '{:.3f}'.format(time.time() - self.converged_time))
