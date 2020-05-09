#! /usr/bin/python3.6

import os
import sys
import numpy as np
import random
import time
from decimal import Decimal

class QLState:
    """
    This class represents each state
    """
    def __init__(self):
        self.value = 0.0
        self.position = (0, 0)
        self.terminal = False
        self.visited = 0
        self.maximum = 0.0
        self.action_with_max = [0, 1, 2, 3, 4, 5, 6]
        self.q_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.confidence = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def get_max(self):
        """
        Returns the maximum q-value.
        """
        max_pos = int(random.choice(self.action_with_max))

        # Update maximum q-value of the state
        max_value = float(self.q_values[max_pos])
        self.maximum = max_value
        return max_value

    def update(self, direction, value):
        """
        Update a direction (d) with a value (v)
        """
        self.visited += 1
        self.q_values[int(direction)] = value

        # Update maximum
        self.maximum = max(self.q_values)

        # Loop thru all directions and update action_with_max
        self.action_with_max = []
        for pos in range(7):
            if self.q_values[pos] == self.maximum:
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
        return self.q_values[a]


    
class QLGrid:
    """
    The class that does Q-learning
    """
    def __init__(self, f, cost, prob):
        self.grid = None
        self.num_rows = 0
        self.num_cols = 0
        self.agent_x = 0
        self.agent_y = 0
        self.exploration = 1
        self.move_cost = cost
        self.move_prob = prob
        self.steps = 0
        self.decay = 0.92
        self.alpha = 0.5
        self.gamma = 0.9
        self.load_grid(f)
        self.ongoing_reward = 0.0
        self.runs = 0
        self.converge_val = Decimal(0.0)
        self.converge_th = Decimal(0.000000001)
        self.converged_trials = 0
        self.visited_confidence_th = move_cost * 10
        self.converged_time = time.time()

    def load_grid(self, path_to_file):
        """
        Loads the CSV file as a grid
        """
        # Open file
        file_ptr = open(path_to_file, "r")        
        
        # Error out if we can't open the file
        if not file_ptr:
            print("Unable to open file: %s" % path_to_file)
            sys.exit(1)

        # Count rows and columns for grid initialization
        num_rows = 0
        num_cols = -1
        for line in file_ptr:
            num_rows += 1
            if num_cols == -1:
                num_cols = len(line.split(","))

        # Reset file ptr
        file_ptr.seek(0)
        
        # Default empty 2D array
        self.grid = []
        self.num_rows = num_rows
        self.num_cols = num_cols

        # Parse file
        row = 0
        for line in file_ptr:
            csv_line = line.split(",")
        
            # Add each column
            col = 0
            self.grid.append([])
            for val in csv_line:
                state = QLState()
                value = int(val.strip())
                state.value = value
                state.terminal = True if value > 0 or value < 0 else False
                if state.terminal:
                    state.q_values = [value, value, value, value]
                state.position = (col, row)
                self.grid[row].append(state)
                col += 1
            row += 1

        # Close file
        file_ptr.close()
        self.init_agent()

        
    def init_agent(self):
        """
        Sets the agent in the bottom left corner of the grid
        """
        self.agent_x = 0
        self.agent_y = self.num_rows-1
        
    def __str__(self):
        """
        Prints the grid
        """
        ret_str = ""
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                ret_str += " "
                if c == self.agent_x and r == self.agent_y:
                    ret_str += '{:>2}'.format("A")
                else:
                    ret_str += '{:>2}'.format(self.grid[r][c].value)
                ret_str += " "
            ret_str += "\n"
        return ret_str

    def print_greatest_directions(self):
        """
        Prints the highest direction from all grid states
        """
        print("Learned Policy:")
        ret_str = ""
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                if self.grid[r][c].terminal:
                    ret_str += '{:>5}'.format(self.grid[r][c].value)
                else:
                    ret_str += '{:>5}'.format(self.val2dir(self.grid[r][c].get_max_action()))
                ret_str += " | "
            ret_str += "\n"
        print(ret_str)

    def print_greatest_qvalues(self):
        """
        Prints the highest qvalues for all grid states 
        """
        print("Max Rewards:")
        ret_str = ""
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                #ret_str += " "
                if self.grid[r][c].terminal:
                    ret_str += '{:>7}'.format(self.grid[r][c].value)
                else:
                    ret_str += '{:>7.4f}'.format(self.grid[r][c].get_max())
                ret_str += " | "
            ret_str += "\n"
        print(ret_str)


    def print_visited(self):
        """
        Prints the highest qvalues for all grid states 
        """
        print("Visited Count:")
        ret_str = ""
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                ret_str += " "
                if self.grid[r][c].terminal:
                    ret_str += '{:>5}'.format(self.grid[r][c].value)
                else:
                    ret_str += '{:>5}'.format(self.grid[r][c].visited)
                ret_str += " "
            ret_str += "\n"
        print(ret_str)                
        
    def val2dir(self, v):
        """ 
        Replaces integer direction with string
        """
        ret_str = "NONE"
        if v == 0:
            ret_str = "UP"
        elif v == 1:
            ret_str = "RIGHT"
        elif v == 2:
            ret_str = "DOWN"
        elif v == 3:
            ret_str = "LEFT" 
        return ret_str
    
        
    def move_agent(self, d):
        """
        Move the agent in the given direction
        """
        if d == 0: # UP
            self.agent_y = self.agent_y-1 if self.agent_y > 0  else 0
        elif d == 1: # RIGHT
            self.agent_x = self.agent_x+1 if self.agent_x < self.num_cols-1 else self.num_cols-1
        elif d == 2: # DOWN
            self.agent_y = self.agent_y+1 if self.agent_y < self.num_rows-1 else self.num_rows-1
        elif d == 3: # LEFT
            self.agent_x = self.agent_x-1 if self.agent_x > 0 else 0


    def transition(self, d):
        """
        Roll for moving in desired direction and move the agent
        """
        actual_move = d
        roll = random.random()

        # Move went as planned
        if(roll <= self.move_prob):
            self.move_agent(d)
        # Moving in unplanned direction
        else:
            left = (1 - self.move_prob) / 2
            roll = random.uniform(0, 1-self.move_prob)

            # Check if we move orthogonally left or right
            if(roll <= left):
                #print("Rolled ORTHO left")
                actual_move = d - 1 if d > 0 else 3
                self.move_agent(actual_move)
            else:
                #print("Rolled ORTHO right")
                actual_move = d + 1 if d < 3 else 0
                self.move_agent(actual_move)
        return actual_move

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

        # Calculate performance
        self.ongoing_reward += new_total

        # Calculate convergence
        old_value = Decimal(self.grid[x][y].get_action_value(a))
        new_value = Decimal(new_total)
        self.converge_val = abs(new_value - old_value)

        # Update current state confidence of moving in given direction
        self.grid[x][y].confidence[a] = new_value - old_value
        #print("updated confidence =", self.converge_val)

        # Update the state
        self.grid[x][y].update(a, new_total)

    def least_visited(self):
        """
        Calculates least visited from current state and returns an action
        """
        # Top
        visited_up = 999999999
        if self.agent_y > 0:
            y = self.agent_y-1 
            visited_up = self.grid[y][self.agent_x].visited            

        # Right
        visited_right = 99999999
        if self.agent_x < self.num_cols-1:
            x = self.agent_x+1
            visited_right = self.grid[self.agent_y][x].visited


        # Down
        visited_down = 99999999
        if self.agent_y < self.num_rows-1:
            y = self.agent_y+1
            visited_down = self.grid[y][self.agent_x].visited

        # Left
        visited_left = 99999999
        if self.agent_x > 0:
            x = self.agent_x-1
            visited_left = self.grid[self.agent_y][x].visited

        # Find least visited
        least_visited = min(visited_up, visited_right, visited_down, visited_left)

        # Determine which direction was least visited
        pick_array = []
        if visited_up == least_visited:
            pick_array.append(0)
        if visited_right == least_visited:
            pick_array.append(1)
        if visited_down == least_visited:
            pick_array.append(2)
        if visited_left == least_visited:
            pick_array.append(3)

        # Randomly pick least visited direction if there are multiple
        action = random.choice(pick_array)

        # Get confidence level from action
        conf = self.grid[self.agent_y][self.agent_x].confidence[action]

        # Repick if confidence is under threshold
        while conf <= self.visited_confidence_th and len(pick_array) > 0:
            print("below conf th: action = ", action, ", conf = ", conf)
            pick_array.remove(action)
            if len(pick_array) == 0:
                action = random.randint(0,3)
                break
            
            action = random.choice(pick_array)
            conf = self.grid[self.agent_y][self.agent_x].confidence[action]

        return action
        
    def random_action(self):
        """
        Returns a random action (0 = up, 1 = right, 2 = down, 3 = left)
        """
        return random.randint(0,3)

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

            
    def learn_20_seconds(self):
        """
        Function to train for 20 seconds or until convergence
        """
        # Take start time to run for 20 seconds if trials == 0
        print("Input trials is 0. Learning...")
        self.converged_time = time.time()

        while(True):
            self.learn(1)
            if (time.time() - self.converged_time) >= 20.0:
                break

            # Update converge counter
            if(self.converge_val / Decimal(self.runs) < self.converge_th):
                self.converged_trials += 1
            else:
                self.converge_trials = 0

            # Stop after hitting 1000 consecutive converged trials
            if self.converged_trials == 500:
                break

    def print_summary(self):
        """
        Prints number of trials and average reward (performance)
        """
        print("Trials run:", self.steps)
        rwd_str = "Average reward: " + '{:.3f}'.format(self.ongoing_reward / self.runs)       
        print(rwd_str)
        if self.converged_trials == 500:
            print("Time to converge: " + '{:.3f}'.format(time.time() - self.converged_time))
