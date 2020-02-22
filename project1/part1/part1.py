#! /usr/bin/python3.6

import os
import sys
import re
from optparse import OptionParser
import math
import numpy as np
import random


#
# Global Variables
#
__options__ = None
starting_board = []

#
# parse command line
#
def parse_cmd_line_options():
    parser = OptionParser()
    parser.add_option("--e", action="store", type="int", dest="heuristic", default=1, help="The heuristic.")
    parser.add_option("--a", action="store", type="int", dest="algorithm", default=1, help="The algorithm.")
    parser.add_option("--f", action="store", type="string", dest="csv", default="heavy_queens_board.csv", help="The local path to the CSV file.")

    (options, args) = parser.parse_args()

    # Check that all options have been provided
    if not options.heuristic:
        print("Execution requires heurisitic (1 for H1 or 2 for H2).")
        sys.exit(1)

    if not os.path.isfile(options.csv):
        print("Execution requires path to CSV file.")
        sys.exit(1)

    if not options.algorithm:
        print("Execution requires algorithm. (1 for A* or 2 for Hill Climbing).")
        sys.exit(1)

    return options

#
# Open the CSV file and get board information
# Format: <Queen Weight>,<Queen Position>
#
def parse_csv_file():
    file_ptr = open(__options__.csv, "r")
    ret_array = []

    # Error out if we can't open the file
    if not file_ptr:
        print("Unable to open file: %s" % __options__.csv)
        sys.exit(1)

    # Loop thru each line (row) and extract wieght and col
    row = 0
    for line in file_ptr:
        csv_line = line.split(",")

        # Loop thru each col to see if there is a queen
        col = 0
        while col < len(csv_line):
            # Strip away any whitespace (/r or /n) and check for a digit
            if(str.isdigit(csv_line[col].strip())):
                weight = int(csv_line[col].strip())
                # Add row, col, weight as a tuple into array
                ret_array.append((row+1, col+1, weight))            
            col += 1
            
        row += 1

    file_ptr.close()
    return ret_array


################################################
# Make up a Board (N_QueenChess Game)
################################################
def makeup_board(board_size):
    # [(row, col, weight)]

    fake_board = [(random.randint(1, board_size), (i+1), random.randint(1, 9)) for i in range(board_size)]

    return fake_board

################################################
# Board (N_QueenChess Game)
################################################
class N_QueenChess:

    def __init__(self, starting_board):
        starting_board.sort(key=lambda x: x[1])

        # get columns, weights, size from csv
        # columns[0] represent which row the first column queen is at
        self.columns = [(i[0] - 1) for i in starting_board]
        self.weights = [i[2] for i in starting_board]
        self.size = len(self.weights)

    def play(self, queen_to_move, move_to_where):

        """
            Play next move

            Input:
                self
                queen_to_move: the column number (which queen to move)
                move_to_where: move to which row

        """
        # can't move to the original places (which means didn't move)
        assert self.columns[queen_to_move] != move_to_where
        # to make sure row is not out of the board
        assert move_to_where < self.size

        self.columns[queen_to_move] = move_to_where

    def cost(self, queen_to_move, move_to_where):

        """
            Calculate the cost of one pontential move

            Input:
                self
                queen_to_move: the column number (which queen to move)
                move_to_where: move to which row

            Output:
                cost of this move

        """


        cost = (self.weights[queen_to_move] ** 2) * (abs(self.columns[queen_to_move] - move_to_where))

        return cost

    def display(self):

        """
        Display the chess board
        """

        for column in range(self.size):
            for row in range(self.size):
                if column == self.columns[row]:
                    print(self.weights[row], end=' ')
                else:
                    print('.', end=' ')
            print()
        print("---------------------------------")

        
    def h1(self, add_cost = False):

        """
            Check Heuristics 1
                    The lightest Queen across all pairs of Queens attacking each other.
                    Moving that Queen is the minimum possible cost to solve the problem
            Input:
                self:

            Outputs:
                h1_board: h1 for each pontential move, displayed as a board
                h1_current: h1 for current board
        """        
        def calculate_h1(columns, weights, size):
            attack_queen_list = set()

            for queen_column in range(size):
                for compare_column in range(queen_column + 1, size):

                    # check if on the same row
                    if columns[queen_column] == columns[compare_column]:
                        attack_queen_list.add(queen_column)
                        attack_queen_list.add(compare_column)

                    # check if on the one of diagonals
                    elif queen_column - columns[queen_column] == compare_column - columns[compare_column]:
                        attack_queen_list.add(queen_column)
                        attack_queen_list.add(compare_column)

                    # check if on the another diagonal
                    elif queen_column + columns[queen_column] == compare_column + columns[compare_column]:
                        attack_queen_list.add(queen_column)
                        attack_queen_list.add(compare_column)

            try:
                h1_self = min([weights[i] for i in attack_queen_list]) ** 2
            except:
                # there is no attacking queen pairs
                h1_self = 0

            return h1_self

        # --------------------------------------------------
        # h1 for each pontential move, displayed as a board
        h1_board = np.zeros((self.size, self.size))
        
        for i in range(self.size):
            for j in range(self.size):
                if self.columns[j] != i:
                    next_move = self.columns.copy()
                    next_move[j] = i
                    
                    if(add_cost):
                        move_cost = self.cost(j, i)
                        #print ("Cost to move queen ", i, "to row ", j, " = ", move_cost)
                        h1_board[i, j] = calculate_h1(next_move, self.weights, self.size) + move_cost
                    else:
                        h1_board[i, j] = calculate_h1(next_move, self.weights, self.size)
                    

                else:
                    # let's assume the current h1 is infinte, then we can get the min of neighbours
                    h1_board[i, j] = float("inf")

        return h1_board, calculate_h1(self.columns, self.weights, self.size)

    def h2(self, add_cost = False):

        """
            Check Heuristics 2
                    Sum across every pair of attacking Queens the weight of the lightest Queen.
            Input:
                self:

            Outputs:
                h2_board: h1 for each pontential move, displayed as a board
                h2_current: h1 for current board
        """

        def calculate_h2(columns, weights, size):

            attack_queen_pair_list = set()

            for queen_column in range(size):
                for compare_column in range(queen_column + 1, size):

                    # check if on the same row
                    if columns[queen_column] == columns[compare_column]:
                        attack_queen_pair_list.add((queen_column, compare_column))

                    # check if on the one of diagonals
                    elif queen_column - columns[queen_column] == compare_column - columns[compare_column]:
                        attack_queen_pair_list.add((queen_column, compare_column))

                    # check if on the another diagonal
                    elif queen_column + columns[queen_column] == compare_column + columns[compare_column]:
                        attack_queen_pair_list.add((queen_column, compare_column))
            try:
                h2_self = sum([min(weights[i], weights[j]) ** 2 for i, j in attack_queen_pair_list])

            except:
                # there is no attacking queen pairs
                h2_self = 0

            return h2_self

        # --------------------------------------------------
        # h2 for each pontential move, displayed as a board
        h2_board = np.zeros((self.size, self.size))

        for i in range(self.size):
            for j in range(self.size):
                if self.columns[j] != i:
                    next_move = self.columns.copy()
                    next_move[j] = i

                    if(add_cost):
                        move_cost = self.cost(j, i)
                        #print ("Cost to move queen ", i, "to row ", j, " = ", move_cost)
                        h2_board[i, j] = calculate_h2(next_move, self.weights, self.size) + move_cost
                    else:                        
                        h2_board[i, j] = calculate_h2(next_move, self.weights, self.size)

                else:
                    # let's assume the current h2 is infinte, then we can get the min of neighbours
                    h2_board[i, j] = float("inf")

        return h2_board, calculate_h2(self.columns, self.weights, self.size)

    def attacks(self):
        '''
            check if the game has ended.
            Input:
                self:

            Outputs:
                n_attack: the result (numbers of pairs of attacking Queen)
        '''

        n_attack = 0

        for queen_column in range(self.size):
            for compare_column in range(queen_column + 1, self.size):

                # check if on the same row
                if self.columns[queen_column] == self.columns[compare_column]:
                    n_attack = n_attack + 1

                # check if on the one of diagonals
                elif queen_column - self.columns[queen_column] == compare_column - self.columns[compare_column]:
                    n_attack = n_attack + 1

                # check if on the another diagonal
                elif queen_column + self.columns[queen_column] == compare_column + self.columns[compare_column]:
                    n_attack = n_attack + 1

        return n_attack

    def test(self):
        """
        check if all the functions in the class working
        """

        print("--------NQ.columns--------")
        print(self.columns)
        print("--------NQ.weight--------")
        print(self.weights)
        print("--------NQ.size--------")
        print(self.size)
        print("--------NQ.attacks--------")
        print(self.attacks())
        print("--------NQ.display--------")
        self.display()
        print("--------NQ.h1--------")
        print(self.h1())
        print("--------NQ.h2--------")
        print(self.h2())

################################################
# Hill Climbing Algorithm
################################################

class Hillclimbing:

    def __init__(self, n_queen_board, heuristic, restart_limit=5, sideway_limit=5):

        self.h = heuristic
        self.cost = 0
        self.sideway = 0
        self.node = [n_queen_board.columns.copy()]
        self.restart = 0
        self.restart_limit = restart_limit
        self.sideway_limit = sideway_limit
        print("Start Game with " + heuristic)
        print("----------------------------------------------")

    def restart_game(self, n_queen_board):

        """
        if get stuck on local optimal, restart the game
        """

        if self.restart <= self.restart_limit:

            print("Restart the Game with " + self.h)
            print("----------------------------------------------")

            # refresh sideway move quota, cost
            self.sideway = 0
            self.cost = 0

            # refresh node and board
            n_queen_board.columns = self.node[0].copy()
            self.node = [n_queen_board.columns.copy()]

            # update the number of restarts
            self.restart += 1

            # keep expanding
            return self.expand(n_queen_board)

        else:

            print("Run out of the restarts limits")

    def expand(self, n_queen_board):

        '''
         Expand the current tree node.
         Add one child node for each possible next move in the game.
         Inputs:
                node: the current tree node to be expanded
         Outputs:
                c.children: a list of children nodes.
        '''

        def simulated_annealing(current_h1, next_h1, T=10):

            if T ** (current_h1 - next_h1) > 0.5:
                return True
            else:
                return False

        def play_rule(h_board, h_self):

            print("play_rule start")
            """
            Given the heuristic for all potential moves and current state, play the next move
            """
            h_min = h_board.min()

            # Sideway occurs and we can make sideways moves within limits
            if (h_min == h_self) and (self.sideway <= self.sideway_limit):

                # get the column number of the queen and move to which row
                choice = np.where(h_board == h_min)
                rd = random.randint(1, len(choice[0]))
                queen_to_move = choice[1][rd - 1]
                move_to_where = choice[0][rd - 1]
                
                # update data the cost
                self.cost += n_queen_board.cost(queen_to_move, move_to_where)

                # move the queen
                n_queen_board.play(queen_to_move, move_to_where)

                # add the node
                self.node.append(n_queen_board.columns)

                # update the number of sideway moves
                self.sideway += 1

                # show board
                n_queen_board.display()

                # keep expanding
                return self.expand(n_queen_board)

            elif (h_min < h_self) or simulated_annealing(h_min, h_self):

                # get the column number of the queen and move to which row
                choice = np.where(h_board == h_min)
                rd = random.randint(1, len(choice[0]))
                queen_to_move = choice[1][rd - 1]
                move_to_where = choice[0][rd - 1]
                
                # update data the cost
                self.cost += n_queen_board.cost(queen_to_move, move_to_where)

                # move the queen
                n_queen_board.play(queen_to_move, move_to_where)

                # add the node
                self.node.append(n_queen_board.columns.copy())

                # show board
                n_queen_board.display()

                # keep expanding
                return self.expand(n_queen_board)

            # else here means that
            # either we run out of the sideways move
            # or we reach the local optimal but not global optimal
            else:

                # we have to restart queens' position
                return self.restart_game(n_queen_board)

        """
        start to play
        """
        # check if game is over
        if n_queen_board.attacks() == 0:
            print("Game over")

        # if game is not over, check which heuristic is used to play game

        # is heuristic 1?
        elif self.h == "h1":

            h1_board, h1_self = n_queen_board.h1()
            play_rule(h1_board, h1_self)

        # is heuristic 2?
        elif self.h == "h2":

            h2_board, h2_self = n_queen_board.h2()
            play_rule(h2_board, h2_self)


###############################################
# PriorityQueue
##############################################
class PriorityQueue:

    def __init__(self):
        self.queue = []

    def add(self, h_board, cost):

        if len(self.queue) == 0:
            self.queue.append(h_board)
        else:
            # Loop thru queue and compare costs
            for i in len(self.queue):
                if(cost <= self.queue[cost]):
                    self.queue.insert(h_board, i)
                    print("PriorityQueue inserting board at position ", i)

    def remove(self):
        return self.queue.pop(0)

    def isEmpty(self):
        return 1 if len(self.queue) == 0 else 0
        
            
################################################
# A* Algorithm
################################################

class A_Star:

    def __init__(self, n_queen_board, heuristic):

        self.h = heuristic
        self.frontier = PriorityQueue()
        self.came_from = {}
        self.cost_so_far = {}
        self.frontier.add(n_queen_board, 0)
        self.came_from[n_queen_board] = 0
        self.cost_so_far[n_queen_board] = 0
        
    def expand(self):

        while not self.frontier.isEmpty():
            current_board = self.frontier.remove()
            print("current board:")
            current_board.display()

            # Exit if solution was found
            if current_board.attacks() == 0:
                break

            for c in range(current_board.size):
                for r in range(current_board.size):

                    neighbor_board = None
                    heurisitc = None
                    
                    if self.h == "h1":
                        neighbor_board, heuristic = current_board.h1(True)                
                    elif self.h == "h2":
                        neighbor_board, heurisitic = current_board.h2(True)

                    
                    new_cost = self.cost_so_far[current_board] + neighbor_board[c][r];
                    print("neighbor_board[c][r] = ", neighbor_board[c][r], "new_cost = ", new_cost)
                    
                    #print("rows of queens = ", )

                    if neighbor_board not in self.cost_so_far:
                        self.cost_so_far[neighbor_board] = new_cost
                        self.frontier.add
                        

            print(neighbor_board)
            sys.exit(1)
            # Loop thrue board and find smallest costs


        
            # What am I adding to the frontier?            
            # - Am I adding the board + the cost 
            # - Is my new_cost = h + move_cost?
            # From the algorithm... what are my neighbors? Entire boards? Just costs of moving.
            # - I will still need to keep the board incase I backtrack
   

#####################
# Script Start
#####################

__options__ = parse_cmd_line_options()
starting_board = parse_csv_file()

#for queen in starting_board:
#    print("Queen weight = %d, Queen row = %d, Queen col = %d" % (queen[0], queen[1], queen[2]))




n_queen = N_QueenChess(starting_board)
#n_queen.display()
#n_queen.test()

#hc_h1 = Hillclimbing(n_queen_board = n_queen, heuristic = "h1")
#hc_h1.expand(n_queen)

#n_queen = N_QueenChess(starting_board)
#hc_h2 = Hillclimbing(n_queen_board = n_queen, heuristic = "h2")
#hc_h2.expand(n_queen)



#n_queen.display()

a_star = A_Star(n_queen_board = n_queen, heuristic = "h1")
a_star.expand()
