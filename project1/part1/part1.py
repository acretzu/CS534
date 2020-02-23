#! /usr/bin/python3.6

import os
import sys
import re
from optparse import OptionParser
import math
import numpy as np
import random
import time
import sys
import copy
import pandas as pd



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
    parser.add_option("--e", action="store", type="string", dest="heuristic", default="h1", help="The heuristic.")
    parser.add_option("--a", action="store", type="int", dest="algorithm", default=2, help="The algorithm.")
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
        self.start_columns = [(i[0] - 1) for i in starting_board]
        self.columns = [(i[0] - 1) for i in starting_board]
        self.weights = [i[2] for i in starting_board]
        self.size = len(self.weights)

    def set_columns_from_string( col_str ):
        print("Foo")
        #self.columns

    def update_board(self, cols):
        """
        Set board to the provided state (cols = queens)

        """
        if len(cols) != len(self.columns):
            print("Updating to a board with incorrect size!")
            sys.exit(1)

        for pos in range(len(cols)):
            self.columns[pos] = cols[pos]

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


    def calculate_h1(self, columns, weights, size):
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

    def h1(self):

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


        # --------------------------------------------------
        # h1 for each potential move, displayed as a board
        h1_board = np.zeros((self.size, self.size))

        for i in range(self.size):
            for j in range(self.size):
                if self.columns[j] != i:
                    next_move = copy.deepcopy(self.columns)
                    next_move[j] = i
                    h1_board[i, j] = self.calculate_h1(next_move, self.weights, self.size)
                else:
                    # let's assume the current h1 is infinite, then we can get the min of neighbours
                    h1_board[i, j] = float("inf")

        return h1_board, self.calculate_h1(self.columns, self.weights, self.size)

    def calculate_h2(self, columns, weights, size):

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


    def h2(self):

        """
            Check Heuristics 2
                    Sum across every pair of attacking Queens the weight of the lightest Queen.
            Input:
                self:

            Outputs:
                h2_board: h1 for each pontential move, displayed as a board
                h2_current: h1 for current board
        """

        # --------------------------------------------------
        # h2 for each pontential move, displayed as a board
        h2_board = np.zeros((self.size, self.size))

        for i in range(self.size):
            for j in range(self.size):
                if self.columns[j] != i:
                    next_move = copy.deepcopy(self.columns)
                    next_move[j] = i
                    h2_board[i, j] = self.calculate_h2(next_move, self.weights, self.size)

                else:
                    # let's assume the current h2 is infinte, then we can get the min of neighbours
                    h2_board[i, j] = float("inf")

        return h2_board, self.calculate_h2(self.columns, self.weights, self.size)

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

    def __init__(self, n_queen_board, heuristic, time_limit=0.1, sideway_limit=3):

        self.h = heuristic
        self.total_cost = 0
        self.sideway = 0
        self.node = [copy.deepcopy(n_queen_board.columns)]
        self.time_limit = time_limit
        self.sideway_limit = sideway_limit
        self.total_start_time = time.time()
        self.start_time = time.time()

        print("Start Game using Hill Climbing with " + heuristic)
        print("----------------------------------------------")

        print("Start board state")

        """
        Just for visualization for the whole process
        """
        # show board, h, total_cost
        n_queen_board.display()
        # print("-----------h1------------")
        # print(n_queen_board.h1())
        # print("-----------h2------------")
        # print(n_queen_board.h2())
        # print("-----------cost so far------------")
        # print(self.total_cost)

    def restart_game(self, n_queen_board):

        """
        if get stuck on local optimal, restart the game
        """

        # print("Restart the Game with " + self.h)
        # print("----------------------------------------------")

        self.start_time = time.time()

        # refresh sideway move quota, cost
        self.sideway = 0
        self.total_cost = 0

        # refresh node and board
        n_queen_board.columns = copy.deepcopy(self.node[0])
        self.node = [copy.deepcopy(n_queen_board.columns)]

    def expand(self, n_queen_board):

        '''
            Expand the current tree node.
            Add one child node for each possible next move in the game.
            Inputs:
                node: the current tree node to be expanded
            Outputs:
                c.children: a list of children nodes.
        '''

        def play_rule(h_board, h_self):

            """
            Given the heuristic for all potential moves and current state, play the next move
            """
            h_min = h_board.min()
            # self.minset.append(h_min)

            # Sideway occurs and we can make sideways moves within limits
            if (h_min == h_self) and (self.sideway <= self.sideway_limit) and ((time.time()-self.start_time) <= self.time_limit):

                # get the column number of the queen and move to which row
                choice = np.where(h_board == h_min)
                rd = random.randint(1, len(choice[0]))
                queen_to_move = choice[1][rd - 1]
                move_to_where = choice[0][rd - 1]

                # update data the cost
                self.total_cost += n_queen_board.cost(queen_to_move, move_to_where)

                # move the queen
                n_queen_board.play(queen_to_move, move_to_where)

                # add the node
                self.node.append(copy.deepcopy(n_queen_board.columns))

                # update the number of sideway moves
                self.sideway += 1


                """
                Just for visualization for the whole process
                """
                # show board, h, total_cost
                # n_queen_board.display()
                # print("-----------h1------------")
                # print(n_queen_board.h1())
                # print("-----------h2------------")
                # print(n_queen_board.h2())
                # print("-----------cost so far------------")
                # print(self.total_cost)

            elif (h_min < h_self) and ((time.time()-self.start_time) <= self.time_limit):

                # get the column number of the queen and move to which row
                choice = np.where(h_board == h_min)
                rd = random.randint(1, len(choice[0]))
                queen_to_move = choice[1][rd - 1]
                move_to_where = choice[0][rd - 1]

                # update data the cost
                self.total_cost += n_queen_board.cost(queen_to_move, move_to_where)

                # move the queen
                n_queen_board.play(queen_to_move, move_to_where)

                # add the node
                self.node.append(copy.deepcopy(n_queen_board.columns))

                """
                Just for visualization for the whole process
                """
                # show board, h, cost
                # n_queen_board.display()
                # print("-----------h1------------")
                # print(n_queen_board.h1())
                # print("-----------h2------------")
                # print(n_queen_board.h2())
                # print("-----------cost so far------------")
                # print(self.total_cost)

                # keep expanding
                # return self.expand(n_queen_board)

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
        while n_queen_board.attacks() != 0:

            if (time.time() - self.total_start_time) > (10*n_queen_board.size):
                break

            # if game is not over, check which heuristic is used to play game

            # is heuristic 1?
            if self.h == "h1":

                h1_board, h1_self = n_queen_board.h1()
                play_rule(h1_board, h1_self)

            # is heuristic 2?
            elif self.h == "h2":

                h2_board, h2_self = n_queen_board.h2()
                play_rule(h2_board, h2_self)

    def display_result(self, n_queen_board):

        print("Heuristic        =", self.h)
        print("Total time (s)   =", (time.time() - self.total_start_time))
        print("Total cost       =", self.total_cost)
        print("Nodes expanded   =", len(self.node)-1)
        print("Moves to solve   =", len(self.node)-1)
        print("Branching factor =", 1)
        for i in self.node:
            for column in range(n_queen_board.size):
                for row in range(n_queen_board.size):
                    if column == i[row]:
                        print(n_queen_board.weights[row], end=' ')
                    else:
                        print('.', end=' ')
                print()
            print("---------------------------------")

        return [n_queen.size, (time.time() - self.total_start_time), self.total_cost, len(self.node)-1, len(self.node)-1, 1, n_queen_board.attacks()]



###############################################
# PriorityQueue
##############################################
class PriorityQueue:

    def __init__(self):
        self.queue = []

    def add(self, cost, state):
        '''
           Adds a tuple (cost, state) to the list where lower costs are at the front of the queue.

           Inputs:
              Cost - Integer cost of the state
              State - Space-separated string representing the new state of queens
           Outputs:
              None
        '''
        # Append tuple if list is empty
        if len(self.queue) == 0:
            self.queue.append( (cost, state) )
            # Debug
            #print("PriorityQueue is empty. Adding at the end")
        else:
            # Loop thru queue and compare costs
            i = 0
            for pair in self.queue:
                qcost = pair[0]
                if cost < qcost:
                    self.queue.insert(i, (cost, state))
                    # Debug
                    #print("PriorityQueue inserting at position ", i, ", cost = ", cost, ", qcost = ", qcost, ", state = ", state)
                    break
                elif i == len(self.queue)-1:
                    self.queue.append( (cost, state) )
                    # Debug
                    #print("PriorityQueue inserting at the end")
                    break
                i += 1

    def remove(self):
        '''
           Removes tuple from the front of the queue

           Inputs:
              None
           Outputs:
              Tuple (cost, state)
        '''
        return self.queue.pop(0)

    def isEmpty(self):
        '''
           Returns 1 if queue is empty, else 0

           Inputs:
              None
           Outputs:
              1 if queue is empty, else 0
        '''
        return 1 if len(self.queue) == 0 else 0

    def contains(self, s):
        '''
           Returns 1 space-seperated string of queens exists in the queue

           Inputs:
              Space-separated string of queens
           Outputs:
              1 if input string exists in the queue, else 0
        '''
        for pair in self.queue:
            if pair[1] == s:
                return 1
        return 0

    def get_cost(self, i):
        '''
           Returns the cost of the tuple at queue position i.

           Inputs:
              None
           Outputs:
              The cost of the tuple at queue position i.
        '''
        return self.queue[i][0]

    def get_state(self, i):
        '''
           Returns the state of the tuple at queue position i.

           Inputs:
              None
           Outputs:
              The state of tuple at queue position i.
        '''
        return self.queue[i][1]

    def print(self):
        '''
           Prints the priority queue. Mainly used for debug

           Inputs:
              None
           Outputs:
              None
        '''
        print("Printing Frontier:")
        for pair in self.queue:
            print("Cost = ", pair[0], ", State = ", pair[1])

################################################
# A* Algorithm
################################################

class A_Star:

    def __init__(self, n_queen_board, heuristic):

        self.board = n_queen_board
        self.h = heuristic
        self.frontier = PriorityQueue()
        self.came_from = {}
        self.cost_so_far = {}
        self.goal_state = ""
        self.frontier.add(0, self.list_2_str(n_queen_board.columns))
        self.came_from[self.list_2_str(n_queen_board.columns)] = 0
        self.cost_so_far[self.list_2_str(n_queen_board.columns)] = 0
        self.start_state = self.frontier.get_state(0)
        self.start_time = time.perf_counter()
        self.stop_time = 0
        self.nodes_expanded = 0
        self.solution_length = 0
        self.total_cost = 0

    def list_2_str(self, l):
        '''
           Converts a list of integers into a space-separated string.

           Inputs:
              List of integers.
           Outputs:
              Space-separated string.
        '''
        return "".join(str(x)+" " for x in l)

    def str_2_list(self, s):
        '''
           Converts a space-separated string into a list of integers.

           Inputs:
              Space-separated string.
           Outputs:
              List of integers.
        '''
        return [int(i) for i in s.split()]


    def find_attacking_queens(self, queen_rows):
        '''
            check if the game has ended.
            Input:
                self:

            Outputs:
                n_attack: the result (numbers of pairs of attacking Queen)
        '''

        n_attack = 0
        size = len(queen_rows)

        for queen_column in range(size):
            for compare_column in range(queen_column + 1, size):

                # check if on the same row
                if queen_rows[queen_column] == queen_rows[compare_column]:
                    n_attack = n_attack + 1

                # check if on the one of diagonals
                elif queen_column - queen_rows[queen_column] == compare_column - queen_rows[compare_column]:
                    n_attack = n_attack + 1

                # check if on the another diagonal
                elif queen_column + queen_rows[queen_column] == compare_column + queen_rows[compare_column]:
                    n_attack = n_attack + 1

        return n_attack

    def get_h_cost(self, state):
        '''
           Utility function which uses the n_queens_board to calculate the heuristic given the state (columns) of queens

           Inputs:
              List of queens per column. Position 0 = left-most queeen.
           Outputs:
              Integer heuristic cost.
        '''
        cost = 0
        if self.h == "h1":
            cost = self.board.calculate_h1(state, self.board.weights, self.board.size)
        else:
            cost = self.board.calculate_h2(state, self.board.weights, self.board.size)
        return cost


    def print_path_to_goal(self):
        '''
           Prints the move from the starting board to the ending (goal) board

           Inputs:
              None
           Outputs:
              None
        '''
        current = self.goal_state
        path = []
        while current != self.start_state:
            path.append(current)
            current = self.came_from[current]
        path.append(self.start_state)
        path.reverse()

        print("A* algorithm path to goal:")
        for state in path:
            self.solution_length += 1
            self.board.update_board(self.str_2_list(state))
            self.board.display()


    def results(self):
        '''
           Prints the result of A*. Must only be called after execution of expand()

           Inputs:
              None
           Outputs:
              None
        '''
        self.print_path_to_goal()
        print("Heuristic        =", self.h)
        print("Total time (s)   =", round(self.stop_time - self.start_time, 3))
        print("Total cost       =", self.total_cost)
        print("Nodes expanded   =", self.nodes_expanded)
        print("Moves to solve   =", (self.solution_length - 1))
        print("Branching factor =", round(self.nodes_expanded / self.solution_length, 2))

    def expand(self):
        '''
           Executes the A* algorithm.

           Inputs:
              None
           Outputs:
              None
        '''
        while not self.frontier.isEmpty():
            # Debug
            #self.frontier.print()

            # Remove head of frontier
            current_state_str = self.frontier.remove()[1]
            current_state_list = self.str_2_list(current_state_str)

            # Debug
            #print("Current State = ", current_state_str)

            # Prints to prove H2 is not admissible
            #current_hx = self.get_h_cost(current_state_list)
            #print("current_hx = ", current_hx)

            self.board.update_board(current_state_list)
            # Debug
            #self.board.display()

            # Debug
            #print("Attacking = ", self.find_attacking_queens(current_state_list))
            # Exit if solution was found
            if self.find_attacking_queens(current_state_list) == 0:
                self.total_cost = self.cost_so_far[current_state_str]
                self.goal_state = current_state_str
                self.stop_time = time.perf_counter()
                break


            # Get neighboring states (i.e. places where we can move)
            self.nodes_expanded += 1
            for c in range(self.board.size):
                for r in range(self.board.size):
                    queen_in_col = current_state_list[c]

                    # If the queen is not in the current state
                    if queen_in_col != r:
                        # Create neighbor from current state, but change the row
                        neighbor_state_list = current_state_list.copy()
                        neighbor_state_list[c] = r
                        neighbor_state_str = self.list_2_str(neighbor_state_list)

                        # Calculate A* parameters
                        next_gx = self.board.cost(c,r)
                        next_hx = self.get_h_cost(neighbor_state_list)
                        next_fx = next_gx + next_hx
                        new_cost = self.cost_so_far[current_state_str] + next_fx

                        # Debug
                        #print("ns = ", neighbor_state_list, ", str = ", neighbor_state_str, "gx = ", next_gx, ", hx = ", next_hx, ", new_cost = ", new_cost)

                        # Update frontier
                        if neighbor_state_str not in self.cost_so_far or new_cost < self.cost_so_far[neighbor_state_str]:
                            self.cost_so_far[neighbor_state_str] = new_cost
                            self.frontier.add(new_cost, neighbor_state_str)
                            self.came_from[neighbor_state_str] = current_state_str


#####################
# Script Start
#####################

__options__ = parse_cmd_line_options()
starting_board = parse_csv_file()


n_queen = N_QueenChess(starting_board)
# n_queen.display()
# n_queen.test()
# #

if __options__.algorithm == 1:
    a_star = A_Star(n_queen, heuristic = __options__.heuristic)
    a_star.expand()
    a_star.results()
else:
    # hc = Hillclimbing(n_queen_board = n_queen, heuristic = __options__.heuristic)
    # hc.expand(n_queen)
    # hc.display_result(n_queen)


    # '''
    # below for analysis
    # '''
    # result = []
    #
    # for s in range(6):
    #     for i in range(5):
    #         for sd in range(4):
    #             starting_board = makeup_board(s+4)
    #             n_queen = N_QueenChess(starting_board)
    #             hc = Hillclimbing(n_queen_board=n_queen, heuristic=__options__.heuristic, time_limit= 1, sideway_limit=sd)
    #             hc.expand(n_queen)
    #
    #             result.append(hc.display_result(n_queen).append(s))
    #
    # result_pd = pd.DataFrame(result,
    #                          columns=['Board Size', 'Total time (s)', '"Total cost', 'Nodes expanded', 'Moves to solve', 'Branching factor', 'n_attacks', 'sideway_limits'])
    #
    # result_pd.to_csv("result_hc_h1.csv")
