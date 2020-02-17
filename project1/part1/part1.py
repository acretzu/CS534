#! /usr/bin/python

import os
import sys
import re
from optparse import OptionParser
import math


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

        cost = self.weights[queen_to_move] * ((self.columns[queen_to_move] - move_to_where) ** 2)

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
                    h1_board[i, j] = calculate_h1(next_move, self.weights, self.size)

                else:
                    # let's assume the current h1 is infinte, then we can get the min of neighbours
                    h1_board[i, j] = float("inf")

        return h1_board, calculate_h1(self.columns, self.weights, self.size)

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
# Nodes being tracked
################################################

class Node():

    def __init__(self, columns, weights, parent=None, isleaf= False, cost=0):
        self.columns = columns
        self.weights = weights
        self.isleaf = isleaf
        self.parent = parent
        self.cost = cost
        self.children= []


#####################
# Script Start
#####################

__options__ = parse_cmd_line_options()
starting_board = parse_csv_file()
n_queen = N_QueenChess(starting_board)

for queen in starting_board:
    print("Queen weight = %d, Queen row = %d, Queen col = %d" % (queen[0], queen[1], queen[2]))

n_queen.test()