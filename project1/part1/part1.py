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
class N_QueenChess():

    def __init__(self, starting_board):

        starting_board.sort(key=lambda x: x[1])

        self.columns = [(i[0]-1) for i in starting_board]
        self.weights = [i[2] for i in starting_board]

        self.size = len(self.weights)

    #     def start_game(self):
    #         self.columns = [random.randint(0, (self.size - 1)) for i in range(self.size)]
    #         self.weights = [random.randint(1, 9) for i in range(self.size)]

    def play(self, queen_to_move, move_to_where):

        """
            Play next move

            Input:
                queen_to_move: the column number (which queen to move)
                move_to_where: move to which row

        """
        self.columns[queen_to_move] = move_to_where

    def cost(self, queen_to_move, move_to_where):

        """
            Calculate the cost of one move

            Input:
                queen_to_move: the column number (which queen to move)
                move_to_where: move to which row

            Output:
                cost of this move

        """

        cost = self.weights[queen_to_move] * (self.columns[queen_to_move] - move_to_where)

        return cost

    # acretzu - Commenting out due to errors. Please fix!
    def display(self):
        for column in range(self.size):
            for row in range(self.size):
                if column == self.columns[row]:
                    print(self.weights[row], end=' ')
                else:
                    print('.', end=' ')
            print()

    def h1(self):

        """
            check heuristics 1
            Input:
                s: the current state of the game

            Outputs:
                h1: The lightest Queen across all pairs of Queens attacking each other.
                    Moving that Queen is the minimum possible cost to solve the problem
        """

        attack_queen_list = set()

        for queen_column in range(self.size):
            for compare_column in range(queen_column + 1, self.size):
                if self.columns[queen_column] == self.columns[compare_column]:
                    attack_queen_list.add(queen_column)
                    attack_queen_list.add(compare_column)

                elif queen_column - self.columns[queen_column] == compare_column - self.columns[compare_column]:
                    attack_queen_list.add(queen_column)
                    attack_queen_list.add(compare_column)

                elif self.columns[queen_column] - self.columns[compare_column] == compare_column - queen_column:
                    attack_queen_list.add(queen_column)
                    attack_queen_list.add(compare_column)

        h1 = min([self.weights[i] for i in attack_queen_list]) ** 2

        return h1

    def h2(self):

        """
            check heuristics 2
            Input:
                s: the current state of the game

            Outputs:
                h2: Sum across every pair of attacking Queens the weight of the lightest Queen.
        """

        attack_queen_pair_list = set()

        for queen_column in range(self.size):
            for compare_column in range(queen_column + 1, self.size):
                if self.columns[queen_column] == self.columns[compare_column]:
                    attack_queen_pair_list.add((queen_column, compare_column))

                elif queen_column - self.columns[queen_column] == compare_column - self.columns[compare_column]:
                    attack_queen_pair_list.add((queen_column, compare_column))

                elif self.columns[queen_column] - self.columns[compare_column] == compare_column - queen_column:
                    attack_queen_pair_list.add((queen_column, compare_column))

        h2 = sum([min(self.weights[i], self.weights[j]) ** 2 for i, j in attack_queen_pair_list])

        return h2

    def attacks(self):
        '''
            check if the game has ended.
            Input:
                s: the current state of the game

            Outputs:
                n_attack: the result (numbers of pairs of attacking Queen)

        '''
        n_attack = 0

        for queen_column in range(self.size):
            for compare_column in range(queen_column + 1, self.size):
                if self.columns[queen_column] == self.columns[compare_column]:
                    n_attack = n_attack + 1

                elif queen_column - self.columns[queen_column] == compare_column - self.columns[compare_column]:
                    n_attack = n_attack + 1

                elif self.columns[queen_column] - self.columns[compare_column] == compare_column - queen_column:
                    n_attack = n_attack + 1

        return n_attack


    def test(self):
        print("--------columns--------")
        print(self.columns)
        print("--------weight--------")
        print(self.weights)
        print("--------size--------")
        print(self.size)
        print("--------display--------")
        self.display()
        print("--------h1--------")
        print(self.h1())
        print("--------h2--------")
        print(self.h2())
        print("--------attacks--------")
        print(self.attacks())

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