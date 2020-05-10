#! /usr/bin/python3.6

import sys
import pandas as pd
import numpy as np
import time

from random_play import RandomPlayer
from qlearning import QLearner
from monte_carlo import MonteCarlo
from nn import NN_Player


class Connect4:

    def __init__(self, player1, player2):
        self.board = [[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]
        self.player1 = player1
        self.player2 = player2
        self.turn = 1  # Player 1
        self.width = len(self.board)
        self.height = len(self.board[0])

    def has_winner(self):
        """
        Determine if there is a winner (4 in a row, col, diag)
        Returns 0 if no winner, 1 if player 1 wins, -1 if player 2 wins
        """
        width = len(self.board)
        height = len(self.board[0])

        # Source: https://stackoverflow.com/questions/29949169/python-connect-4-check-win-function
        # check horizontal spaces
        for y in range(height):
            for x in range(width - 3):
                if self.board[x][y] == 1 and self.board[x + 1][y] == 1 and self.board[x + 2][y] == 1 and \
                        self.board[x + 3][y] == 1:
                    return 1
                if self.board[x][y] == -1 and self.board[x + 1][y] == -1 and self.board[x + 2][y] == -1 and \
                        self.board[x + 3][y] == -1:
                    return -1

        # check vertical spaces
        for x in range(width):
            for y in range(height - 3):
                if self.board[x][y] == 1 and self.board[x][y + 1] == 1 and self.board[x][y + 2] == 1 and self.board[x][
                    y + 3] == 1:
                    return 1
                if self.board[x][y] == -1 and self.board[x][y + 1] == -1 and self.board[x][y + 2] == -1 and \
                        self.board[x][y + 3] == -1:
                    return -1

        # check / diagonal spaces
        for x in range(width - 3):
            for y in range(3, height):
                if self.board[x][y] == 1 and self.board[x + 1][y - 1] == 1 and self.board[x + 2][y - 2] == 1 and \
                        self.board[x + 3][y - 3] == 1:
                    return 1
                if self.board[x][y] == -1 and self.board[x + 1][y - 1] == -1 and self.board[x + 2][y - 2] == -1 and \
                        self.board[x + 3][y - 3] == -1:
                    return -1

        # check \ diagonal spaces
        for x in range(width - 3):
            for y in range(height - 3):
                if self.board[x][y] == 1 and self.board[x + 1][y + 1] == 1 and self.board[x + 2][y + 2] == 1 and \
                        self.board[x + 3][y + 3] == 1:
                    return 1
                if self.board[x][y] == -1 and self.board[x + 1][y + 1] == -1 and self.board[x + 2][y + 2] == -1 and \
                        self.board[x + 3][y + 3] == -1:
                    return -1

        return 0

    def clear_board(self):
        """
        Sets the board to all 0s
        """
        self.board = [[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]

    def can_place(self, column):
        """
        Returns true if the board is NOT full at given column
        """
        ret_val = True

        if self.board[0][column] == 1 or self.board[0][column] == -1:
            ret_val = False

        return ret_val

    def available_columns(self):

        """
        Returns the columns which still can be placed
        """

        available_col = []

        for i in range(self.height):
            if self.can_place(i):
                available_col.append(i)

        return available_col

    def full(self):
        """
        Returns true if the board is full
        """
        ret_val = True

        for i in range(len(self.board)):
            if self.board[0][i] == 0:
                ret_val = False

        return ret_val

    def place(self, column):
        """
        Place a piece at the given column
        """
        # Place a piece
        print("Player", self.turn, "placed piece at column:", column + 1)

        # Sanity check that column is not full
        if self.board[0][column] == 1 or self.board[0][column] == -1:
            print("Board is full at that column! Col =", column)
            return

        # Add player to bottom-most row
        for h in reversed(range(6)):
            if self.board[h][column] is 0:
                self.board[h][column] = self.turn
                break

        # Change turn
        if self.turn == 1:
            self.turn = -1
        else:
            self.turn = 1

        # Debug
        print(self)

    def __str__(self):
        """
        String representation of connect4 board
        """
        ret_str = ""

        for r in range(len(self.board)):
            for c in range(len(self.board[0])):
                ret_str += self.int2str(self.board[r][c]) + " "
            ret_str += "\n"

        return ret_str

    def int2str(self, x):
        """
        Converts 1 (player 1) to X.
        Converts -1 (player 2) to O.
        Converts 0 (blanks) to "-".
        """
        ret_val = "-"

        if x == 1:
            ret_val = "X"
        elif x == -1:
            ret_val = "O"

        return ret_val

    def target(self):

        """
        # return game results
        #  1 -> Player 1 wins
        # -1 -> Player 2 wins
        #  0 -> draw
        """

        if self.full():
            t = 0
        else:
            t = self.has_winner()
        return t

    def play(self, games=1, is_savedata=False, save_filename=str(int(time.time()))):
        """
        Main game loop. Plays full game iterations.
        """

        p1 = None
        p2 = None

        traindata_feature = []
        traindata_target = []

        iter_n = games


        while games > 0:
            print("Play iteration = ", games)

            while self.has_winner() == 0:

                if self.full():
                    print("It's a draw!")
                    break

                if self.turn == 1:

                    # Which Strategy for Palyer 1
                    if self.player1 == "Random":
                        # update
                        p1 = RandomPlayer(self.available_columns())
                        # place
                        self.place(p1.choose_col())

                    elif self.player1 == "QL":
                        p1 = QLearner(1)
                        self.place(p1.random_action())

                    elif self.player1 == "MonteCarlo":
                        p1 = MonteCarlo(1, self)

                    elif self.player1 == "NN":
                        # update
                        p1 = NN_Player(1, self.board, self.available_columns())
                        # place
                        self.place(p1.choose_col())

                else:

                    # Which Strategy for Palyer 2
                    if self.player1 == "Random":
                        # update
                        p2 = RandomPlayer(self.available_columns())
                        # place
                        self.place(p2.choose_col())

                    elif self.player2 == "QL":
                        p2 = QLearner(-1)
                        self.place(p2.random_action())


                    elif self.player2 == "MonteCarlo":
                        p2 = MonteCarlo(-1, self)
                        self.place(p2.choose_col())


                    elif self.player2 == "NN":
                        # update
                        p2 = NN_Player(-1, self.board, self.available_columns())
                        # place
                        self.place(p2.choose_col())


            # add training data for NN
            if is_savedata:
                "Save data for DNN"
                traindata_feature.append(np.array(self.board).reshape(42))
                traindata_target.append(self.target())

            print("The winner is player ", self.has_winner())
            self.clear_board()
            games -= 1

        # save training data for NN
        if is_savedata:
            np.savetxt('TrainingData/features_' + str(iter_n) + '_' + save_filename + '.csv',
                       traindata_feature, delimiter=',', fmt='%10.0f')
            np.savetxt('TrainingData/targets_' + str(iter_n) + '_' + save_filename + '.csv',
                       traindata_target, delimiter=',', fmt='%10.0f')


############
# Main Start
############


"""
========== Algorithm Competition ==========
"""

""" 1) Random VS NN  """
connect4 = Connect4("NN", "Random")
# connect4 = Connect4("Random", "NN")
connect4.play(games=1000, is_savedata=True, save_filename = "NNVSRandom")


""" 2) MonteCarlo VS NN """
# connect4 = Connect4("NN", "MonteCarlo")
# connect4.play(games=1, is_savedata=False)


""" 3) QL VS NN  """




""" 4) QL VS MonteCarlo"""




""" 5) Random VS MonteCarlo """




""" 6) Random VS QL """





"""
========== Self Game ==========
"""

""" Random VS Random """
# connect4 = Connect4("Random", "Random")
# connect4.play(games=1, is_savedata=False)


""" NN VS Nn """
# connect4 = Connect4("NN", "NN")
# connect4.play(games=1, is_savedata=False)


""" QL VS QL """



""" MonteCarlo VS MonteCarlo """



############
# NOTE: has_winner() is not working correctly
############



