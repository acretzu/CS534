#! /usr/bin/python3.6
from termcolor import colored
import sys
import pandas as pd
import numpy as np
import time

from pip._vendor.distlib.compat import raw_input
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
        self.draw = False
        self.prev_board = None
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

        for x in range(width):
            if self.board[x][0] == 1 or self.board[x][0] == 2:
                self.draw = True
            else:
                self.draw = False
                break

        if self.draw is True:
            return 3

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

        # Sanity check that column is not full
        if self.board[0][column] == 1 or self.board[0][column] == -1:
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

    def place_with_print(self, column):
        """
        Place a piece at the given column
        """
        # Place a piece
        # print("Player", self.turn, "placed piece at column:", column + 1)

        # Sanity check that column is not full
        if self.board[0][column] == 1 or self.board[0][column] == -1:
            print("Board is full at that column! Col =", column)
            return

        # Save current board as previous
        self.prev_board = [row[:] for row in self.board]

        foo = ""
        for r in range(len(self.board)):
            for c in range(len(self.board[0])):
                foo += self.int2str(self.prev_board[r][c]) + " "
            foo += "\n"
        #print("Prev_board:")
        #print(foo)

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
        Converts 1 (player 1) to green O.
        Converts -1 (player 2) to blue O.
        Converts 0 (blanks) to "-".
        """
        ret_val = "-"

        if x == 1:
            ret_val = colored("O", 'red')
        elif x == -1:
            ret_val = colored("O", 'yellow')

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

    def get_state(self):
        """
        Return the 2d list numerical representation of the board
        """
        result = tuple(tuple(x) for x in self.board)

        return result

    def get_prev_state(self):
        """
        Return the previous state of the board
        """
        result = tuple(tuple(x) for x in self.prev_board)

        return result

    def play(self, games=1, traindata_flag=False, saveresults_flag=True, save_filename=str(int(time.time()))):
        """
        Main game loop. Plays full game iterations.
        """

        p1 = None
        p2 = None

        traindata_feature = []
        traindata_target = []

        iter_n = games

        # Select player1 outside of game loop
        if self.player1 == "Random":
            p1 = RandomPlayer(self)
        elif self.player1 == "QL":
            p1 = QLearner(1, self)
        elif self.player1 == "MonteCarlo":
            p1 = MonteCarlo(1, self)
        elif self.player1 == "NN":
            p1 = NN_Player(1, self)
            # p1 = NN_Player(1, self.board, self.available_columns())

        # Select player1 outside of game loop
        if self.player2 == "Random":
            p2 = RandomPlayer(self)
        elif self.player2 == "QL":
            p2 = QLearner(-1, self)
        elif self.player2 == "MonteCarlo":
            p2 = MonteCarlo(-1, self)
        elif self.player2 == "NN":
            p2 = NN_Player(-1, self)
            # p2 = NN_Player(-1, self.board, self.available_columns())

        # record the total time each player uses in each game

        total_time_player1 = []
        total_time_player2 = []

        while games > 0:
            print("Play iteration = ", games)

            # record total move in each game
            total_move = 0

            # record the total time each player uses in each game
            player1_time = 0
            player2_time = 0

            while self.has_winner() == 0:

                # print(self.board)

                if self.full():

                    print("It's a draw!")
                    print("Player 1 uses: ", player1_time, "s")
                    print("player 2 uses: ", player2_time, "s")

                    break

                if self.turn == 1:

                    start_time = time.time()

                    # Which Strategy for Palyer 1
                    if self.player1 == "Random":
                        self.place(p1.choose_col())

                    elif self.player1 == "QL":
                        # p1 = QLearner(1)
                        p1.learn()

                    elif self.player1 == "MonteCarlo":
                        self.place(p1.choose_col())

                    elif self.player1 == "NN":
                        self.place(p1.choose_col())

                    if self.player2 == "QL":
                        p2.check_if_lost()

                    end_time = time.time()

                    player1_time = player1_time + (end_time - start_time)

                else:

                    start_time = time.time()

                    # Which Strategy for Palyer 2
                    if self.player2 == "Random":
                        self.place(p2.choose_col())

                    elif self.player2 == "QL":
                        p2.learn()

                    elif self.player2 == "MonteCarlo":
                        p2 = MonteCarlo(-1, self)
                        self.place(p2.choose_col())

                    elif self.player2 == "NN":
                        self.place(p2.choose_col())

                    if self.player1 == "QL":
                        p1.check_if_lost()

                    end_time = time.time()

                    player2_time = player2_time + (end_time - start_time)

                if traindata_flag:
                    # add features for training data for NN
                    traindata_feature.append(np.array(self.board).reshape(42))

                total_move = total_move + 1

            # add targets for training data for NN
            if traindata_flag:
                for m in range(total_move):
                    traindata_target.append(self.target())

            # complete results
            if saveresults_flag:
                traindata_target.append(self.target())
                total_time_player1.append(player1_time)
                total_time_player2.append(player2_time)

            print("The winner is player ", self.has_winner())
            print("Player 1 uses: ", player1_time, "s")
            print("player 2 uses: ", player2_time, "s")

            self.clear_board()
            games -= 1

        # save training data for NN
        if traindata_flag:
            np.savetxt('TrainingData/features_' + str(iter_n) + '_' + save_filename + '.csv',
                       traindata_feature, delimiter=',', fmt='%10.0f')
            np.savetxt('TrainingData/targets_' + str(iter_n) + '_' + save_filename + '.csv',
                       traindata_target, delimiter=',', fmt='%10.0f')

        # save game results for comparison
        if saveresults_flag:
            results = np.array([traindata_target, total_time_player1, total_time_player2]).T
            fmt = '%10.0f', '%10.10f', '%10.10f'
            np.savetxt(
                'Game_results/' + self.player1 + '_' + self.player2 + '_' + str(iter_n) + '_' + save_filename + '.csv',
                results, delimiter=',', fmt=fmt, header="win,Player1_time, Player2_time")

        print("-------------Among all the games----------------- ")
        print("Player 1 is ", self.player1)
        print("Player 2 is ", self.player2)
        print("Total games Player 1 wins: ", sum([i for i in traindata_target if i == 1]))
        print("Total games Player 2 wins: ", sum([i for i in traindata_target if i == -1])*(-1))
        print("Total Time Player 1 takes: ", sum(total_time_player1), "s")
        print("Total Time Player 2 takes: ", sum(total_time_player2), "s")

    def play_human(self, player):
        """
        Main game loop. Waits for human input.
        """
        opp = QLearner(1, self)

        while self.has_winner() == 0:

            # opp = MonteCarlo(1, self, depth=100, rollouts=1000)

            if self.full():
                print("It's a draw!")
                return

            if self.turn is player:
                human_move = int(raw_input(">>> "))
                self.place_with_print(human_move - 1)
                opp.check_if_lost()
            else:
                # opp_move = opp.choose_col()
                opp.learn()
                # while self.can_place(opp_move) is False:
                #    print(opp_move)
                #    opp_move = opp.choose_col()
                # opp.print(opp.root)
                # self.place_with_print(opp_move)

        if self.has_winner() is player:
            print("You won!")
        else:
            print("The winner is Bot!")
        self.clear_board()


############
# Main Start
############


"""
========== Algorithm Competition ==========
"""

""" 1) Random VS NN  """
# connect4 = Connect4("NN", "Random")
# connect4.play(100)
# connect4 = Connect4("Random", "NN")
# connect4.play(100)

""" 2) MonteCarlo VS NN """
connect4 = Connect4("NN", "MonteCarlo")
connect4.play(10)
# connect4 = Connect4("MonteCarlo","NN")
# connect4.play(games=10)


""" 3) QL VS NN  """
# connect4 = Connect4("NN", "QL")
# connect4.play(1)
# connect4 = Connect4("QL", "NN")
# connect4.play(100)
# connect4.play_human(-1);


""" 4) QL VS MonteCarlo"""
# connect4 = Connect4("MonteCarlo", "QL")
# connect4.play(10)
# connect4 = Connect4("QL", "MonteCarlo")
# connect4.play(100)

""" 5) Random VS MonteCarlo """
# connect4 = Connect4("Human", "QL")
#connect4.play(games=100)
# while True:
#     print(connect4.__str__())
#     connect4.play_human(-1)


# connect4 = Connect4("Random", "MonteCarlo")
# connect4.play(games=10)
# connect4 = Connect4("MonteCarlo", "Random")
# connect4.play(games=10)
# print(connect4.__str__())
# connect4.play_human(-1)


""" 6) Random VS QL """
# connect4 = Connect4("Random", "QL")
# connect4.play(100)
# connect4 = Connect4("QL", "Random")
# connect4.play(100)
# connect4.play_human(-1)

"""
========== Self Game ==========
"""

""" Random VS Random """
# connect4 = Connect4("Random", "Random")
# connect4.play(games=100000, traindata_flag=True, saveresults_flag=False,
#               save_filename="Random_Random_eachstep" + str(int(time.time()))
#               )

""" NN VS NN """
# connect4 = Connect4("NN", "NN")
# connect4.play(games=10)


""" QL VS QL """

""" MonteCarlo VS MonteCarlo """
# connect4 = Connect4("MonteCarlo", "MonteCarlo")
# connect4.play(games=10)
