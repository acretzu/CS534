import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential, load_model

from copy import deepcopy

import numpy as np

model_path = 'model_1589096133.h5'


class NN_Player:

    def __init__(self, player, board, potential_move, model_path='model_1589096133.h5'):
        self.player = player
        self.model = load_model(model_path)
        self.board = deepcopy(board)
        self.potential_move = deepcopy(potential_move)

    def choose_col(self):

        # place one and get potential board

        potential_board_list = []

        for pm in self.potential_move:
            potential_board = deepcopy(self.board)
            for h in reversed(range(6)):
                if potential_board[h][pm] is 0:
                    potential_board[h][pm] = self.player
                    break
            potential_board_list.append(potential_board)
        potential_board_list_np = np.array(potential_board_list)

        # reshape for model
        potential_board_list_np = potential_board_list_np.reshape(len(self.potential_move), 6, 7, 1)

        # run NN on the board and get the score
        potential_score_list = self.model.predict(potential_board_list_np).flatten()

        if self.player == -1:
            bestmove_index = np.argmax(potential_score_list*(-1))
            bestmove = self.potential_move[bestmove_index]

        else:
            bestmove_index = np.argmax(potential_score_list )
            bestmove = self.potential_move[bestmove_index]

        return bestmove
