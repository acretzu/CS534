
from keras.models import Sequential, load_model

from copy import deepcopy

import numpy as np

# model_path_set = 'model_1589096133.h5'
# model_path_set = 'TrainingData/10000_Random_Random_eachstep1589140617.h5'
# model_path_set = 'TrainingData/10000_Random_Random_eachstep1589140617_100ep.h5'

# six neural networks
# "100000_RR_32x2_32x2_32_32.h5"
# "100000_RR_32x2_32x2_32x2_32_32_32.h5"
# "100000_RR_64x2_64.h5"  (best one)
# "100000_RR_64x2_64x2_64_64.h5"
# "100000_RR_128x4_32x2_64_32.h5"
# "100000_RR_256x4_64x2_32x2_64_32"

# "100000_RR_256x4_64x2_32x2_64_32_dropout"

class NN_Player:

    def __init__(self, player, c4, model_path='Model/100000_RR_64x2_64.h5'):
        self.c4 = c4
        self.player = player
        self.model = load_model(model_path)
        # self.board = deepcopy(board)
        # self.potential_move = deepcopy(potential_move)

    # def update(self, board, potential_move):
    #     self.board = deepcopy(board)
    #     self.potential_move = deepcopy(potential_move)

    def choose_col(self):

        # place one and get potential board

        potential_board_list = []

        for pm in self.c4.available_columns():
            # potential_board = deepcopy(self.board)
            potential_board = deepcopy(self.c4.board)
            for h in reversed(range(6)):
                if potential_board[h][pm] is 0:
                    potential_board[h][pm] = self.player
                    break
            potential_board_list.append(potential_board)
        potential_board_list_np = np.array(potential_board_list)

        # reshape for model
        potential_board_list_np = potential_board_list_np.reshape(len(self.c4.available_columns()), 6, 7, 1)

        # run NN on the board and get the score
        potential_score_list = self.model.predict(potential_board_list_np).flatten()

        if self.player == -1:
            bestmove_index = np.argmax(potential_score_list*(-1))
            bestmove = self.c4.available_columns()[bestmove_index]

        else:
            bestmove_index = np.argmax(potential_score_list )
            bestmove = self.c4.available_columns()[bestmove_index]

        return bestmove
