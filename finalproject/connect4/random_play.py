import random

class RandomPlayer:

    def __init__(self, potential_move):
        self.potential_move = potential_move

    def choose_col(self):
        return random.choice(self.potential_move)