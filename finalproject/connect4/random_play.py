import random

class RandomPlayer:

    def __init__(self, c4):
        self.c4 = c4

    def choose_col(self):
        return random.choice(self.c4.available_columns())