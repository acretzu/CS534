from random import randrange

from .connect4 import Connect4


class Node:
    def __init__(self):
        self.children = []
        self.wins = 0
        self.count = 0


class Connect4Tree:

    def __init__(self, player):
        self.root = Node()
        self.c4 = Connect4()
        self.player = player

    def select(self):
        root = self.root
        while root.children is not []:

            # Find Best Child
            best = Node()
            for child in root.children:

                if child.count is 0:
                    child_ratio = 0
                else:
                    child_ratio = child.wins / child.count

                if best.count is 0:
                    best_ratio = 0
                else:
                    best_ratio = best.wins / best.count

                if child_ratio >= best_ratio:
                    best = child

            root = best

        return root

    def expand(self, root):
        for i in range(7):
            # Expand
            root.children.append(Node())

    def simulate(self, root):
        # Returns 0 if loss, 1 if won
        simulated_child = 0
        while self.c4.has_winner() is 0:

            col = randrange(7)
            while self.c4.can_place(col) is False:
                col = randrange(7)

            self.c4.place(col)

        if self.c4.has_winner() is self.player:
            return 1
        else:
            return 0


c4 = Connect4()

