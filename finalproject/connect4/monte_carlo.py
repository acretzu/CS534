from copy import deepcopy
from random import randrange

#from connect4 import Connect4


class Node:
    def __init__(self, parent, moves):
        self.parent = parent
        self.children = []
        self.wins = 0
        self.count = 0
        self.moves = moves


class MonteCarlo:

    def __init__(self, player, c4):
        self.root = Node(None, [])
        self.c4 = c4
        self.player = player

    def win_ratio(self, root):
        if root.count is 0:
            win_ratio = 0
        else:
            win_ratio = root.wins / root.count

        return win_ratio

    def select(self):
        root = self.root
        while len(root.children) is not 0:

            # Find Best Child
            best = Node(root, [])
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
            # Get prior moves and add the latest
            moves = root.moves
            moves.append(i)

            # Expand
            root.children.append(Node(root, moves))

    def simulate(self, game, root, col):
        # Returns 0 if loss, 1 if won, -1 if the game cant be continued through that col

        if game.full() is True:
            return -2  # dont try again

        if game.can_place(col) is False:
            return -1  # must try again at different column

        column = col
        while game.has_winner() is 0 and not game.full():

            # Place a piece
            game.place(column)

            column = randrange(7)

        if game.has_winner() is self.player:

            root.children[col].count += 1
            root.children[col].wins += 1
            return 1
        else:
            root.children[col].count += 1
            return 0

    def update(self, root, col):

        # Get chosen child
        leaf = root.children[col]

        while leaf.parent is not None:
            for child in leaf.parent.children:

                leaf.parent.wins += child.wins
                leaf.parent.count += child.count
                #print(leaf.parent.wins, leaf.parent.count, child.wins, child.count)
            leaf = leaf.parent

    def get_board(self, leaf):
        # Return the board for leaf node

        # Create temp board
        game = deepcopy(self.c4)

        for col in leaf.moves:
            if game.can_place(col):
                game.place(col)

        return game

    def choose_col(self):

        # Create this moves tree with depth [range(x)]
        for i in range(3):

            # Select best child
            root = self.select()

            # Get board for this child
            game = self.get_board(root)
            #print(game.board)

            # Expand the child node
            self.expand(root)

            # Simulate games for random children
            for i in range(40):

                # Board is back to state of interest
                #print(game.board)

                col = randrange(7)
                while self.simulate(deepcopy(game), root, col) is -1:
                    col = randrange(7)

            # Update along chosen path
            self.update(root, col)

        # Choose the best col
        best = Node(None, [])
        index = 0
        for i in range(len(self.root.children)):
            #print(self.root.children[i].count)

            if self.root.children[i].count is 0:
                child_ratio = 0
            else:
                child_ratio = self.root.children[i].wins / self.root.children[i].count

            if best.count is 0:
                best_ratio = 0
            else:
                best_ratio = best.wins / best.count

            if child_ratio >= best_ratio:
                #print("child", child_ratio)
                #print("best", best_ratio)
                best = self.root.children[i]
                index = i

        # Return the chosen column
        return index

    def print(self, root):
        if len(root.children) is not 0:
            i = 0
            for child in root.children:
                print(i, ": [", self.win_ratio(child), child.wins, child.count, "]")
                i += 1



#connect4 = Connect4("Random", "Random")
#P1 = MonteCarlo(1, connect4)
#P2 = MonteCarlo(2, connect4)

# root = P1.select()
# P1.expand(root)
# for i in range(1000):
#     c4 = deepcopy(connect4)
#     print(i)
#     col = randrange(7)
#     P1.simulate(c4, root, col)
#     # Update along chosen path
#     P1.update(root, col)
# connect4.place()

# while connect4.has_winner() is 0:
#     print(P2.choose_col())
#     P2.print(P2.root)


