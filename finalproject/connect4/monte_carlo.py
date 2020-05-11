from copy import deepcopy, copy
from math import ceil, floor
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

    def __init__(self, player, c4, depth=100, rollouts=1000):
        self.root = Node(None, [])
        self.c4 = c4
        self.player = player
        self.depth = depth
        self.rollouts = rollouts

    def win_ratio(self, root):
        if root.count is 0:
            win_ratio = 0
        else:
            win_ratio = root.wins / root.count

        return win_ratio

    def select(self, degree):
        root = self.root
        level = 0
        while len(root.children) is not 0:

            # Ratios
            ratios = []

            # Create list of ratios
            for child in root.children:

                child_ratio = self.win_ratio(child)

                # Add ratio to list
                ratios.append(child)

            # Sort
            #print(ratios)
            ratios.sort(key = lambda x: self.win_ratio(x))
            #print(ratios)

            # Get best
            if level is not 0:
                degree = 0
            best = ratios[len(ratios)-degree-1]

            # Keep track of depth
            level += 1

            root = best

        #print(root.children)
        #print("selected (", root.moves, "): [", root.wins, root.count, "]")
        #print(root)
        return root

    def expand(self, root):
        moves = []
        for i in range(7):
            # Get prior moves and add the latest
            moves = copy(root.moves)
            moves.append(i)

            # Expand
            root.children.append(Node(root, moves))
            #print("expanded (", root.moves, "): [", root.wins, root.count, "]")

    def simulate(self, game, root, col):
        # Returns 0 if loss, 1 if won, -1 if the game cant be continued through that col

        if game.can_place(col) is False:
            return -1  # must try again at different column

        column = col
        while game.has_winner() is 0 and not game.full():

            # Place a piece
            if game.can_place(column) is True:
                game.place(column)

            column = randrange(7)

        if game.has_winner() is self.player:
            root.children[col].count += 1
            root.children[col].wins += 1
            return 1
        else:
            root.children[col].count += 1
            return 0

    def update(self, leaf):

        # Before update, save the past wins
        past_wins = leaf.wins
        past_count = leaf.count

        # Update leaf
        for child in leaf.children:
            # Wins update
            leaf.wins += child.wins
            leaf.count += child.count

        new_wins = leaf.wins - past_wins
        new_count = leaf.count - past_count

        #print("updated (", leaf.moves, "): [", past_wins, past_count, "]")

        while leaf.parent is not None:

            # Add leaf wins
            leaf.parent.wins += new_wins

            # Add rollouts
            leaf.parent.count += new_count

            # climb up path
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

        # 1st best, 2nd best, 3rd best, etc.
        degree = 0

        # Create this moves tree with depth [range(x)]
        for i in range(self.depth):

            #print(degree)

            # Select best child
            root = self.select(degree)

            # Get board for this child
            game = self.get_board(root)

            # Check if the tree is expandable
            cell = 0
            if len(root.moves) > 0:
                cell = game.board[0][root.moves[len(root.moves) - 1]]
            if game.has_winner() is not 0 or \
                    game.full() is True or \
                    cell is not 0:
                # we are stuck so we need to retry
                if degree > 6:
                    degree = 0
                else:
                    degree += 1
            else:
                #print("expand")
                # Expand
                self.expand(root)

                # Simulate games for random children
                for j in range(floor(self.rollouts/((len(root.moves)+1) ** 2))):

                    col = randrange(7)
                    while self.simulate(deepcopy(game), root, col) is -1:
                        col = randrange(7)

                # Update along chosen path
                self.update(root)

        # Choose the best col
        best = Node(None, [])
        index = 0
        for i in range(len(self.root.children)):

            if self.root.children[i].count is 0:
                child_ratio = 0
            else:
                child_ratio = self.root.children[i].wins / self.root.children[i].count

            if best.count is 0:
                best_ratio = 0
            else:
                best_ratio = best.wins / best.count

            if child_ratio >= best_ratio:
                best = self.root.children[i]
                index = i

        # Return the chosen column
        return index

    def print(self, root):
        if len(root.children) is not 0:
            i = 0
            for child in root.children:
                print(i+1, ": [", self.win_ratio(child), child.wins, child.count, "]")
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


