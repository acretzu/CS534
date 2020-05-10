from copy import deepcopy, copy
from math import ceil
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

    def __init__(self, player, c4, depth=50, rollouts=1000):
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

    def select(self):
        root = self.root
        i = 0
        while len(root.children) is not 0:

            # Find Best Child
            best = Node(root, [])
            for child in root.children:

                child_ratio = self.win_ratio(child)
                best_ratio = self.win_ratio(best)

                # choose a different one
                if best_ratio == 1:
                    best_ratio = 0
                if child_ratio == 1:
                    child_ratio = 0

                if child_ratio >= best_ratio:
                    best = child

            i += 1

            root = best

        #print(root.children)

        #print("selected (", i , "): [", root.wins, root.count, "]")
        #print(root)
        return root

    def expand(self, root):
        moves = []
        for i in range(7):
            # Get prior moves and add the latest
            moves = copy(root.moves)
            moves.append(i)
            #print("root moves",root.moves)
            #print("child" , moves)

            # Expand
            root.children.append(Node(root, moves))

    def simulate(self, game, root, col):
        # Returns 0 if loss, 1 if won, -1 if the game cant be continued through that col

        if game.can_place(col) is False:
            #print("cant place")
            return -1  # must try again at different column

        column = col
        while game.has_winner() is 0 and not game.full():

            # Place a piece
            if game.can_place(column) is True:
                game.place(column)

            column = randrange(7)

        #print(game.has_winner())
        #print(game.__str__())
        if game.has_winner() is self.player:
            #print("win")
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

        #print("leaf:", leaf.wins, leaf.count)

        # Update leaf
        #print("children")
        for child in leaf.children:
            # Wins update
            leaf.wins += child.wins
            #print(child.wins, child.count)
            leaf.count += child.count

        #print("leaf (past wins): ", past_wins)
        #print("leaf (wins): ", leaf.wins)
        new_wins = leaf.wins - past_wins
        new_count = leaf.count - past_count

        #print("leaf (after):", leaf.wins, leaf.count)

        while leaf.parent is not None:
            #print("parent (before):", leaf.parent.wins, leaf.parent.count)

            # Before update save past
            #parent_past_wins = leaf.parent.wins

            # Add leaf wins
            leaf.parent.wins += new_wins

            #print("new wins:", new_wins)
            # Add rollouts
            leaf.parent.count += new_count

            #print("parent (after):",leaf.parent.wins, leaf.parent.count)

            # climb up path
            leaf = leaf.parent

    def get_board(self, leaf):
        # Return the board for leaf node

        # Create temp board
        game = deepcopy(self.c4)
        #print("root board:\n", game.__str__())

        for col in leaf.moves:
            if game.can_place(col):
                game.place(col)

        #print("selected:\n", game.__str__())

        return game

    def choose_col(self):

        # Create this moves tree with depth [range(x)]
        for i in range(self.depth):

            # Select best child
            root = self.select()

            # Get board for this child
            game = self.get_board(root)
            #print("selected:\n",game.__str__())

            # Expand the child node
            if game.has_winner() is 0 and game.full() is False:
                self.expand(root)


                #print(ceil(self.rollouts/((len(root.moves)+1)**2)))
                # Simulate games for random children
                for j in range(ceil(self.rollouts/((len(root.moves)+1)**2))):

                    # Board is back to state of interest
                    #print(game.board)

                    col = randrange(7)
                    while self.simulate(deepcopy(game), root, col) is -1:
                        col = randrange(7)

                # Update along chosen path
                self.update(root)

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
                #print(i+1, ": [", self.win_ratio(child), child.wins, child.count, "]")
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


