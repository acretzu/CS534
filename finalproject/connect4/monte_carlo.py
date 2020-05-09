from random import randrange

from connect4 import Connect4


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
            print("yo")

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

    def simulate(self, c4, root, col):
        # Returns 0 if loss, 1 if won, -1 if the game cant be continued through that col

        if c4.can_place(col) is False:
            return -1  # must try again at different column

        while c4.has_winner() is 0:

            # Place a piece
            c4.place(col)

            col = randrange(7)

        if c4.has_winner() is self.player:
            root.count += 1
            root.wins += 1
            return 1
        else:
            root.count += 1
            return 0

    def update(self, root, col):

        # Get chosen child
        leaf = root.children[col]

        while leaf.parent is not None:
            for child in leaf.parent.children:
                leaf.parent.wins += child.wins
                leaf.parent.count += child.count

    def get_board(self, leaf):
        # Return the board for leaf node

        # Create temp board
        c4 = Connect4(self.c4)

        for col in leaf.moves:
            if c4.can_place(col):
                c4.place(col)

        return c4

    def choose_col(self):

        # Create this moves tree with depth [range(x)]
        for i in range(5):

            # Select best child
            root = self.select()

            # Get board for this child
            c4 = self.get_board(root)

            # Expand the child node
            self.expand(root)

            # Simulate games for random children
            for i in range(100):

                col = randrange(7)
                self.simulate(c4, root, col)

                # Update along chosen path
                self.update(root, col)

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

        # Enter the best state
        self.root = best

        # Return the chosen column
        return i

    def print(self):
        root = self.root
        while len(root.children) is not 0:
            i = 0
            for child in root.children:
                print("|", i, "- [", self.win_ratio(child), "]")
                self.print(child)



connect4 = Connect4("Random", "Random")
P1 = MonteCarlo(1, connect4)
P2 = MonteCarlo(2, connect4)


P1.select()
P1.print()


