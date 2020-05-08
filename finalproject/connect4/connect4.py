class Connect4:

    def __init__(self, player1, player2):
        self.board = [[0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0]]
        self.turn = 1  # Player 1

    def place(self, column):
        # Place a piece
        print("player", self.turn, "placed a piece")

        # Change turn
        if self.turn == 1:
            self.turn = 2
        elif self.turn == 2:
            self.turn = 1
