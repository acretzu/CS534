import sys

class Connect4:

    def __init__(self, player1, player2):
        self.board = [[0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0]]
        self.turn = 1  # Player 1


    def has_winner(self):
        """ 
        Determine if there is a winner (4 in a row, col, diag)
        """
        height = len(self.board)
        width = len(self.board[0])
        
        # Source: https://stackoverflow.com/questions/29949169/python-connect-4-check-win-function
        # check horizontal spaces
        for y in range(height):
            for x in range(width - 3):
                if self.board[y][x] == 1 and self.board[y+1][x] == 1 and self.board[y+2][x] == 1 and self.board[y+3][x] == 1:
                    return True
                if self.board[y][x] == 2 and self.board[y+1][x] == 2 and self.board[y+2][x] == 2 and self.board[y+3][x] == 2:
                    return True                

        # check vertical spaces
        for x in range(width):
            for y in range(height - 3):
                if self.board[y][x] == 1 and self.board[y][x+1] == 1 and self.board[y][x+2] == 1 and self.board[y][x+3] == 1:
                    return True
                if self.board[y][x] == 2 and self.board[y][x+1] == 2 and self.board[y][x+2] == 2 and self.board[y][x+3] == 2:
                    return True                

        # check / diagonal spaces
        for x in range(width - 3):
            for y in range(3, height):
                if self.board[y][x] == 1 and self.board[y+1][x-1] == 1 and self.board[y+2][x-2] == 1 and self.board[y+3][x-3] == 1:
                    return True
                if self.board[y][x] == 2 and self.board[y+1][x-1] == 2 and self.board[y+2][x-2] == 2 and self.board[y+3][x-3] == 2:
                    return True                

        # check \ diagonal spaces
        for x in range(width - 3):
            for y in range(height - 3):
                if self.board[y][x] == 1 and self.board[y+1][x+1] == 1 and self.board[y+2][x+2] == 1 and self.board[y+3][x+3] == 1:
                    return True
                if self.board[y][x] == 2 and self.board[y+1][x+1] == 2 and self.board[y+2][x+2] == 1 and self.board[y+3][x+3] == 2:
                    return True                

        return False
        
    def can_place(self, column):
        """
        Returns true if the board is NOT full at given column
        """
        ret_val = True

        if self.board[column][0] == 1 or self.board[column][0] == 2:
            ret_val = False

        return ret_val

    
        
    def place(self, column):
        """
        Place a piece at the given column
        """
        # Place a piece
        print("player", self.turn, "placed a piece")

        # Sanity check that column is not full
        if self.board[column][0] == 1 or self.board[column][0] == 2:
            print("Board is full at that column! Col =", column)
            sys.exit(1)

        # Add player to bottom-most row
        for row in reversed(range(7)):
            if self.board[column][row] is 0:
                self.board[column][row] = self.turn

        # Change turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
