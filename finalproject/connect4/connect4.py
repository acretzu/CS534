#! /usr/bin/python3.6

import sys
from qlearning import QLearner

class Connect4:

    def __init__(self, player1, player2):
        self.board = [[0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0]]
        self.player1 = player1
        self.player2 = player2
        self.draw = False
        self.prev_board = None
        self.turn = 1  # Player 1


    def check_winner(self):
        """ 
        Determine if there is a winner (4 in a row, col, diag)
        Returns 0 if no winner, 1 if player 1 wins, 2 if player 2 wins, 3 if draw
        """
        width = len(self.board)
        height = len(self.board[0])
        
        # Source: https://stackoverflow.com/questions/29949169/python-connect-4-check-win-function
        # check horizontal spaces
        for y in range(height):
            for x in range(width - 3):
                if self.board[x][y] == 1 and self.board[x+1][y] == 1 and self.board[x+2][y] == 1 and self.board[x+3][y] == 1:
                    return 1
                if self.board[x][y] == 2 and self.board[x+1][y] == 2 and self.board[x+2][y] == 2 and self.board[x+3][y] == 2:
                    return 2                

        # check vertical spaces
        for x in range(width):
            for y in range(height - 3):
                if self.board[x][y] == 1 and self.board[x][y+1] == 1 and self.board[x][y+2] == 1 and self.board[x][y+3] == 1:
                    return 1
                if self.board[x][y] == 2 and self.board[x][y+1] == 2 and self.board[x][y+2] == 2 and self.board[x][y+3] == 2:
                    return 2                

        # check / diagonal spaces
        for x in range(width - 3):
            for y in range(3, height):
                if self.board[x][y] == 1 and self.board[x+1][y-1] == 1 and self.board[x+2][y-2] == 1 and self.board[x+3][y-3] == 1:
                    return 1
                if self.board[x][y] == 2 and self.board[x+1][y-1] == 2 and self.board[x+2][y-2] == 2 and self.board[x+3][y-3] == 2:
                    return 2                

        # check \ diagonal spaces
        for x in range(width - 3):
            for y in range(height - 3):
                if self.board[x][y] == 1 and self.board[x+1][y+1] == 1 and self.board[x+2][y+2] == 1 and self.board[x+3][y+3] == 1:
                    return 1
                if self.board[x][y] == 2 and self.board[x+1][y+1] == 2 and self.board[x+2][y+2] == 1 and self.board[x+3][y+3] == 2:
                    return 2                


        for x in range(width):
            if self.board[x][0] == 1 or self.board[x][0] == 2:
                self.draw = True
            else:
                self.draw = False
                break

        if self.draw is True:
            return 3
            
        return 0

    def clear_board(self):
        """
        Sets the board to all 0s
        """
        self.board = [[0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0]]
    
    def can_place(self, column):
        """
        Returns true if the board is NOT full at given column
        """
        ret_val = True

        if self.board[column][0] == 1 or self.board[column][0] == 2:
            ret_val = False

        return ret_val

    def available_cols(self):
        ret_val = [0, 1, 2, 3, 4, 5, 6]

        for i in range(7):
            if self.board[i][0] == 1 or self.board[i][0] == 2:
                ret_val.remove[i]
                print("col ", i , " is not available")
                print(ret_val)
        
        return ret_val


        
    def place(self, column):
        """
        Place a piece at the given column
        """
        # Place a piece
        print("Player", self.turn, "placed piece at column:", column+1)
        
        # Sanity check that column is not full
        if self.board[0][column] == 1 or self.board[0][column] == 2:
            print("Board is full at that column! Col =", column)
            sys.exit(1)

        # Save current board as previous
        self.prev_board = [row[:] for row in self.board]

        foo = ""
        for r in range(len(self.board)):
            for c in range(len(self.board[0])):
                foo += self.int2str(self.prev_board[r][c]) + " "
            foo += "\n"
        print("Prev_board:")
        print(foo)

        
        # Add player to bottom-most row
        for h in reversed(range(6)):
            if self.board[h][column] is 0:
                self.board[h][column] = self.turn
                break

        # Change turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1

        # Debug
        print(self)

    def __str__(self):
        """
        String representation of connect4 board
        """
        ret_str = ""

        for r in range(len(self.board)):
            for c in range(len(self.board[0])):
                ret_str += self.int2str(self.board[r][c]) + " "
            ret_str += "\n"
                
        return ret_str

    def int2str(self, x):
        """
        Converts 1 (player 1) to X.
        Converts 2 (player 2) to O.
        Converts 0 (blanks) to "-".
        """
        ret_val = "-"

        if x == 1:
            ret_val = "X"
        elif x == 2:
            ret_val = "O"
            
        return ret_val

            
    def play(self, games=1):
        """
        Main game loop. Plays full game iterations.
        """    

        p1 = None
        p2 = None
        
        if self.player1 == "Random":
            p1 = QLearner(1) #TODO: Make random player

        if self.player2 == "QL":
            p2 = QLearner(2)
        
        while games > 0:
            print("Play iteration = ", games)

            while self.check_winner() == 0:
                p1_move = p1.random_action()
                p2_move = p2.random_action()

                if(self.turn == 1):
                    self.place(p1_move)
                else:
                    self.place(p2_move)
            
            
            print("The winner is player ", self.check_winner())
            self.clear_board()
            games -= 1
            


############
# Main Start
############

connect4 = Connect4("Random", "QL")
connect4.play(2)
