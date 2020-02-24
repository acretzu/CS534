Part 1

#####################  
Running the program
#####################
part1.py --a <algorithm: 1=A*, 2=HC> --e <h1 or h2> --f <path/to/csv>

Flag description:
--a is the algorithm. Use 1 for A* and 2 for Hill Climbing.
--f is the path to the CSV file.
--e is the heuristic. Use h1 for H1 and h2 for H2.

Example:
(Windows)
python3 part1.py --a 2 --e h2 --f heavy_queens_board.csv
    
(Linux)    
part1.py --a 1 --e h1 --f heavy_queens_board.csv

###################
The output
###################

For the both A* and Hill Climbing algorithms will output the following:
1) The moves from the starting board to the goal.
2) Statistics including: total time, total cost, nodes expanded, moves to solve, and effective branching factor.

Example A* output:
    
A* algorithm - Moves from start to goal:
. . . . . 
. . . . 2 
. . . 4 . 
. 3 . . . 
9 . 1 . . 
---------------------------------
. . . . . 
. . . . 2 
. 3 . 4 . 
. . . . . 
9 . 1 . . 
---------------------------------
. . . . . 
. . . . 2 
. 3 . . . 
. . . 4 . 
9 . 1 . . 
---------------------------------
. . 1 . . 
. . . . 2 
. 3 . . . 
. . . 4 . 
9 . . . . 
---------------------------------
Heuristic        = 1
Total time (s)   = 0.017
Total cost       = 45
Nodes expanded   = 51
Moves to solve   = 3
Branching factor = 2.672

 
Example Hill Climbing output:
    
Start Game using Hill Climbing with h2
----------------------------------------------
Start board state
. . . . . 
. . . . 2 
. . . 4 . 
. 3 . . . 
9 . 1 . . 
---------------------------------
Heuristic        = h2
Total time (s)   = 0.0015015602111816406
Total cost       = 363
Nodes expanded   = 4
Moves to solve   = 4
Branching factor = 1.3195079107728942
. . . . . 
. . . . 2 
. . . 4 . 
. 3 . . . 
9 . 1 . . 
---------------------------------
9 . . . . 
. . . . 2 
. . . 4 . 
. 3 . . . 
. . 1 . . 
---------------------------------
9 . . . . 
. . . . 2 
. . . . . 
. 3 . . . 
. . 1 4 . 
---------------------------------
9 . . . . 
. . 1 . 2 
. . . . . 
. 3 . . . 
. . . 4 . 
---------------------------------
9 . . . . 
. . 1 . . 
. . . . 2 
. 3 . . . 
. . . 4 . 
---------------------------------
