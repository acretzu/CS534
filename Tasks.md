Part 1
1. Input reading -- Adrian 
  a. Command line arguments
  b. Put initial board state into memory
2. Classes to store data -- Tianfang/Marc
  a. Board (H1 and H2)
    - Keep track of number of nodes expanded.
    - Keep track of effective branching factor
    - Keep track of cost
    - Keep track of moves
  b. Function to move 
3. Implementing the A* algorithm -- Adrian
4. Implementing the hill-climbing algorithm -- Tianfang

Note: Heuristic functions might be coupled closer to the Board class rather than the algorithm.

Part 2
1. Input reading -- Marc
  a. Command line arguments
  b. Storing of the map.
2. Class for map -- Marc/Adrian
  a. Map
    - The score for this map
    - Industrial, commercial, and residential sites marked.  
    - Classes for different areas (residential, industrial, toxic, etc.)
    - Make rules for points
3. Implementing the genetic algorithm  -- Marc
4. Implementing the hill-climbing algorithm  -- Tianfang 

