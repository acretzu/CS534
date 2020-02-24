Part 2

Run the program like this:
"python3 part2.py --f urban_1.txt --a GA"
"python3 part2.py --f urban_2.txt --a HC"

New txt files can be added and ran
"python3 part2.py --f urban_3.txt --a GA"

###################
The output
###################

First it prints the initial configuration.  The second map is the same map without the numbers showing, it makes it easier to read.
___________________________________________
Industrial Max:  1
Commercial Max:  2
Residential Max:  2
Starting Map:
 [['2' '3' '3' 'X' '6']
 ['4' 'X' '3' '2' '3']
 ['3' '1' '1' '6' 'X']
 ['7' '6' '5' '8' '5']
 ['S' '6' 'S' '9' '1']
 ['4' '7' '2' '6' '5']]

[[' ' ' ' ' ' 'X' ' ']
 [' ' 'X' ' ' ' ' ' ']
 [' ' ' ' ' ' ' ' 'X']
 [' ' ' ' ' ' ' ' ' ']
 ['S' ' ' 'S' ' ' ' ']
 [' ' ' ' ' ' ' ' ' ']]
___________________________________________

Next it prints the best score and the best map, and the times.
___________________________________________
Best Score:
 32

Best Map:
[[' ' ' ' ' ' 'X' ' ']
 [' ' 'X' ' ' ' ' ' ']
 [' ' ' ' ' ' ' ' 'X']
 ['R' ' ' 'C' ' ' ' ']
 ['S' 'R' 'S' ' ' ' ']
 ['C' ' ' ' ' ' ' ' ']]
Best Map Achieved At:  4.232215166091919
Total Time:  4.840843915939331
___________________________________________