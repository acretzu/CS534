#! /usr/bin/python

import os
import sys
import re
from optparse import OptionParser

__options__ = None
starting_board = []

#
# parse command line
#
def parse_cmd_line_options():
    parser = OptionParser()
    parser.add_option("--e", action="store", type="int", dest="heuristic", default=1, help="The heuristic.")
    parser.add_option("--a", action="store", type="int", dest="algorithm", default=1, help="The algorithm.")
    parser.add_option("--f", action="store", type="string", dest="csv", default="", help="The local path to the CSV file.")

    (options, args) = parser.parse_args()

    # Check that all options have been provided
    if not options.heuristic:
        print("Execution requires heurisitic (1 for H1 or 2 for H2).")
        sys.exit(1)

    if not os.path.isfile(options.csv):
        print("Execution requires path to CSV file.")
        sys.exit(1)

    if not options.algorithm:
        print("Execution requires algorithm. (1 for A* or 2 for Hill Climbing).")
        sys.exit(1)

    return options

#
# Open the CSV file and get board information
# Format: <Queen Weight>,<Queen Position>
#
def parse_csv_file():
    file_ptr = open(__options__.csv, "r")
    ret_array = []

    # Error out if we can't open the file
    if not file_ptr:
        print("Unable to open file: %s" % __options__.csv)
        sys.exit(1)

    # Loop thru each line and extract wieght and position
    for line in file_ptr:
        csv_info = line.split(",")
        i = 0
        while i < len(csv_info):
            queen_weight = int(csv_info[i])
            queen_position = int(csv_info[i+1])
            # Add weight and position as a tuple into array
            ret_array.append((queen_weight, queen_position))
            i += 2

    return ret_array



#####################
# Script Start
#####################

__options__ = parse_cmd_line_options()
starting_board = parse_csv_file()

for queen in starting_board:
    print("Queen weight = %d, Queen position = %d" % (queen[0], queen[1]))
