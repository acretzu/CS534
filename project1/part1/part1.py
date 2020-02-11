#! /usr/bin/python

import os
import sys
import re
from optparse import OptionParser

__options__ = None

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





#####################
# Script Start
#####################
print('Hello World!')

__options__ = parse_cmd_line_options()
