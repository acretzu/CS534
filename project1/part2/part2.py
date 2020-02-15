#! /usr/bin/python

import os
import sys
import re
from optparse import OptionParser
import math



__options__ = None
starting_map = []

INDUSTRIAL_MAX = 0
COMMERCIAL_MAX = 0
RESIDENTIAL_MAX = 0

#
# parse command line
#
def parse_cmd_line_options():
    parser = OptionParser()
    parser.add_option("--f", action="store", type="string", dest="csv", default="urban_2.txt", help="The local path to the CSV file.")
    parser.add_option("--e", action="store", type="string", dest="algorithm", default="HC", help="The algorithm.")

    (options, args) = parser.parse_args()

    # Check that all options have been provided
    if not os.path.isfile(options.csv):
        print("Execution requires path to CSV file.")
        sys.exit(1)

    if not options.algorithm:
        print("Execution requires algorithm. (GA for genetic algorithm or HC for hill climbing).")
        sys.exit(1)

    return options

#
# Open the CSV file and copy map
#
def parse_csv_file_map():
    file_ptr = open(__options__.csv, "r")
    ret_array = []

    # Error out if we can't open the file
    if not file_ptr:
        print("Unable to open file: %s" % __options__.csv)
        sys.exit(1)

    # Get the first three values for industrial, commercial, residential maximums
    for line in file_ptr:
        csv_info = line.split(',')
        csv_info[-1] = csv_info[-1].strip()

        # Add the line to the return array
        if(len(csv_info) > 1):
            ret_array.append(csv_info)

    return ret_array

#
# Open the CSV file and get the maximums
#
def parse_csv_file_maximums():
    file_ptr = open(__options__.csv, "r")
    loc_maximums = []

    # Error out if we can't open the file
    if not file_ptr:
        print("Unable to open file: %s" % __options__.csv)
        sys.exit(1)

    # Get the first three values for industrial, commercial, residential maximums
    for line in file_ptr:
        csv_info = line.split(',')
        csv_info[-1] = csv_info[-1].strip()

        if(len(csv_info) == 1):
            loc_maximum = int(csv_info[0])
            loc_maximums.append(loc_maximum)

    return loc_maximums


################################################
# Map
################################################


class Map:

    def __init__(self, starting_map):
        self.map = starting_map
        self.score = 0
        self.industrial = 0
        self.commercial = 0
        self.residential = 0

    def place_site(self, site_type, x, y):

        """
            Checks if sites have reached maximum
            Checks if the square is valid
            Places site

            Input:
                site_type: the type of site to place on the map (0, 1, 2)
                x: x-coordinate
                y: y-coordinate

        """

        # Checks if sites have reached maximum
        if (site_type == 0 and self.industrial == INDUSTRIAL_MAX):
            return
        if (site_type == 1 and self.commercial == COMMERCIAL_MAX):
            return
        if (site_type == 2 and self.residential == RESIDENTIAL_MAX):
            return

        # Checks if the square is valid
        if self.map[x][y] == 'X':
            return

    def place_all(self):

        """
            Places all sites on random places on the map

            Input:
                None

        """

        for x in range(len(starting_map)):
            for y in range(len(starting_map[x])):
                print(starting_map[x][y])

    def penalty(self):

        """
            Calculate total penalty of the map

            Input:
                None

            Output:
                penalty: int

        """

    def score(self):

        """
            Update total score of the map

            Input:
                None

        """

    def crossover(self, partner_map):

        """
            Combine two maps to create a new map

            Input:
                partner_map: another map to crossover with

            Output:
                map: new 2d array representing a new map

        """

    def mutate(self):

        """
            Make a random swap of two sites on the map

            Input:
                None

        """

    def print(self):

        """
            Print the map in block format

            Input:
                None

        """

#####################
# Script Start
#####################

__options__ = parse_cmd_line_options()
starting_map = parse_csv_file_map()
loc_maximums = parse_csv_file_maximums()

INDUSTRIAL_MAX = loc_maximums[0]
COMMERCIAL_MAX = loc_maximums[1]
RESIDENTIAL_MAX = loc_maximums[2]

# Print the 2d map
print(starting_map)
