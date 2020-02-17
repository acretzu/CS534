#! /usr/bin/python

import os
import sys
import re
from itertools import product
from optparse import OptionParser
import math
import random
import numpy as np

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
    parser.add_option("--f", action="store", type="string", dest="csv", default="urban_2.txt",
                      help="The local path to the CSV file.")
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
        if (len(csv_info) > 1):
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

        if (len(csv_info) == 1):
            loc_maximum = int(csv_info[0])
            loc_maximums.append(loc_maximum)

    return loc_maximums


################################################
# Map
################################################


class Map:

    def __init__(self, starting_map):
        self.map = starting_map
        self.height = len(starting_map)
        self.width = len(starting_map[0])
        self.score = 0
        self.industrial = 0
        self.commercial = 0
        self.residential = 0

    def place_site(self, site_type, x, y):

        """
            Checks if sites have reached maximum
            Checks if the square is valid
            Adds cost of placing
            Places site

            Input:
                site_type: the type of site to place on the map (0, 1, 2)
                x: x-coordinate
                y: y-coordinate

        """

        # Checks if sites have reached maximum
        if (site_type == 'I' and self.industrial == INDUSTRIAL_MAX):
            return 0
        if (site_type == 'C' and self.commercial == COMMERCIAL_MAX):
            return 0
        if (site_type == 'R' and self.residential == RESIDENTIAL_MAX):
            return 0

        # Checks if the square is invalid
        if self.map[y][x] == 'X':
            return 0
        elif self.map[y][x] == 'I':
            return 0
        elif self.map[y][x] == 'C':
            return 0
        elif self.map[y][x] == 'R':
            return 0

        # Compute cost and add
        if self.map[y][x] == 'S':
            cost = 1
        else:
            cost = int(self.map[y][x]) + 2
        self.score -= cost

        # Place the site
        self.map[y][x] = site_type

        # Increment
        if (site_type == 'I'):
            self.industrial += 1
        elif (site_type == 'C'):
            self.commercial += 1
        elif (site_type == 'R'):
            self.residential += 1

        return 1

    def place_all(self):

        """
            Places all sites on random places on the map

        """

        cells = list(range(0, (self.width * self.height) - 1))

        site = ''

        # Fill the map
        while True:

            # Break when max is reached
            if (self.industrial == INDUSTRIAL_MAX and
                    self.commercial == COMMERCIAL_MAX and
                    self.residential == RESIDENTIAL_MAX):
                break

            # Randomly pick a spot on the map
            cell = random.choice(cells)
            x = cell % self.width
            y = math.floor(cell / self.width)

            # Randomly pick a site type
            rand = random.randint(0, 2)
            if rand == 0:
                site = 'I'
            elif rand == 1:
                site = 'C'
            elif rand == 2:
                site = 'R'

            # Try to place the site
            if self.place_site(site, x, y):
                cells.remove(cell)

    def toxic_penalty(self):

        """
            Calculate penalty of toxic sites

            Output:
                penalty: int

        """

        penalty = 0

        for y in range(self.height):
            for x in range(self.width):
                if self.map[y][x] == 'X':

                    # Get neighbors
                    neighbors = self.neighbors(x, y, 2)

                    for s in neighbors:

                        # Industrial penalty
                        if s == 'I':
                            penalty += 10
                        # Commercial penalty
                        elif s == 'C':
                            penalty += 20
                        # Residential penalty
                        elif s == 'R':
                            penalty += 20

        return penalty

    def scenic_bonus(self):

        """
            Calculate bonus of scenic sites

            Output:
                bonus: int

        """

        bonus = 0

        for y in range(self.height):
            for x in range(self.width):
                if self.map[y][x] == 'S':

                    # Get neighbors
                    neighbors = self.neighbors(x, y, 2)

                    for s in neighbors:

                        # Residential bonus
                        if s == 'R':
                            bonus += 10

        return bonus

    def industrial_bonus(self):

        """
            Calculate bonus for industrial neighbors

            Output:
                bonus: int

        """

        bonus = 0

        for y in range(self.height):
            for x in range(self.width):
                if self.map[y][x] == 'I':

                    # Get neighbors
                    neighbors = self.neighbors(x, y, 2)

                    for s in neighbors:

                        # Industrial bonus
                        if s == 'I':
                            bonus += 2

        return bonus

    def commercial_bonus(self):

        """
            Calculate bonus for commercial sites

            Output:
                bonus: int

        """

        bonus = 0

        for y in range(self.height):
            for x in range(self.width):
                if self.map[y][x] == 'C':

                    # Get neighbors
                    neighbors = self.neighbors(x, y, 3)

                    for s in neighbors:

                        # Residential bonus
                        if s == 'R':
                            bonus += 4

                    neighbors = self.neighbors(x, y, 2)

                    for s in neighbors:

                        # Commercial penalty
                        if s == 'C':
                            bonus -= 4

        return bonus

    def residential_bonus(self):

        """
            Calculate bonus for residential sites

            Output:
                bonus: int

        """

        bonus = 0

        for y in range(self.height):
            for x in range(self.width):
                if self.map[y][x] == 'R':

                    # Get neighbors
                    neighbors = self.neighbors(x, y, 3)

                    for s in neighbors:

                        # Industrial penalty
                        if s == 'I':
                            bonus -= 5

                        # Commercial bonus
                        if s == 'C':
                            bonus += 4

        return bonus

    def update_score(self):

        """
            Update total score of the map (bonus - penalty)

        """
        self.score = 0
        self.score -= self.toxic_penalty()
        self.score += self.scenic_bonus()
        self.score += self.industrial_bonus()
        self.score += self.commercial_bonus()
        self.score += self.residential_bonus()
        return self.score

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
            Make a random swap of one or two sites on the map

            Input:
                None

        """

    def neighbors(self, x, y, distance):

        """
            Return neighbors in Manhattan distance away from x,y cell

            Input:
                x: x-coordinate of cell
                y: y-coordinate of cell

            Output:
                list of neighbors within distance of the cell

        """

        neighbors = []

        for i in range(-distance, distance + 1):
            for j in range(-distance, distance + 1):
                if (abs(j) + abs(i) <= distance and
                        x-j >= 0 and
                        y-i >= 0 and
                        x-j < self.width and
                        y-i < self.height and
                        not (j == 0 and i == 0)):
                    neighbors.append(self.map[y-i][x-j])

        return neighbors

    def print(self):

        """
            Print the map in block format

        """

        print(self.map)


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
print(INDUSTRIAL_MAX)
print(COMMERCIAL_MAX)
print(RESIDENTIAL_MAX)
print(np.array(starting_map))

maps = []
scores = []

for i in range(10):

    m = Map(np.array(starting_map))
    m.place_all()
    score = m.update_score()

    scores.append(score)
    maps.append(m)

print(scores)