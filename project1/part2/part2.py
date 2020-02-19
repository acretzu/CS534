#! /usr/bin/python

import os
import sys
import re
from itertools import product
from optparse import OptionParser
import math
import random
import numpy as np
import time

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
    parser.add_option("--f", action="store", type="string", dest="csv", default="urban_3.txt",
                      help="The local path to the CSV file.")
    parser.add_option("--e", action="store", type="string", dest="algorithm", default="GA", help="The algorithm.")

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
        if len(csv_info) > 1:
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
        self.map = np.array(starting_map)
        self.starting_map = np.array(starting_map)
        self.height = len(starting_map)
        self.width = len(starting_map[0])
        self.score = 0
        self.industrial = 0
        self.commercial = 0
        self.residential = 0

    def place_site(self, site, x, y):

        """
            Checks if sites have reached maximum
            Checks if the cell is taken
            Adds cost of placing
            Increments total site count
            Places site

            Input:
                site: the type of site to place on the map
                x: x-coordinate
                y: y-coordinate

        """

        # Checks if the cell is taken
        if self.map[y][x] == 'X':
            return -5
        elif self.map[y][x] == 'I':
            return -4
        elif self.map[y][x] == 'C':
            return -3
        elif self.map[y][x] == 'R':
            return -2

        # Checks if sites have reached maximum
        if site == 'I' and self.industrial == INDUSTRIAL_MAX:
            return 0
        if site == 'C' and self.commercial == COMMERCIAL_MAX:
            return 0
        if site == 'R' and self.residential == RESIDENTIAL_MAX:
            return 0

        # Place the site
        self.map[y][x] = site

        # Increment
        if site == 'I':
            self.industrial += 1
        elif site == 'C':
            self.commercial += 1
        elif site == 'R':
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

            # Max is reached
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

    def build_cost(self):

        """
            Calculate cost for building

            Output:
                cost: int

        """

        cost = 0

        for y in range(self.height):
            for x in range(self.width):

                if (self.map[y][x] == 'I' or
                        self.map[y][x] == 'C' or
                        self.map[y][x] == 'R'):

                    if self.starting_map[y][x] == 'S':
                        cost += 1
                    else:
                        cost += int(self.starting_map[y][x])

        return cost

    def update_score(self):

        """
            Update total score of the map (bonus - penalty)

        """
        self.score = 0
        self.score -= self.build_cost()
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
                map: new child map

        """

        new_map = Map(np.array(self.starting_map))

        cells = list(range(0, (self.width * self.height) - 1))

        while True:

            # Max is reached
            if (new_map.industrial == INDUSTRIAL_MAX and
                    new_map.commercial == COMMERCIAL_MAX and
                    new_map.residential == RESIDENTIAL_MAX):
                break

            # Check if all cells have been checked
            if len(cells) == 0:
                break

            # Randomly pick a spot on the map
            cell = random.choice(cells)
            x = cell % self.width
            y = math.floor(cell / self.width)

            if new_map.industrial < INDUSTRIAL_MAX and (self.map[y][x] == 'I' or partner_map.map[y][x] == 'I'):

                # Place an industrial site on the same cell in new map
                new_map.place_site('I', x, y)

            elif new_map.commercial < COMMERCIAL_MAX and (self.map[y][x] == 'C' or partner_map.map[y][x] == 'C'):

                # Place an industrial site on the same cell in new map
                new_map.place_site('C', x, y)

            elif new_map.residential < RESIDENTIAL_MAX and (self.map[y][x] == 'R' or partner_map.map[y][x] == 'R'):

                # Place an industrial site on the same cell in new map
                new_map.place_site('R', x, y)

            cells.remove(cell)

        # Randomly place rest
        new_map.place_all()

        return new_map

    def mutate(self):

        """
            Make a random swap of one or two sites on the map

        """

        cells = list(range(0, (self.width * self.height) - 1))

        while True:

            # Randomly pick a spot on the map
            cell = random.choice(cells)
            x = cell % self.width
            y = math.floor(cell / self.width)

            if self.map[y][x] == 'I':

                # Set the cell back to starting value
                self.map[y][x] = self.starting_map[y][x]
                self.industrial -= 1

                # Replace the site
                self.place_all()
                break

            elif self.map[y][x] == 'C':

                # Set the cell back to starting value
                self.map[y][x] = self.starting_map[y][x]
                self.commercial -= 1

                # Replace the site
                self.place_all()
                break

            elif self.map[y][x] == 'R':

                # Set the cell back to starting value
                self.map[y][x] = self.starting_map[y][x]
                self.residential -= 1

                # Replace the site
                self.place_all()
                break

            cells.remove(cell)

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
                        x - j >= 0 and
                        y - i >= 0 and
                        x - j < self.width and
                        y - i < self.height and
                        not (j == 0 and i == 0)):
                    neighbors.append(self.map[y - i][x - j])

        return neighbors

    def copy_map(self, other_map):

        """
            Return a new copy of the given map

        """

        new_map = np.array(other_map)
        return new_map

    def print_fancy(self):

        """
            Print the map so only sites are present

        """

        output = np.array(self.map)

        for y in range(self.height):
            for x in range(self.width):
                if self.map[y][x] == 'X':
                    output[y][x] = 'X'
                elif self.map[y][x] == 'S':
                    output[y][x] = 'S'
                elif self.map[y][x] == 'I':
                    output[y][x] = 'I'
                elif self.map[y][x] == 'C':
                    output[y][x] = 'C'
                elif self.map[y][x] == 'R':
                    output[y][x] = 'R'
                else:
                    output[y][x] = ' '

        print(output)

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

if __options__.algorithm == 'GA':

    # Start a timer
    start_time = time.time()

    # Print the 2d map
    print("Industrial Max: ", INDUSTRIAL_MAX)
    print("Commercial Max: ", COMMERCIAL_MAX)
    print("Residential Max: ", RESIDENTIAL_MAX)
    print("Starting Map:\n", np.array(starting_map), "\n")
    init_map = Map(starting_map)
    init_map.print_fancy()
    print("\n")

    # Helper function
    def get_score(map):
        return map.score


    # Population pool urban_3
    pool_size = 100  # this has to be even
    elite_percent = 5  # percent
    generations = 50
    mutation_chance = 1  # percent
    map_pool = []
    parents = []
    new_map_pool = []

    # Initial population
    for i in range(pool_size):
        m1 = Map(np.array(starting_map))
        m1.place_all()
        score = m1.update_score()
        map_pool.append(m1)

    for generation in range(generations):

        # Sort the pool
        map_pool.sort(key=get_score)

        # Shift onto 0
        offset = map_pool[0].score
        for i in range(pool_size):
            map_pool[i].score -= offset

        # Normalize
        score_total = 0
        for m in map_pool:
            score_total += m.score

        if score_total != 0:
            for m in map_pool:
                m.score = m.score / score_total

        # Accumulated scores
        accumulated_score = 0
        for m in map_pool:
            accumulated_score += m.score
            m.score = accumulated_score

        # Breed
        while len(new_map_pool) < math.ceil(pool_size - (pool_size*elite_percent/100)):

            # Randomly choose two maps to breed
            i1 = random.uniform(0, 1)
            i2 = random.uniform(0, 1)

            for m in map_pool:
                if m.score >= i1:
                    parents.append(m)
                    break
            for m in map_pool:
                if m.score >= i2:
                    parents.append(m)
                    break

            if score_total == 0:
                i3 = random.randint(0, len(map_pool) - 1)
                i4 = random.randint(0, len(map_pool) - 1)
                parents.append(map_pool[i3])
                parents.append(map_pool[i4])

            # Crossover and add child to new generation
            child = parents[0].crossover(parents[1])

            # Add child to new population
            new_map_pool.append(child)

            # Clear parents array
            parents.clear()

        # Elitism
        for i in range(math.ceil(pool_size - pool_size*elite_percent/100), pool_size):
            new_map_pool.append(map_pool[i])

        # Mutation
        for j in range(len(new_map_pool) - math.ceil(pool_size * elite_percent / 100)):
            mutate = random.uniform(0, 100 / mutation_chance)
            if mutate < 1:
                new_map_pool[j].mutate()

        # Clear population
        map_pool.clear()

        # New population becomes the current
        map_pool = new_map_pool.copy()

        # Reset new population for later use
        new_map_pool.clear()

        for m in map_pool:
            m.update_score()

        #print(map_pool[pool_size - 1].score)

    scores = []
    for m in map_pool:
        m.update_score()
        scores.append(m.score)
    print("Last Generation Scores: \n", scores, "\n")
    map_pool[pool_size - 1].update_score()
    print("Best Score: \n", map_pool[pool_size - 1].score, "\n")
    print("Best Map:")
    map_pool[pool_size - 1].print_fancy()
    print("Total Time: ", time.time()-start_time)


elif __options__.algorithm == 'HC':
    print("implement HC here")