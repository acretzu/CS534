#! /usr/bin/python3.6

import os
import sys
import re
import math
import numpy as np
import random
import time
import sys
import copy
import pandas as pd






#
# Open the CSV file and get data
# Format: <x float>, <y float>
#
def parse_csv_file(path_to_file):
    file_ptr = open(path_to_file, "r")
    ret_array = []

    # Error out if we can't open the file
    if not file_ptr:
        print("Unable to open file: %s" % __options__.csv)
        sys.exit(1)

    # Loop thru each line (row) and extract wieght and col
    row = 0
    for line in file_ptr:
        csv_line = line.split(",")
        print("csv 0 = ", csv_line[0], "csv 1 = ", csv_line[1])
        ret_array.append( (float(csv_line[0].strip()), float(csv_line[1].strip())) )
        
    file_ptr.close()
    return ret_array


#
# start -> expectation -> maximization -> is_converged -- yes? --> done
#  ^                                            | 
#  |------------------- no? --------------------|
#

def start():
    return 0

def expectation():
    return 0

def maximization():
    return 0

def is_converged():
    return 0




#####################
# Script Start
#####################

# Default values
file_name = "sample_em_data.csv"
num_groups = 0

# File is arg1
if len(sys.argv) >= 2:
    file_name = sys.argv[1]

# Groups is arg2
if len(sys.argv) >= 3:
    file_name = sys.argv[1]
    num_groups = sys.argv[2]

# Parse CSV
data = parse_csv_file(file_name)


# Debug print
print("file = ", file_name)
print("num_groups = ", num_groups)
for xy in data:
    print(xy[0], ", ", xy[1])

