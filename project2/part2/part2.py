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
        ret_array.append((float(csv_line[0].strip()), float(csv_line[1].strip())))

    file_ptr.close()
    return ret_array


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


# print(type(data))


# Logics
# 1) start with k randomly placed Guassians (mean, standard deviation)
# 2) for each point: P(ki|xi) = does it look like it came from?
# 3) adjust (mean, standard deviation) to fit the points assigned to them


def initial_starting_centers(data, k):
    """
    initialize starting centers

    :param data: raw data
    :param k: k clusters
    :return k_center: center points [[], [], [], ...]
                      [[k0_f0, k0_f1, k0_f2],
                       [k1_f0, k1_f1, k1_f2],
                       [k2_f0, k2_f1, k2_f2],
                       ...]
    :return k_cov: covariance for each cluster (m*m *k) (m = number of features
                   [[[ , ], [, ]],
                    [[ , ], [, ]]
                    ...]
    """

    # number of features
    m = len(data[0])

    # randomly choose centers from the data
    k_center = random.sample(data, k=k)

    # initialize covariance as identity matrix
    k_cov = [np.identity(m, dtype=np.float64).tolist() for i in range(k)]

    return k_center, k_cov


def maximization(data, cluster_prob):
    """
    adjust (mean, standard deviation) to fit the points assigned to them

    :param data:
    :param cluster_prob: the probability of each point assigned to each cluster
                             [[P(k1|x1), P(k2|x1), P(k3|x1), ...],
                              [P(k1|x2), P(k2|x2), P(k3|x2), ...],
                              ......]
    :return k_center: center points [[], [], [], ...]
                      [[k0_f0, k0_f1, k0_f2],
                       [k1_f0, k1_f1, k1_f2],
                       [k2_f0, k2_f1, k2_f2],
                       ...]
    :return k_cov: covariance for each cluster (m*m *k) (m = number of features
                   [[[ , ], [, ]],
                    [[ , ], [, ]]
                    ...]
    """

    # number of clusters
    k = len(cluster_prob[0])
    # number of data
    n = len(data)
    # number of features
    m = len(data[0])

    # using numpy
    cluster_prob_np = np.array(cluster_prob)
    data_np = np.array(data)

    # update the k centers
    k_center_np = np.dot(cluster_prob_np.T, data_np) / cluster_prob_np.sum(axis=0).T

    # update the cov
    # TODO

    # back to list
    k_center = k_center_np.tolis()
    k_cov = k_cov_np.list()

    return k_center, k_cov


def expectation(data, k_center, k_cov):

    """
    for each point: P(ki|xi) = does it look like it came from?

    :param data:
    :param k_center: center points [[], [], [], ...]
                      [[k0_f0, k0_f1, k0_f2],
                       [k1_f0, k1_f1, k1_f2],
                       [k2_f0, k2_f1, k2_f2],
                       ...]
    :param k_cov: covariance for each cluster (m*m *k) (m = number of features)
                   [[[ , ], [ , ]],
                    [[ , ], [ , ]]
                    ...]

    :return cluster_prob: the probability of each point assigned to each cluster
                          [[P(k1|x1), P(k2|x1), P(k3|x1), ...],
                           [P(k1|x2), P(k2|x2), P(k3|x2), ...],
                            ......]
    """
