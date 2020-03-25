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
        #print("csv 0 = ", csv_line[0], "csv 1 = ", csv_line[1])
        ret_array.append( (float(csv_line[0].strip()), float(csv_line[1].strip())) )

    file_ptr.close()
    return ret_array

##############################################################################

# class EM_Data:
#     """
#     A class containing all the necessary data structures for executing EM.
#     data                    : list containing the real data (tuple)
#     num_data                : amount of real data
#     num_clusters            : number of clusters
#     prob_cluster            : probability of a cluster
#     prob_data_given_cluster : 2d-matrix representing the probability each data
#                               belongs to a given cluster. [cluster][data] = %
#     """
#     def __init__(self, d, n):
#         self.data = d
#         self.num_data = len(d)
#         self.num_clusters = n

#         # Make up guesses for all probabilities
#         self.prob_cluster = [0 for x in range(self.num_clusters)]
        
#         # Randomize cluster probabilities so that they add to 1
#         temp = 1
#         for c in range(self.num_clusters):
#             if c == 0:
#                 self.prob_cluster[c] = random.random()
#                 temp -= self.prob_cluster[c]
#             elif c < self.num_clusters-1:
#                 self.prob_cluster[c] = random.uniform(0, 1-self.prob_cluster[c-1])
#                 temp -= self.prob_cluster[c]
#             else:
#                 self.prob_cluster[c] = temp

#         # Debug print
#         #for c in range(self.num_clusters):
#         #    print(self.prob_cluster[c])
                
#         self.prob_data_given_cluster = [[random.uniform(0,1) for x in range(self.num_data)] for y in range(self.num_clusters)]

#         # The likely cluster of each data point
#         self.likely_cluster = []

#         # Debug print
#         #for c in range(self.num_clusters):
#         #    for d in range(self.num_data):
#         #        print(self.prob_data_given_cluster[c][d])

        
##############################################################################

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


    cluster_prob = em_data.prob_data_given_cluster[0]
    data = em_data.data

    k_cov = []

    # using numpy
    cluster_prob_np = np.array(cluster_prob)
    data_np = np.array(data)

    # update the k centers    
    k_center_np = np.dot(cluster_prob_np.T, data_np) / cluster_prob_np.sum(axis=0).reshape(1, 2).T
    
    # update the cov
    for ki in range(k):
        ki_cov = np.dot(np.dot((data - k_center_np[ki]).T, np.diag(cluster_prob_np[:, ki])),
                        (data - k_center_np[ki]))
        ki_cov_normalize = ki_cov / cluster_prob_np[:, ki].sum()
        ki_cov_list = ki_cov_normalize.tolist()

        k_cov.append(ki_cov_list)

    # back to list
    k_center = k_center_np.tolist()

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
    size = len(k_center)
    a = 1 / ( ((2 * np.pi) ** (size/2)) * (np.linalg.det(k_cov) ** (1/2)) )
    b = (-1/2) * ((data - k_center).T.dot(np.linalg.inv(cov))).dot((data-k_center))
    return float(a * np.exp(b))
    
    

#####################
# Script Start
#####################

random.seed(3)

# Default values
file_name = "sample_em_data.csv"
num_clusters = 0


# File is arg1
if len(sys.argv) >= 2:
    file_name = sys.argv[1]

# Groups is arg2
if len(sys.argv) >= 3:
    file_name = sys.argv[1]
    num_clusters = int(sys.argv[2])

data = parse_csv_file(file_name)


#maximization()
#expectation()



