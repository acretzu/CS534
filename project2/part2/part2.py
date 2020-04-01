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
import matplotlib.pyplot as plt



#
# Open the CSV file and get data
# Format: <x float>, <y float>
#
def parse_csv_file(path_to_file):
    file_ptr = open(path_to_file, "r")
    

    # Error out if we can't open the file
    if not file_ptr:
        print("Unable to open file: %s" % __options__.csv)
        sys.exit(1)

    rows = 0
    cols = -1
    for line in file_ptr:
        rows += 1
        if cols == -1:
            cols = len(line.split(","))
    file_ptr.seek(0)
    
    ret_array = np.empty((rows, cols), float)
 
    row = 0    
    for line in file_ptr:        
        csv_line = line.split(",")
        # Add each column
        temp_arr = np.array([])
        for col in range(cols):            
            converted_line = (float(csv_line[col].strip()))
            ret_array[row, col] = converted_line

        row += 1

    file_ptr.close()
    return np.array(ret_array)

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
    m = data.shape[1]

    # randomly choose centers from the data
    k_center = data[random.sample([i for i in range(data.shape[0])], k=k)]

    # initialize covariance as identity matrix
    k_cov = np.array([np.identity(m, dtype=np.float64) for i in range(k)])


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
    k = cluster_prob.shape[1]
    # number of data
    n = data.shape[0]
    # number of features
    m = data.shape[1]


    # cluster_prob = em_data.prob_data_given_cluster[0]
    # data = em_data.data

    k_cov = []

    # using numpy
    cluster_prob_np = np.array(cluster_prob)
    data_np = np.array(data)

    # update the k centers    
    k_center = np.dot(cluster_prob_np.T, data_np) / cluster_prob_np.sum(axis=0).reshape(1, k).T

    # update the cov
    for ki in range(k):
        ki_cov = np.dot(np.dot((data - k_center[ki]).T, np.diag(cluster_prob_np[:, ki])),
                        (data - k_center[ki]))
        ki_cov_normalize = ki_cov / cluster_prob_np[:, ki].sum()
        ki_cov_list = ki_cov_normalize.tolist()

        k_cov.append(ki_cov_list)

    # back to list
    # k_center = k_center_np.tolist()

    k_cov = np.array(k_cov)

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

    # number of features
    size = k_center.shape[1]

    # number of data
    n = data.shape[0]

    data_np = np.array(data)

    cluster_prob = []

    for k in range(k_center.shape[0]):
        a = 1 / (((2 * np.pi) ** (size / 2)) * (np.linalg.det(k_cov[k]) ** (1 / 2)))
        b = (-1 / 2) * ((data_np - k_center[k]).dot(np.linalg.inv(k_cov[k]))).dot((data_np - k_center[k]).T)

        #log(a^b) = b*log(a)
        log_val = b * np.log(a)
        #cluster_prob_k = np.diagonal(a*np.exp(b))
        cluster_prob_k = np.diagonal(log_val)

        cluster_prob.append(cluster_prob_k)
    cluster_prob = np.array(cluster_prob).T


    # (just remember to normalize the results in the end,
    # since the point must belong to one of the clusters
    # so probabilities have to sum to 1)
    # from professor's reply

    cluster_prob = cluster_prob / cluster_prob.sum(axis=1).reshape(n, 1)

    return cluster_prob


def get_loglikelihood(cluster_prob):
    
    total_likelihood = np.log(cluster_prob).sum()

    return total_likelihood

def get_bic(total_likelihood, n_data, m_fea, k):

    parameters = k*(m_fea + m_fea ** 2)

    bic = -2*total_likelihood + math.log(n_data)*parameters

    return bic


def plot_loglikelihood(total_likelihood_list, plot_filename = "plot_ll.png"):

    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    fig.set_size_inches(8, 6)

    ax.plot([i for i in range(len(total_likelihood_list))], total_likelihood_list)

    plt.title('log-likelihood vs. # of iterations')
    plt.xlabel('# of iterations')
    plt.ylabel('log-likelihood')

    fig.savefig(plot_filename)  # save the figure to file
    plt.close(fig)


# Main function
def train_em(data, k, n_epochs):

    k_center, k_cov = initial_starting_centers(data, k)

    total_likelihood_list = []

    # iterate
    # TODO: add random restarts  (and sideways moves)
    for e in range(n_epochs):

        # expectation
        cluster_prob = expectation(data, k_center, k_cov)

        # maximization
        k_center, k_cov = maximization(data, cluster_prob)

        # record likelihood
        total_likelihood = get_loglikelihood(cluster_prob)
        total_likelihood_list.append(total_likelihood)

    # save log-likelihood vs. # of iteration
    plot_loglikelihood(total_likelihood_list, plot_filename="plot_ll/plot_ll_"+str(k)+".png")

    return total_likelihood_list



def plot_bic(bic_list, k_list, plot_filename = "plot_bic.png"):

    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    fig.set_size_inches(8, 6)

    ax.plot(k_list, bic_list)

    plt.title('BIC vs. # of clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('BIC')

    fig.savefig(plot_filename)  # save the figure to file
    plt.close(fig)

def determine_lowest_k_using_bic(data, k_range = 10):

    bic_list = []
    for ki in range(k_range):
        total_likelihood_list = train_em(data, ki + 2, 20)

        bic = get_bic(total_likelihood_list[-1], data.shape[0], data.shape[1], ki)
        bic_list.append(bic)

    k_list = [ki + 1 for ki in range(k_range)]

    plot_bic(bic_list, k_list)




#####################
# Script Start
#####################

# random.seed(3)

# Default values
file_name = "sample_em_data.csv"
num_clusters = 5


# File is arg1
if len(sys.argv) >= 2:
    file_name = sys.argv[1]

# Groups is arg2
if len(sys.argv) >= 3:
    file_name = sys.argv[1]
    num_clusters = int(sys.argv[2])

data = parse_csv_file(file_name)
print(data)

if num_clusters == 0:
    determine_lowest_k_using_bic(data, k_range = 13)
else:
    train_em(data, num_clusters, 20)

#maximization()
#expectation()



