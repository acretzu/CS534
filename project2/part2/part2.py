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
from sklearn import datasets
from sklearn.cluster import KMeans


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


def kmeans_initial_starting_centers(data, k):
    m = data.shape[1]

    # randomly choose 70% of the data to do the kmeans
    subset_number = int(data.shape[0] * 0.5)
    data_subset = data[random.sample([i for i in range(data.shape[0])], k=subset_number)]

    kmeans = KMeans(n_clusters=k).fit(data_subset)

    k_center = kmeans.cluster_centers_

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

    k_cov = []

    k_center = np.dot(cluster_prob.T, data) / cluster_prob.sum(axis=0).reshape(1, k).T

    # update the cov for each cluster
    for ki in range(k):
        ki_cov = np.dot(np.dot((data - k_center[ki]).T, np.diag(cluster_prob[:, ki])),
                        (data - k_center[ki]))
        ki_cov_normalize = ki_cov / cluster_prob[:, ki].sum()

        k_cov.append(ki_cov_normalize)

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

    cluster_prob = []

    for k in range(k_center.shape[0]):
        a = 1 / (((2 * np.pi) ** (size / 2)) * (np.linalg.det(k_cov[k]) ** (1 / 2)))

        b = (-1 / 2) * ((data - k_center[k]).dot(np.linalg.inv(k_cov[k]))).dot((data - k_center[k]).T)

        cluster_prob_k = np.diagonal(a * np.exp(b))

        cluster_prob.append(cluster_prob_k)

    cluster_prob = np.array(cluster_prob).T

    cluster_prob_normal = []

    # n means the number of data
    for i in range(n):
        # if the sum of all probabilities for one data is not 0
        # Normalize
        if cluster_prob.sum(axis=1).reshape(n, 1)[i] != 0:
            cluster_prob_normal.append(cluster_prob[i] / cluster_prob.sum(axis=1).reshape(n, 1)[i])
        # if the sum of all probabilities for one data == 0
        # Then we just keep [0,0,0,0]
        else:
            cluster_prob_normal.append(cluster_prob[i])

    # return: the first one is normalzied probability, the second one is unnormalzied probability
    return np.array(cluster_prob_normal), cluster_prob


def get_loglikelihood(cluster_prob):
    total_likelihood = np.log(cluster_prob.sum(axis=1)).sum()

    return total_likelihood


def get_bic(total_likelihood, n_data, m_fea, k):
    parameters = k * (m_fea + m_fea ** 2)

    # bic = -2*total_likelihood + ((math.log(n_data))*parameters)
    # bic = -2*total_likelihood + ((math.log(n_data*(n_data +2)/24))*parameters)
    # bic = -2*total_likelihood + 2*((math.log(n_data))*parameters)
    # bic = -2*total_likelihood

    bic = -2 * total_likelihood + 3 * ((math.log2(n_data)) * parameters)

    return bic


def plot_loglikelihood(total_likelihood_list, plot_filename = "plot_ll.png"):
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    fig.set_size_inches(8, 6)

    ax.plot([i + 1 for i in range(len(total_likelihood_list[1:]))], total_likelihood_list[1:])

    plt.title('log-likelihood vs. # of iterations')
    plt.xlabel('# of iterations')
    plt.ylabel('log-likelihood')

    fig.savefig(plot_filename)  # save the figure to file
    plt.close(fig)


# Main function
def train_em(data, k, time_up=9.5, diff_thred=0.1):
    total_likelihood_series_list = []
    k_center_list = []
    k_cov_list = []

    start_time = time.time()

    time_limit = time.time() - start_time

    while time_limit < time_up:

        k_center, k_cov = kmeans_initial_starting_centers(data, k)
        # k_center, k_cov = initial_starting_centers(data, k)

        total_likelihood_list = []

        total_likelihood_old = float('-inf')
        diff_ll = float('inf')

        # diff threshold and time limit
        while (diff_ll > diff_thred) & (time_limit < time_up):
            # expectation
            cluster_prob, cluster_prob_nn = expectation(data, k_center, k_cov)

            # maximization
            k_center, k_cov = maximization(data, cluster_prob)

            # record likelihood
            total_likelihood = get_loglikelihood(cluster_prob_nn)

            total_likelihood_list.append(total_likelihood)

            diff_ll = total_likelihood - total_likelihood_old

            total_likelihood_old = total_likelihood

            time_limit = time.time() - start_time

        total_likelihood_series_list.append(total_likelihood_list)

        k_center_list.append(k_center)
        k_cov_list.append(k_cov)

    # find best one
    ll_best_list = [ll[-1] for ll in total_likelihood_series_list]

    postion_best_ll = ll_best_list.index(max(ll_best_list))

    best_ll_series = total_likelihood_series_list[postion_best_ll]
    best_k_center = k_center_list[postion_best_ll]
    best_k_cov = k_cov_list[postion_best_ll]

    plot_loglikelihood(best_ll_series, plot_filename="plot_ll/plot_ll_" + str(k) + ".png")

    return best_ll_series, best_k_center, best_k_cov


def plot_bic(bic_list, k_list, plot_filename="plot_bic.png"):
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    fig.set_size_inches(8, 6)

    ax.plot(k_list, bic_list)

    plt.title('BIC vs. # of clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('BIC')

    fig.savefig(plot_filename)  # save the figure to file
    plt.close(fig)


def determine_lowest_k_using_bic(data, time_up_bic=9.5, bic_diff_thred=0.01):
    start_time_bic = time.time()

    k_start = 3
    bic_old = get_bic(train_em(data, k=(k_start - 1), time_up=2, diff_thred=1)[0][-1],
                      data.shape[0],
                      data.shape[1],
                      k=(k_start - 1))

    bic_list = []
    bic_list.append(bic_old)
    bic_diff_pct = 100

    # k_range = 16

    while (bic_diff_pct > bic_diff_thred) & ((time.time() - start_time_bic) < time_up_bic):
        # while k_start < k_range:

        total_likelihood_list, centers, cov = train_em(data, k=k_start, time_up=2, diff_thred=1)

        bic = get_bic(total_likelihood_list[-1], data.shape[0], data.shape[1], k_start)

        bic_diff_pct = (bic_old - bic) / bic_old

        bic_list.append(bic)

        bic_old = bic

        k_start = k_start + 1

    k_list = [ki + 2 for ki in range(len(bic_list))]
    plot_bic(bic_list, k_list)

    return (k_start - 1), total_likelihood_list, centers, cov, bic


#####################
# Script Start
#####################

# random.seed(3)

# Default values
file_name = "sample_em_data.csv"
# file_name = "mini_sample.csv"
# file_name = "data_3_cluster.csv"
num_clusters = 5

# File is arg1
if len(sys.argv) >= 2:
    file_name = sys.argv[1]

# Groups is arg2
if len(sys.argv) >= 3:
    file_name = sys.argv[1]
    num_clusters = int(sys.argv[2])

data = parse_csv_file(file_name)
# print(data)

total_start_time = time.time()
if num_clusters == 0:
    best_k, ll_data, final_centers, final_cov, bic = determine_lowest_k_using_bic(data,
                                                                                  bic_diff_thred=10e-4,
                                                                                  time_up_bic=9.5)
    print("Best number of clusters:", best_k)
    print("BIC:", bic)
    print("Final Cluster Centers:")
    for k in range(len(final_centers)):
        print("Cluster", k + 1, )
        print("Mean:", final_centers[k])
        print("Cov: ", )
        print(final_cov[k])
    print("------------------------------------------------")
    print("BIC:", bic)
    print("LL:", ll_data[-1])
    print("Total Execution Time = ", time.time() - total_start_time)

else:
    ll_data, final_centers, final_cov = train_em(data, num_clusters, time_up=9.5, diff_thred=1)
    print("Final Cluster Centers:")
    for k in range(len(final_centers)):
        print("Cluster", k + 1, )
        print("Mean:", final_centers[k])
        print("Cov: ", )
        print(final_cov[k])
    print("------------------------------------------------")
    print("LL:", ll_data[-1])
    print("Total Execution Time = ", time.time() - total_start_time)

# iris = datasets.load_iris()
# X = iris.data

# determine_lowest_k_using_bic(data, k_range=10)
# determine_lowest_k_using_bic(X, k_range=10)

# total_likelihood_list, cluster_prob = train_em(X, 3)

# print(cluster_prob.sum(axis=1))
