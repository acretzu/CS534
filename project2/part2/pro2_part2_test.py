import numpy as np


from part2 import *


def test_initial_starting_centers():

    test_data = np.genfromtxt('mini_sample.csv', delimiter=',')
    k = 4

    k_center, k_cov = initial_starting_centers(test_data, k)

    assert k_center.shape == (k, test_data.shape[1])
    assert k_cov.shape == (k, test_data.shape[1], test_data.shape[1])


def test_maximization():

    test_data = np.genfromtxt('mini_sample.csv', delimiter=',')

    cluster_prob = np.array([[5.76898296e-066, 1.00000000e+00, 6.75367966e-012],
                             [3.44721725e-033, 1.00000000e+00, 6.64130326e-045],
                             [2.10374007e-035, 1.00000000e+00, 4.27114643e-040],
                             [8.80372672e-023, 1.00000000e+00, 2.44899817e-058],
                             [1.00000000e+000, 5.47623657e-12, 1.27914086e-107],
                             [2.66511928e-147, 1.68422026e-42, 1.00000000e+000],
                             [1.00000000e+000, 2.69296288e-34, 2.66511928e-147],
                             [9.11547521e-013, 1.00000000e+00, 9.86203532e-071],
                             [8.80781967e-013, 1.00000000e+00, 2.11270340e-055],
                             [1.00000000e+000, 2.81152797e-22, 3.42927717e-102]])

    k_center, k_cov = maximization(test_data, cluster_prob)

    # set groud truth
    k_center_true = np.array([[11.63533185, 18.624955],
                              [9.50933994, 8.87316693],
                              [14.79985897, -4.7340844]])

    k_cov_true = np.array([[[36.95287458,  3.91131058], [3.91131058,  3.1952501 ]],
                           [[7.01813031,  2.3296442 ], [2.3296442 , 11.59339858]],
                           [[2.69785951e-10, -2.97216595e-10], [-2.97216595e-10,  3.27436266e-10]]])

    assert np.allclose(k_center, k_center_true)
    assert np.allclose(k_cov, k_cov_true)


def test_expectation():

    test_data = np.genfromtxt('mini_sample.csv', delimiter=',')

    k_center = np.array([[ 8.99827417, 20.59062949],
                         [9.73764022, 8.17902846],
                         [14.79985897, -4.7340844 ]])

    k_cov = np.array([[[1., 0.], [0., 1.]],
                      [[1., 0.], [0., 1.]],
                      [[1., 0.], [0., 1.]]])

    cluster_prob = expectation(test_data, k_center, k_cov)

    # set ground truth

    cluster_prob_true = np.array([[5.76898296e-066, 1.00000000e+00, 6.75367966e-012],
                                  [3.44721725e-033, 1.00000000e+00, 6.64130326e-045],
                                  [2.10374007e-035, 1.00000000e+00, 4.27114643e-040],
                                  [8.80372672e-023, 1.00000000e+00, 2.44899817e-058],
                                  [1.00000000e+000, 5.47623657e-12, 1.27914086e-107],
                                  [2.66511928e-147, 1.68422026e-42, 1.00000000e+000],
                                  [1.00000000e+000, 2.69296288e-34, 2.66511928e-147],
                                  [9.11547521e-013, 1.00000000e+00, 9.86203532e-071],
                                  [8.80781967e-013, 1.00000000e+00, 2.11270340e-055],
                                  [1.00000000e+000, 2.81152797e-22, 3.42927717e-102]])

    assert cluster_prob.shape == (test_data.shape[0], k_center.shape[0])
    assert np.allclose(cluster_prob, cluster_prob_true)

