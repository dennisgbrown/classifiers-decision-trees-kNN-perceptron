# -*- coding: utf-8 -*-
"""
Perceptron homework

Dennis Brown, COMP6636, 21 FEB 2021
"""

import numpy as np


def Perceptron(X, Y, beta, iteration_limit):
    """
    Perceptron. Given a 2-D set of data X (samples are rows, columns
    features), a vector Y of classifications, a learning rate (beta),
    and an iteration limit, train and return a weight vector that
    can be used to classify the given data.
    """

    # Initialize the weight vector and add entry for bias term
    w = np.zeros(len(X[0]) + 1)

    # Initialize Y_hat
    Y_hat = np.zeros(len(X))

    # Repeat the main loop until we have convergence or reach the
    # iteration limit
    iterations = 0
    converged = False
    while(not(converged) and (iterations < iteration_limit)):

        # For each sample in X, calculate w's classification error
        # and update w.
        for i in range(len(X)):
            # Add a 1 to the front of every term to account for w's bias
            sample = np.insert(X[i], 0, 1)
            Y_hat[i] = 1 if (np.matmul(w.T, sample) > 0) else -1
            error = Y[i] - Y_hat[i]
            w += sample * error * beta
            iterations += 1

        # If the difference between Y ajd Y_hat is effectively 0,
        # consider it converged.
        if (np.linalg.norm(Y - Y_hat) < .0000001):
            converged = True

    print('Final w = ', w, 'in', iterations, 'steps; converged?', converged)

    return w


def testPerceptron(X, Y, w):
    Y_hat = np.zeros(len(Y))
    for i in range(len(X)):
        print('norm of', X[i], '=', np.linalg.norm(X[i]))
        sample = np.insert(X[i], 0, 1)
        Y_hat[i] = 1 if (np.matmul(w.T, sample) > 0) else -1
        print('delta =', np.matmul(sample.T, w))
    print('Y   :', Y)
    print('Y^  :', Y_hat)
    print('Diff:', Y - Y_hat)


# Test it out

# XOR does not converge
# X = np.array([[0, 0],
#               [0, 1],
#               [1, 0],
#               [1, 1]])
# Y = np.array([1, -1, -1, -1])

# Augmented XOR does converge
X = np.array([[0, 0, 1, 0, 0, 0],
              [0, 1, 0, 1, 0, 0],
              [1, 0, 0, 0, 1, 0],
              [1, 1, 0, 0, 0, 1]])
Y = np.array([1, -1, -1, -1])

# X = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [.5, .5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, .25, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#               [.75, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#               [.67, .67, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#               [.2, .3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#               [.3, .4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#               [.5, .6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#               [.7, .8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#               [.2, .6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#               [.3, .8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
# Y = np.array([1, -1, -1, 1,  1, -1, 1, -1,  1, 1, 1, -1, 1, 1])


w = Perceptron(X, Y, .01, 9999)
testPerceptron(X, Y, w)

