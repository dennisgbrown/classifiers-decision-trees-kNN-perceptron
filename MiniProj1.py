# -*- coding: utf-8 -*-
"""
Mini project 1

Dennis Brown, COMP6636, 03 MAR 2021
"""

import numpy as np


def libsvm_scale_import(filename):
    """
    Read data from a libsvm .scale file
    """
    datafile = open(filename, 'r')
    
    # First pass: get dimensions of data
    num_samples = 0
    max_feature_id = 0
    for line in datafile:
        # print(line)
        num_samples += 1
        tokens = line.split()
        for feature in tokens[1:]:
            # print(token)
            feature_id = int(feature.split(':')[0])
            max_feature_id = max(feature_id, max_feature_id)
       
    print(num_samples, 'x', max_feature_id)

    # Second pass: read data into arrays
    y = np.zeros(num_samples)
    x = np.zeros((num_samples, max_feature_id))
    curr_sample = 0
    datafile.seek(0)
    for line in datafile:
        tokens = line.split()
        y[curr_sample] = float(tokens[0])        
        for feature in tokens[1:]:
            # print(token)
            feature_id = int(feature.split(':')[0]) - 1
            feature_val = float(feature.split(':')[1])
            x[curr_sample][feature_id] = feature_val
        curr_sample += 1

    datafile.close()
    return y, x


def trainPerceptron(X, Y, beta, step_limit):
    """
    Perceptron. Given a 2-D set of data X (samples are rows, columns
    features), a vector Y of classifications, a learning rate (beta),
    and a step limit, train and return a weight vector that
    can be used to classify the given data.
    """

    # Initialize the weight vector and add entry for bias term
    w = np.zeros(len(X[0]) + 1)

    # Initialize Y_hat
    Y_hat = np.zeros(len(X))

    # Repeat the main loop until we have convergence or reach the
    # iteration limit
    steps = 0
    converged = False
    while(not(converged) and (steps < step_limit)):

        # For each sample in X, calculate w's classification error
        # and update w.
        for i in range(len(X)):
            # Add a 1 to the front of every term to account for w's bias
            sample = np.insert(X[i], 0, 1)
            Y_hat[i] = 1 if (np.matmul(w.T, sample) > 0) else -1
            error = Y[i] - Y_hat[i]
            w += sample * error * beta
            steps += 1

        # If the difference between Y ajd Y_hat is effectively 0,
        # consider it converged.
        if (np.linalg.norm(Y - Y_hat) < .0000001):
            converged = True

    print('Final w = ', w, 'in', steps, 'steps; converged?', converged)

    return w


def testPerceptron(X, Y, w):
    Y_hat = np.zeros(len(Y))
    for i in range(len(X)):
        sample = np.insert(X[i], 0, 1)
        Y_hat[i] = 1 if (np.matmul(w.T, sample) > 0) else -1
    print('Y   :', Y)
    print('Y^  :', Y_hat)
    print('Diff:', Y - Y_hat)



def main():
    
    # y, x = libsvm_scale_import('data/iris.scale')
    y, x = libsvm_scale_import('data/a4a')
    # y, x = libsvm_scale_import('data/a4a.t')

    print(x)
    
    # Test it out
    
    # # XOR does not converge
    # X = np.array([[0, 0],
    #               [0, 1],
    #               [1, 0],
    #               [1, 1]])
    # Y = np.array([1, -1, -1, 1])
    
    # # Augmented XOR does converge
    # X = np.array([[0, 0, 1, 0, 0, 0],
    #               [0, 1, 0, 1, 0, 0],
    #               [1, 0, 0, 0, 1, 0],
    #               [1, 1, 0, 0, 0, 1]])
    # Y = np.array([1, -1, -1, 1])
    
    # w = trainPerceptron(X, Y, .01, 9999)
    # testPerceptron(X, Y, w)

    w = trainPerceptron(x, y, .01, 9999)
    testPerceptron(x, y, w)

if __name__ == '__main__':
    main()
    
