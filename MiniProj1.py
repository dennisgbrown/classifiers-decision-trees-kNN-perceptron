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
    X = np.zeros((num_samples, max_feature_id))
    curr_sample = 0
    datafile.seek(0)
    for line in datafile:
        tokens = line.split()
        y[curr_sample] = float(tokens[0])        
        for feature in tokens[1:]:
            # print(token)
            feature_id = int(feature.split(':')[0]) - 1
            feature_val = float(feature.split(':')[1])
            X[curr_sample][feature_id] = feature_val
        curr_sample += 1

    datafile.close()
    return X, y



def get_neighbors(X, test_sample, num_neighbors):
    """
    Given training data, a test sample, and a number of
    neighbors, return the closest neighbors.
    """
    # Calculate all distances from the training samples
    # to this test sample. Collect index, distance
    indices_and_distances = list()
    for i in range(len(X)):
        indices_and_distances.append([i, np.linalg.norm(test_sample - X[i])])
    # Sort by distance
    indices_and_distances.sort(key=lambda _: _[1])
    # Make a list of requested number of closest neighbors
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(indices_and_distances[i][0])
    return neighbors
 

def predict_classification(X, y, test_sample, num_neighbors):
    """
    Given training data, classification data, a test sample, 
    and a number of neighbors, predict which classification
    the test sample belongs to.
    """
    neighbors = get_neighbors(X, test_sample, num_neighbors)
    output_values = list()
    for i in range(len(neighbors)):
        output_values.append(y[neighbors[i]])
    prediction = max(set(output_values), key=output_values.count)
    return prediction

    
def k_nearest_neighbors(X, y, test_samples, num_neighbors):
    """
    Given training data, classification data, test data, 
    and a number of neighbors, predict which classification
    each test sample belongs to.
    """
    predictions = list()
    for i in range(len(test_samples)):
        output = predict_classification(X, y, test_samples[i], num_neighbors)
        predictions.append(output)
    return(predictions)

# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/




def trainPerceptron(X, y, beta, step_limit):
    """
    Perceptron. Given a 2-D set of data X (samples are rows, columns
    features), a vector Y of classifications, a learning rate (beta),
    and a step limit, train and return a weight vector that
    can be used to classify the given data.
    """

    # Initialize the weight vector and add entry for bias term
    w = np.zeros(len(X[0]) + 1)

    # Initialize Y_hat
    y_hat = np.zeros((len(X), 1))

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
            y_hat[i] = 1 if (np.matmul(w.T, sample) > 0) else -1
            error = y[i] - y_hat[i]
            w += sample * error * beta
            steps += 1

        # If the difference between y and y_hat is effectively 0,
        # consider it converged.
        if (np.linalg.norm(y - y_hat) < .0000001):
            converged = True

    print('Final w = ', w, 'in', steps, 'steps; converged?', converged)

    return w


def testWeights(X, y, w):
    y_hat = np.zeros(len(y))
    for i in range(len(X)):
        sample = np.insert(X[i], 0, 1)
        y_hat[i] = 1 if (np.matmul(w.T, sample) > 0) else -1
    print('y   :', y)
    print('y^  :', y_hat)
    print('Diff:', y - y_hat)
    print('sum Diff', sum(y - y_hat))





def main():
    
    X, y = libsvm_scale_import('data/iris.scale')
    # X, y = libsvm_scale_import('data/a4a')
    # X, y = libsvm_scale_import('data/a4a.t')

    print(X)


    blah = k_nearest_neighbors(X[:100], y[:100], X[100:], 5)
    print(blah)
    
    # w = trainPerceptron(X, y, .01, 99999)
    # testWeights(X, y, w)

if __name__ == '__main__':
    main()
    
