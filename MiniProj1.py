# -*- coding: utf-8 -*-
"""
Mini project 1

Dennis Brown, COMP6636, 03 MAR 2021
"""

import numpy as np
import copy


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

    # Second pass: read data into array
    X = np.zeros((num_samples, max_feature_id + 1))
    curr_sample = 0
    datafile.seek(0)
    for line in datafile:
        tokens = line.split()
        X[curr_sample][0] = float(tokens[0])        
        for feature in tokens[1:]:
            # print(token)
            feature_id = int(feature.split(':')[0])
            feature_val = float(feature.split(':')[1])
            X[curr_sample][feature_id] = feature_val
        curr_sample += 1
    datafile.close()

    return X



def get_neighbors(data, test_sample, num_neighbors):
    """
    Given training data, a test sample, and a number of
    neighbors, return the closest neighbors.
    """
    # Calculate all distances from the training samples
    # to this test sample. Collect index, distance
    indices_and_distances = list()
    for i in range(len(data)):
        dist = np.linalg.norm(test_sample[1:] - (data[i])[1:]) # leave out classification at pos 0
        indices_and_distances.append([i, dist])

    # Sort by distance
    indices_and_distances.sort(key=lambda _: _[1])

    # Make a list of requested number of closest neighbors
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(indices_and_distances[i][0])
    return neighbors
 

def predict_classification(data, test_sample, num_neighbors):
    """
    Given training data, a test sample, and a number of neighbors, 
    predict which classification the test sample belongs to.
    """
    # Get closest neighbors
    neighbors = get_neighbors(data, test_sample, num_neighbors)
    
    # Create list of classifications of the neighbors
    classifications = list()
    for i in range(len(neighbors)):
        classifications.append(data[neighbors[i]][0]) # 0 = classification
    
    # Return the most common classification of the neighbors
    prediction = max(set(classifications), key = classifications.count)
    return prediction

    
def k_nearest_neighbors(data, test_samples, num_neighbors):
    """
    Given sample data (samples are rows, columns
    features, and samples have classifications in position 0),
    test data, and a number of neighbors, predict which classification
    each test sample belongs to.
    """
    predictions = list()
    for i in range(len(test_samples)):
        output = predict_classification(data, test_samples[i], num_neighbors)
        predictions.append(output)
    return(predictions)

# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/




def trainPerceptron(data, beta, step_limit):
    """
    Perceptron. Given a set of data (samples are rows, columns
    features, and samples have classifications in position 0), 
    a learning rate (beta), and a step limit, train and return a 
    weight vector that can be used to classify the given data.
    """

    # Initialize the weight vector including bias element
    w = np.zeros(len(data[0]))

    # Initialize y_hat
    y_hat = np.zeros(len(data))

    # Slice off y
    y = data[:,1]

    # Repeat the main loop until we have convergence or reach the
    # iteration limit
    steps = 0
    converged = False
    while(not(converged) and (steps < step_limit)):
        converged = True
        
        # For each sample in X, calculate w's classification error
        # and update w.
        for i in range(len(data)):
            # Replace classification in sample[0] with a 1 to allow
            # for a biased weight vector
            biased_sample = np.copy(data[i])
            biased_sample[0] = 1

            # Get prediction and error, then update weight vector
            y_hat[i] = 1 if (np.matmul(w.T, biased_sample) > 0) else -1
            error = y[i] - y_hat[i]
            w += biased_sample * error * beta
            steps += 1

            # If error on this element is > a very small value, we have
            # not converged.
            if (abs(error) > 0.000001):
                converged = False

    print('Final w = ', w, 'in', steps, 'steps; converged?', converged)

    return w


def testWeights(data, w):
    y_hat = np.zeros(len(data))
    y = data[:,1]
    for i in range(len(data)):
        biased_sample = np.copy(data[i])
        biased_sample[0] = 1
        y_hat[i] = 1 if (np.matmul(w.T, biased_sample) > 0) else -1
    print('y   :', y)
    print('y^  :', y_hat)
    print('Diff:', y - y_hat)
    print('sum Diff', sum(data[:,1] - y_hat))





def main():
    
    data = libsvm_scale_import('data/iris.scale')
    # data = libsvm_scale_import('data/a4a')
    # data = libsvm_scale_import('data/a4a.t')

    # print(data)
    # print('---------------')


    # # Test kNN
    # shuffleData = copy.deepcopy(data)
    # np.random.shuffle(shuffleData)
    # print(shuffleData[140:])    
    # blah = k_nearest_neighbors(shuffleData[:140], shuffleData[140:], 5)
    # print(blah)

    # Test Perceptron    
    w = trainPerceptron(data, .01, 99999)
    testWeights(data, w)

if __name__ == '__main__':
    main()
    
