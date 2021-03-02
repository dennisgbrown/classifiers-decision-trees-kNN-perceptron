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

    # Second pass: read data into array
    data = np.zeros((num_samples, max_feature_id + 1))
    curr_sample = 0
    datafile.seek(0)
    for line in datafile:
        tokens = line.split()
        data[curr_sample][0] = float(tokens[0])
        for feature in tokens[1:]:
            # print(token)
            feature_id = int(feature.split(':')[0])
            feature_val = float(feature.split(':')[1])
            data[curr_sample][feature_id] = feature_val
        curr_sample += 1
    datafile.close()

    print(filename, ':', data.shape)

    return data


def get_neighbors(data, test_sample, num_neighbors):
    """
    Given training data, a test sample, and a number of
    neighbors, return the closest neighbors.
    """
    # Calculate all distances from the training samples
    # to this test sample. Collect index, distance into a list.
    indices_and_distances = list()
    for i in range(len(data)):
        dist = np.linalg.norm(test_sample[1:] - (data[i])[1:]) # leave out classification at pos 0
        indices_and_distances.append([i, dist])

    # Sort list by distance
    indices_and_distances.sort(key=lambda _: _[1])

    # Make a list of requested number of closest neighbors from sorted
    # list of indices+distances
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(indices_and_distances[i][0])

    return neighbors


def classify_one_sample(data, test_sample, num_neighbors):
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
    classifications = list()
    for i in range(len(test_samples)):
        output = classify_one_sample(data, test_samples[i], num_neighbors)
        classifications.append(output)
        if ((i % 20) == 0): 
            print('\rknn test sample', i, end='')
    print()
    return(classifications)


def check_knn_classifications(y, y_hat):
    misclassifications = 0
    for i in range(len(y)):
        if (y[i] != y_hat[i]):
            misclassifications += 1
    print(misclassifications, 'errors in', len(y), 'samples')


def train_perceptron(data, beta, step_limit):
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
    y = data[:,0]

    # Repeat the main loop until we have convergence or reach the
    # iteration limit
    steps = 0
    converged = False
    while(not(converged) and (steps < step_limit)):
        converged = True

        # For each sample in the data, calculate w's classification error
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


def test_weights(data, w):
    """
    Given test data and a weight vector w, determine the error rate
    when classifying the test data using the weights. 
    """
    misclassifications = 0

    # Initialize y_hat
    y_hat = np.zeros(len(data))

    # Slice off y
    y = data[:,0]
    
    # Determine how weights classify each test sample and count
    # misclassifications
    for i in range(len(data)):
        biased_sample = np.copy(data[i])
        biased_sample[0] = 1
        y_hat[i] = 1 if (np.matmul(w.T, biased_sample) > 0) else -1
        if (y[i] != y_hat[i]):
            misclassifications += 1        

    print(misclassifications, 'errors in', len(y), 'samples')
    # print('y   :', y)
    # print('y^  :', y_hat)
    # print('Diff:', y - y_hat)
    # print('sum Diff', sum(y - y_hat))


def iris_knn(num_neighbors):
    """
    Run kNN on the iris dataset for the given number of neighbors.
    """
    # Load data
    iris_data = libsvm_scale_import('data/iris.scale')
    
    # Shuffle the data because otherwise we can't effectively split it
    # into training & test 
    shuffle_data = copy.deepcopy(iris_data)
    np.random.seed(1) # ensure consistent shuffling 
    np.random.shuffle(shuffle_data)
    
    # Split up data into training and test data based on split value
    split = 100
    train_data = shuffle_data[:split]
    test_data = shuffle_data[split:]

    # Classify the test data    
    classifications = k_nearest_neighbors(train_data, test_data, num_neighbors)
    
    # Check accuracy
    check_knn_classifications(test_data[:,0], classifications)
    

def iris_perceptron():
    """
    Run Perceptron on the iris dataset for the given number of neighbors.
    """
    # Load data
    iris_data = libsvm_scale_import('data/iris.scale')
    
    # Pass 1: Classify all the data into "1" vs "2 and 3" by setting
    # all "2" and "3" classifications to "-1"
    pass1_data = copy.deepcopy(iris_data)
    for i in range(len(pass1_data)):
        if (pass1_data[i][0] != 1):
            pass1_data[i][0] = -1    
    w = train_perceptron(pass1_data, 0.01, 999)
    test_weights(pass1_data, w)

    # Pass 2: Classify the "2 and 3" data into "2" vs "3"
    # First remove all "1" samples
    pass2_data = copy.deepcopy(iris_data[np.where(iris_data[:,0] > 1)])
    # Next, set all "2" samples to "1" and "3" to "-1"    
    for i in range(len(pass2_data)):
        if (pass2_data[i][0] == 2):
            pass2_data[i][0] = 1
        elif (pass2_data[i][0] == 3):
            pass2_data[i][0] = -1
    w = train_perceptron(pass2_data, 0.01, 999)
    test_weights(pass2_data, w)


def a4a_knn(num_neighbors):
    """
    Run kNN on the a4a dataset for the given number of neighbors.
    """

    # Load data
    train_data = libsvm_scale_import('data/a4a')    
    test_data = libsvm_scale_import('data/a4a.t')

    # Subsample test data because it's huge
    test_data = test_data[::100]

    # Training data has 1 fewer feature than test data, so add a column
    # of zeros to it so samples have same number of features in train and test
    zero_col = np.zeros((len(train_data), 1))
    train_data = np.hstack((train_data, zero_col))

    # Classify the test data    
    classifications = k_nearest_neighbors(train_data, test_data, num_neighbors)

    # Check accuracy
    check_knn_classifications(test_data[:,0], classifications)


def a4a_perceptron():
    """
    Run Perceptron on the a4a dataset for the given number of neighbors.
    """
    # Load data
    train_data = libsvm_scale_import('data/a4a')    
    test_data = libsvm_scale_import('data/a4a.t')

    # Training data has 1 fewer feature than test data, so add a column
    # of zeros to it so samples have same number of features in train and test
    zero_col = np.zeros((len(train_data), 1))
    train_data = np.hstack((train_data, zero_col))

    w = train_perceptron(train_data, 0.01, 999)
    test_weights(test_data, w)
    

def main():

    # iris_knn(5)    

    iris_perceptron()

    # a4a_knn(5)
    
    # a4a_perceptron()


if __name__ == '__main__':
    main()

