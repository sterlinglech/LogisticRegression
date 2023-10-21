#!/usr/bin/python3
# Homework 3 Code
import numpy as np
import pandas as pd


def find_binary_error(w, X, y, cutoff=0.5):
    # find_binary_error: compute the binary error of a linear classifier w on data set (X, y)
    # Inputs:
    #        w: weight vector
    #        X: data matrix (without an initial column of 1s)
    #        y: data labels (plus or minus 1)
    # Outputs:
    #        binary_error: binary classification error of w on the data set (X, y)
    #           this should be between 0 and 1.

    # Your code here, assign the proper value to binary_error:

    # Compute the probability using the logistic function
    p = 1 / (1 + np.exp(-np.dot(X, w)))

    # Classify based on the cutoff
    predictions = np.where(p >= cutoff, 1, -1)

    # Compute binary error
    binary_error = np.mean(predictions != y)

    return binary_error


def logistic_reg(X, y, w_init, max_its, eta, grad_threshold):
    # logistic_reg learn logistic regression model using gradient descent
    # Inputs:
    #        X : data matrix (without an initial column of 1s)
    #        y : data labels (plus or minus 1)
    #        w_init: initial value of the w vector (d+1 dimensional)
    #        max_its: maximum number of iterations to run for
    #        eta: learning rate
    #        grad_threshold: one of the terminate conditions;
    #               terminate if the magnitude of every element of gradient is smaller than grad_threshold
    # Outputs:
    #        t : number of iterations gradient descent ran for
    #        w : weight vector
    #        e_in : in-sample error (the cross-entropy error as defined in LFD)

    # Your code here, assign the proper values to t, w, and e_in:
    w = w_init
    t = 0
    while t < max_its:
        z = y * np.dot(X, w)
        gradient = -np.mean((y[:, np.newaxis] * X) / (1 + np.exp(z)[:, np.newaxis]), axis=0)
        w = w - eta * gradient
        if np.all(np.abs(gradient) < grad_threshold):
            break
        e_in = np.mean(np.log(1 + np.exp(-z)))
        t += 1

    return t, w, e_in


def logistic_reg_regularizer(X, y, w_init, max_its, eta, reg, penalty, grad_threshold):
    # logistic_reg learn logistic regression model using gradient descent with L1 or L2 regularizer
    # Inputs:
    #        X : data matrix (without an initial column of 1s)
    #        y : data labels (plus or minus 1)
    #        w_init: initial value of the w vector (d+1 dimensional)
    #        max_its: maximum number of iterations to run for
    #        eta: learning rate
    #        reg: lambda for regularizer
    #        penalty: specify the norm used in the penalization, L1 or L2
    # Outputs:
    #        t : number of iterations gradient descent ran for
    #        w : weight vector
    #        e_in : in-sample error (the cross-entropy error as defined in LFD)

    # Your code here, assign the proper values to t, w, and e_in:
    w = w_init
    t = 0
    e_in = 0

    while t < max_its:
        z = y * np.dot(X, w)

        # Gradient of the loss function
        gradient = -np.mean((y[:, np.newaxis] * X) / (1 + np.exp(z)[:, np.newaxis]), axis=0)

        # add the penalty term
        if penalty == 'L1':
            gradient += reg * np.sign(w)
        elif penalty == 'L2':
            gradient += reg * 2 * w
        else:
            raise ValueError("Invalid penalty type")

        # Update the weights
        w = w - eta * gradient

        # Check termination condition
        if np.all(np.abs(gradient) < grad_threshold):
            break

        # Compute E_in
        e_in = np.mean(np.log(1 + np.exp(-z)))

        t += 1

    return t, w, e_in


def main_cleveland():

    # Load training data
    train_data = pd.read_csv('Data/cleveland_data/clevelandtrain.csv')
    X_train = train_data.drop('heartdisease::category|0|1', axis=1).values
    y_train = train_data['heartdisease::category|0|1'].values

    # Load test data
    test_data = pd.read_csv('Data/cleveland_data/clevelandtest.csv')
    X_test = test_data.drop('heartdisease::category|0|1', axis=1).values
    y_test = test_data['heartdisease::category|0|1'].values

    # Add a column of 1s for the bias term
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    # Convert Labels from 0/1 to -1/+1
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    # Calculate the mean and standard deviation of each feature in the training set
    mean_train = np.mean(X_train, axis=0)
    std_train = np.std(X_train, axis=0)

    # Replace zero with 1 to avoid division by zero
    std_train[std_train == 0] = 1

    # Normalize the features in the training set
    X_train_normalized = (X_train - mean_train) / std_train

    # Normalize the features in the test set using the mean and std from the training set
    X_test_normalized = (X_test - mean_train) / std_train

    # Initialize common parameters
    w_init = np.zeros(X_train_normalized.shape[1])
    grad_threshold = 1e-6  # Terminate learning if the magnitude of every element of the gradient (of E_in) is less than 10^-6
    max_its = 1e4  # 10^4 max iterations per 1c
    eta = 0.01 # Set learning rate per 1c

    # Loop over different lamdas for the regularizers
    for reg in [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]:
        print(f"Training with different lamdas = {reg}")
        for penalty in ['L1', 'L2']:

            # Call logistic_reg
            t, w, e_in = logistic_reg_regularizer(X_train_normalized, y_train, w_init, int(max_its), eta, reg, penalty,
                                                  grad_threshold)

            # Compute binary error on normalized training and test data
            train_error = find_binary_error(w, X_train_normalized, y_train)
            test_error = find_binary_error(w, X_test_normalized, y_test)

            # Count the number of zeros
            count_zeros = np.sum(w == 0)

            # Print results
            print(f"Penalty: {penalty}")
            # print(f"Final weight vector: {w}")
            print(f"Final weight vector Number of 0's: {count_zeros}")
            print(f"Number of iterations: {t}")
            print(f"Lamda: {reg}")
            print(f"In-sample error (E_in): {e_in}")
            print(f"Training binary error: {train_error}")
            print(f"Test binary error: {test_error}")
            print("------")



def main_digits():

    # Load data
    X_train, X_test, y_train, y_test = np.load("Data/digits_data/digits_preprocess.npy", allow_pickle=True)

    # Add a column of 1s for the bias term
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    # Convert Labels from 0/1 to -1/+1
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    # Calculate the mean and standard deviation of each feature in the training set
    mean_train = np.mean(X_train, axis=0)
    std_train = np.std(X_train, axis=0)

    # Replace zero with 1 to avoid division by zero
    std_train[std_train == 0] = 1

    # Normalize the features in the training set
    X_train_normalized = (X_train - mean_train) / std_train

    # Normalize the features in the test set using the mean and std from the training set
    X_test_normalized = (X_test - mean_train) / std_train

    # Initialize common parameters
    w_init = np.zeros(X_train_normalized.shape[1])
    grad_threshold = 1e-6  # Terminate learning if the magnitude of every element of the gradient (of E_in) is less than 10^-6
    max_its = 1e4  # 10^4 max iterations per 1c
    eta = 0.01  # Set learning rate per 1c

    # Loop over different lamdas for the regularizers
    for reg in [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]:
        print(f"Training with different lamdas = {reg}")
        for penalty in ['L1', 'L2']:

            # Call logistic_reg
            t, w, e_in = logistic_reg_regularizer(X_train_normalized, y_train, w_init, int(max_its), eta, reg, penalty,
                                                  grad_threshold)

            # Compute binary error on normalized training and test data
            train_error = find_binary_error(w, X_train_normalized, y_train)
            test_error = find_binary_error(w, X_test_normalized, y_test)

            # Count the number of zeros
            count_zeros = np.sum(w == 0)

            # Print results
            print(f"Penalty: {penalty}")
            #print(f"Final weight vector: {w}")
            print(f"Final weight vector Number of 0's: {count_zeros}")
            print(f"Number of iterations: {t}")
            print(f"Lamda: {reg}")
            print(f"In-sample error (E_in): {e_in}")
            print(f"Training binary error: {train_error}")
            print(f"Test binary error: {test_error}")
            print("------")



if __name__ == "__main__":
    #main_cleveland()
    main_digits()
