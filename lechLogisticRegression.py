#!/usr/bin/python3
import numpy as np
import pandas as pd
import time

def find_binary_error(w, X, y):
    # find_binary_error: compute the binary error of a linear classifier w on data set (X, y)
    # Inputs:
    #        w: weight vector
    #        X: data matrix (without an initial column of 1s)
    #        y: data labels (plus or minus 1)
    # Outputs:
    #        binary_error: binary classification error of w on the data set (X, y)
    #           this should be between 0 and 1.

    # Your code here, assign the proper value to binary_error:

    predictions = np.sign(np.dot(X, w))
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

    ## Step 1. Initialization
    # Initialize weight vector
    w = w_init

    # Initialize iteration counter
    t = 0

    # Initialize in-sample error
    e_in = 0

    ## Step 2. Gradient Descent
    while t < max_its:
        # Compute z = y * (X * w)
        z = y * np.dot(X, w)

        # Compute the gradient of E_in
        gradient = -np.mean((y[:, np.newaxis] * X) / (1 + np.exp(z)[:, np.newaxis]), axis=0)



        # Update the weight vector
        w = w - eta * gradient

        # Check the termination condition
        if np.all(np.abs(gradient) < grad_threshold):
            break

        # Compute E_in
        e_in = np.mean(np.log(1 + np.exp(-z)))

        # Increment the iteration counter
        t += 1

    return t, w, e_in

def Run_Logistic_Regression():
    # Load training data
    train_data = pd.read_csv('Data/clevelandtrain.csv')
    X_train = train_data.drop('heartdisease::category|0|1', axis=1).values
    y_train = train_data['heartdisease::category|0|1'].values

    # Load test data
    test_data = pd.read_csv('Data/clevelandtest.csv')
    X_test = test_data.drop('heartdisease::category|0|1', axis=1).values
    y_test = test_data['heartdisease::category|0|1'].values

    # Add a column of 1s for the bias term
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    # Convert Labels from 0/1 to -1/+1
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    # Initialize common parameters
    w_init = np.zeros(X_train.shape[1])
    eta = 1e-5  # Learning rate as per requirement
    grad_threshold = 1e-3  # Gradient threshold as per requirement

    # Loop over different maximum iterations
    for max_its in [1e4, 1e5, 1e6]:
        print(f"Training with max_its = {max_its}")

        start_time = time.time()  # Record the start time

        # Call logistic_reg
        t, w, e_in = logistic_reg(X_train, y_train, w_init, int(max_its), eta, grad_threshold)

        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time

        # Compute binary error on training and test data
        train_error = find_binary_error(w, X_train, y_train)
        test_error = find_binary_error(w, X_test, y_test)

        # Print results
        print(f"Number of iterations: {t}")
        print(f"Final weight vector: {w}")
        print(f"In-sample error (E_in): {e_in}")
        print(f"Training binary error: {train_error}")
        print(f"Test binary error: {test_error}")
        print(f"Time taken for training: {elapsed_time} seconds")
        print("------")

def Run_Logistic_Regression_With_Various_Learning_Rates():

    # Load training data
    train_data = pd.read_csv('Data/clevelandtrain.csv')
    X_train = train_data.drop('heartdisease::category|0|1', axis=1).values
    y_train = train_data['heartdisease::category|0|1'].values

    # Load test data
    test_data = pd.read_csv('Data/clevelandtest.csv')
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
    grad_threshold = 1e-6  # New gradient threshold as per 1b
    max_its = 1e6  # New max iterations as per 1b

    # Loop over different learning rates
    for eta in [0.01, 0.1, 1, 4, 7, 7.5, 7.6, 7.7]:
        print(f"Training with learning rate = {eta}")

        start_time = time.time()  # Record the start time

        # Call logistic_reg
        t, w, e_in = logistic_reg(X_train_normalized, y_train, w_init, int(max_its), eta, grad_threshold)

        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time

        # Compute binary error on normalized training and test data
        train_error = find_binary_error(w, X_train_normalized, y_train)
        test_error = find_binary_error(w, X_test_normalized, y_test)

        # Print results
        print(f"Number of iterations: {t}")
        print(f"Final weight vector: {w}")
        print(f"In-sample error (E_in): {e_in}")
        print(f"Training binary error: {train_error}")
        print(f"Test binary error: {test_error}")
        print(f"Time taken for training: {elapsed_time} seconds")
        print("------")


def main():

    # Run Logistic Regression
    Run_Logistic_Regression()

    # Run using various Learning Rates
    Run_Logistic_Regression_With_Various_Learning_Rates()


if __name__ == "__main__":
    main()
