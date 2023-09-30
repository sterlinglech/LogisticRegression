#!/usr/bin/python3
import numpy as np
import pandas as pd


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

    return t, w, e_in


def main():
    # Load training data
    train_data = pd.read_csv('clevelandtrain.csv')

    # Load test data
    test_data = pd.read_csv('clevelandtest.csv')

    # Your code here


if __name__ == "__main__":
    main()
