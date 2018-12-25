# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:24:31 2018

@author: Tran Bao Loc
"""

import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE
    s = 1 / (1 + np.exp(-x))
    ### END YOUR CODE

    return s


def sigmoid_grad(s):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input s should be the sigmoid
    function value of your original input x.

    Arguments:
    s -- A scalar or numpy array.

    Return:
    ds -- Your computed gradient.
    """

    ### YOUR CODE HERE
    ds = s - s*s
    ### END YOUR CODE

    return ds

def affine_forward(X, W, b=None):
    """
    Input : 
        X shape (m, n) : m number of example and n is the number of features
        W shape (n, o) : o is the number of output in the next layer
        b shape (o,)   : bias term
    Output:
        out shape (m, o) : output for next layer
        cache          : stuff needed for backward pass
    """
    n, o = W.shape
    if b is None:
        b = np.zeros(shape=(o,))
    out = np.dot(X, W) + b
    cache = (X, W)
    
    return out, cache

def affine_backward(dout, cache):
    """
    Input :
        dout shape (m, o) : gradient of the loss with respect to the current affine layer
        cache type tuple  : stuff needed for backward pass
    Output :
        dX shape (m, n)   : gradient of the loss with respect to the input of the previous layer
        dW shape (n, o)   : gradient of the loss with respect to the weights of the current layer
        db shape (o, )    : gradient of the loss with respect to the bias of the current layer
    """
    X, W = cache
    n, o = W.shape
    dX = np.dot(dout, W.T)
    dW = np.dot(X.T, dout)
    db = np.sum(dout, axis=0)
    
    return dX, dW, db



def sigmoid_forward(X):
    """
    Compute sigmoid element-wise on X
    """
    s =  1 / (1 + np.exp(-X))
    cache = (s)
    
    return s, cache

def sigmoid_backward(dout, cache):
    """
    
    """
    s = cache
    
    return dout*(s - s*s)

def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    D-dimensional vector (treat the vector as a single row) and
    for N x D matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        x = x - np.max(x, axis=1, keepdims=True)
        x = np.exp(x)
        x = x / np.sum(x, axis=1, keepdims=True)
        ### END YOUR CODE
    else:
        # Vector
        x = x - np.max(x)
        x = np.exp(x)
        x = x / np.sum(x)
        ### END YOUR CODE

    assert x.shape == orig_shape
    return x


def cross_entropy_forward(X, labels):
    """
    Input:
        X shape (m, n) : Output layer of neural network 
        with m is the number of examples and n is the number of node in the output layer
    Output:
        cost (scalar) : value of cross entropy loss function
        Cache : stuff needed for backward pass
    """

    s = softmax(X)
    cost = np.sum(-labels*np.log(s))
    cache = (s, labels)
    
    return cost, cache
    

def cross_entropy_backward(cache):
    """
    """
    s, labels = cache
    
    return s - labels
            

def neg_sampling_forward(X, counts):
    """
    """
#    sampling = np.zeros_like(X)
#    unique, counts = np.unique(indices, return_counts=True)
#    sampling[:,unique] = counts

    S = sigmoid(X)
    cost = np.sum(-np.log(S)*counts)
    cache = (S, counts)

    return cost, cache


def neg_sampling_backward(cache):
    
    S, counts = cache
    return counts*(S-1)