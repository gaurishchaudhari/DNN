#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 16:53:40 2018

@author: gaurish
"""

import numpy as np
import matplotlib.pyplot as plt
import math_utils as mu
import data_utils as du
from plot_utils import plot_decision_boundary

def initialize_parameters(layers_dims, initialization='he'):
    
    np.random.seed(2)
    parameters = {}
    L  = len(layers_dims)
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        
        if initialization == 'he':
            parameters['W' + str(l)] *= np.sqrt(2. / layers_dims[l-1]) 
        else:
            parameters['W' + str(l)] *= 0.01
        
        
        assert(parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layers_dims[l], 1))        
    
    return parameters

def forward_step(A_prev, W, b, activation, keep_prob):
    
    Z = np.dot(W, A_prev) + b
    
    if activation == 'relu':
        A = mu.relu(Z)
    elif activation == 'tanh':
        A = mu.tanh(Z)
    elif activation == 'sigmoid':
        A = mu.sigmoid(Z)
    else:
        raise ValueError('Unknown %s' % (activation))
    
    # drop-out
    D = np.random.rand(A.shape[0], A.shape[1])
    D = D < keep_prob
    A = A * D
    A = A / keep_prob
    
    assert(D.shape == A.shape)
    
    cache = (A_prev, W, b, Z, D)
    
    return A, cache 

def forward_propagation(X, parameters, hidden_activation, keep_prob):
    
    L = len(parameters) // 2
    A = X
    caches = []
    
    for l in range(L):
        
        A_prev = A
        W = parameters['W' + str(l+1)]
        b = parameters['b' + str(l+1)]
        
        if l < (L - 1):
            A, cache = forward_step(A_prev, W, b, hidden_activation, keep_prob)
        elif l == (L - 1):
            A, cache = forward_step(A_prev, W, b, 'sigmoid', keep_prob = 1.0)
            
        caches.append(cache)
    
    assert(A.shape == (1, X.shape[1]))
    assert(len(caches) == L)
    
    return A, caches

def compute_cost(AL, Y, parameters, lambd):
    assert(np.any(AL))
    assert(np.any(1-AL))
    
    m = Y.shape[1]
    cost =  -1./m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y,  np.log(1 - AL))) 
    cost = np.squeeze(cost)
    
    # L2-regularizer
    L = len(parameters) // 2
    reg = 0
    for l in range(L):
        reg += np.sum(np.square(parameters['W' + str(l+1)]))
    reg = reg * (1./m) * (lambd/2.)
    
    cost = cost + reg
    return cost       

def compute_cost_derivative(AL, Y):
    
    return - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

def backward_step(dA, cache, activation, lambd, keep_prob):
    
    (A_prev, W, b, Z, D) = cache
    
    m = A_prev.shape[1]
    
    # drop-out
    dA = dA * D
    dA = dA / keep_prob
    
    if activation == 'relu':
        dZ = dA * mu.relu_derivative(Z)
    elif activation == 'tanh':
        dZ = dA * mu.tanh_derivative(Z)
    elif activation == 'sigmoid':
        dZ = dA * mu.sigmoid_derivative(Z)
    else:
        raise ValueError('Unknown %s' % (activation))
    
    dW = 1./m * np.dot(dZ, A_prev.T)
    dW += (lambd / m) * W           # L2-regularizer
    
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    
    dA_prev = np.dot(W.T, dZ)
    
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)
    
    return (dW, db, dA_prev)
    

def backward_propagation(dAL, caches, hidden_activation, lambd, keep_prob):
    
    L = len(caches)
    grads = {}
    
    dA_prev = dAL
    
    for l in reversed(range(L)):
        dA = dA_prev
        if l == L - 1:
            dW, db, dA_prev = backward_step(dA, caches[l], 'sigmoid', lambd, keep_prob = 1.0)
        else:
            dW, db, dA_prev = backward_step(dA, caches[l], hidden_activation, lambd, keep_prob)
            
        grads['dW' + str(l+1)] = dW
        grads['db' + str(l+1)] = db
        
    return grads

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2
    
    for l in range(L):
        
        parameters['W' + str(l+1)] -= learning_rate * grads['dW' + str(l+1)]
        parameters['b' + str(l+1)] -= learning_rate * grads['db' + str(l+1)]
        
    return parameters

def forward_prop_and_compute_cost(X, Y, parameters, hidden_activation, lambd, keep_prob):
    AL, _ = forward_propagation(X, parameters, hidden_activation, keep_prob)
    return compute_cost(AL, Y, parameters, lambd)

def model(X, 
          Y, 
          hidden_layers_dims, 
          learning_rate = 1.2,
          num_iter = 5000,
          hidden_activation = 'relu',
          lambd = 0.4,
          keep_prob = 0.99,
          print_cost = True):
    
    layers_dims = []
    layers_dims.append(X_train.shape[0])
    layers_dims.extend(hidden_layers_dims)
    layers_dims.append(Y_train.shape[0])
    
    parameters = initialize_parameters(layers_dims)
    
    costs = []
    
    for i in range(num_iter):
        
        AL, caches = forward_propagation(X, parameters, hidden_activation, keep_prob)
        
        cost = compute_cost(AL, Y, parameters, lambd)
        
        dAL = compute_cost_derivative(AL, Y)
        
        grads = backward_propagation(dAL, caches, hidden_activation, lambd, keep_prob)
        
        if i > 0 and keep_prob == 1.0 and i % 1000 == 0:
            mu.grad_check(lambda params: forward_prop_and_compute_cost(X, Y, params, hidden_activation, lambd, keep_prob), parameters, grads)
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 100 == 0:
            print('Cost after iteration %d: %f' %(i, cost))
            costs.append(cost)
    
    # Plot Learning curve
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('# of iterations (per 100)')
    plt.title('Learning rate = ' + str(learning_rate))  
    plt.show()      
    
    # Plot Decision Boundary
    print('decision boundary')
    plot_decision_boundary(lambda x: predict(x.T, parameters, hidden_activation), X, Y)
    
    return parameters

def predict(X, parameters, hidden_activation):
    AL, _ = forward_propagation(X, parameters, hidden_activation, keep_prob = 1.0)
    predictions = (AL > 0.5)
    return predictions

def evaluate(X, Y, parameters, hidden_activation):
    predictions = predict(X, parameters, hidden_activation)
    accuracy = (np.dot(Y, predictions.T) + np.dot(1-Y, (1-predictions).T)) * 100.0 / float(Y.shape[1])
    return accuracy
    

if __name__ == '__main__':
    
    X_train, Y_train, X_test, Y_test = du.load_conc_circles_data()
    m = Y_train.shape[1]
    print('X_train shape = ' + str(X_train.shape))
    print('Y_train shape = ' + str(Y_train.shape))
    print('# of Training Examples = %d' % (m))
          
    hidden_activation = 'relu'
    parameters = model(X_train, Y_train, 
                       hidden_layers_dims = [10, 5, 3],
                       learning_rate = 0.1,
                       num_iter = 10000,
                       hidden_activation = hidden_activation,
                       lambd = 0.5,
                       keep_prob = 1,
                       print_cost = True)
    
    print('Train Accuracy = %f %%' % (evaluate(X_train, Y_train, parameters, hidden_activation)))
    print('Test Accuracy = %f %%' % (evaluate(X_test, Y_test, parameters, hidden_activation)))

    
    