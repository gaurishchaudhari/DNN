#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 12:18:48 2018

@author: gaurish
"""
import numpy as np
from math_utils import *
from data_utils import load_planar_data
from plot_utils import plot_decision_boundary

def initialize_parameters(layers_dims):
    
    np.random.seed(2)
    parameters = {}
    L  = len(layers_dims)
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layers_dims[l], 1))        
    
    return parameters

def forward_step(A_prev, W, b, activation):
    
    Z = np.dot(W, A_prev) + b
    
    if activation == 'relu':
        A = relu(Z)
    elif activation == 'tanh':
        A = tanh(Z)
    elif activation == 'sigmoid':
        A = sigmoid(Z)
    else:
        raise ValueError('Unknown %s' % (activation))
    
    cache = (A_prev, W, b, Z)
    
    return A, cache 

def forward_propagation(X, parameters):
    
    L = len(parameters) // 2
    A = X
    caches = []
    
    for l in range(L):
        
        A_prev = A
        W = parameters['W' + str(l+1)]
        b = parameters['b' + str(l+1)]
        
        if l < (L - 1):
            A, cache = forward_step(A_prev, W, b, 'tanh')
        elif l == (L - 1):
            A, cache = forward_step(A_prev, W, b, 'sigmoid')
            
        caches.append(cache)
    
    assert(A.shape == (1, X.shape[1]))
    assert(len(caches) == L)
    
    return A, caches

def compute_cost(AL, Y):
    assert(np.any(AL))
    assert(np.any(1-AL))
    
    m = Y.shape[1]
    cost =  -1./m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y,  np.log(1 - AL))) 
    cost = np.squeeze(cost)
    return cost       

def compute_cost_derivative(AL, Y):
    
    return - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

def backward_step(dA, cache, activation):
    
    (A_prev, W, b, Z) = cache
    
    m = A_prev.shape[1]
    
    if activation == 'relu':
        dZ = dA * relu_derivative(Z)
    elif activation == 'tanh':
        dZ = dA * tanh_derivative(Z)
    elif activation == 'sigmoid':
        dZ = dA * sigmoid_derivative(Z)
    else:
        raise ValueError('Unknown %s' % (activation))
    
    dW = 1./m * np.dot(dZ, A_prev.T)
    
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    
    dA_prev = np.dot(W.T, dZ)
    
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)
    
    return (dW, db, dA_prev)
    

def backward_propagation(dAL, caches):
    
    L = len(caches)
    grads = {}
    
    dA_prev = dAL
    
    for l in reversed(range(L)):
        dA = dA_prev
        if l == L - 1:
            dW, db, dA_prev = backward_step(dA, caches[l], 'sigmoid')
        else:
            dW, db, dA_prev = backward_step(dA, caches[l], 'tanh')
            
        grads['dW' + str(l+1)] = dW
        grads['db' + str(l+1)] = db
        
    return grads

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2
    
    for l in range(L):
        
        parameters['W' + str(l+1)] -= learning_rate * grads['dW' + str(l+1)]
        parameters['b' + str(l+1)] -= learning_rate * grads['db' + str(l+1)]
        
    return parameters

def model(X, 
          Y, 
          hidden_layers_dims, 
          learning_rate = 1.2,
          num_iter = 5000,
          print_cost = True):
    
    layers_dims = []
    layers_dims.append(X_train.shape[0])
    layers_dims.extend(hidden_layers_dims)
    layers_dims.append(Y_train.shape[0])
    
    parameters = initialize_parameters(layers_dims)
    
    costs = []
    
    for i in range(num_iter):
        
        AL, caches = forward_propagation(X, parameters)
        
        cost = compute_cost(AL, Y)
        
        dAL = compute_cost_derivative(AL, Y)
        
        grads = backward_propagation(dAL, caches)
        
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
    plot_decision_boundary(lambda x: predict(x.T, parameters), X, Y)
    
    return parameters

def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    predictions = (AL > 0.5)
    return predictions

def evaluate(X, Y, parameters):
    predictions = predict(X, parameters)
    accuracy = (np.dot(Y, predictions.T) + np.dot(1-Y, (1-predictions).T)) * 100.0 / float(Y.shape[1])
    return accuracy
    

if __name__ == '__main__':
    
    X, Y = load_planar_data()
    
    X_train, Y_train, X_test, Y_test = (X, Y, X, Y)
    m = Y_train.shape[1]
    print('X_train shape = ' + str(X_train.shape))
    print('Y_train shape = ' + str(Y_train.shape))
    print('# of Training Examples = %d' % (m))
    parameters = model(X_train, Y_train, [5])
    
    print('Train Accuracy = %f %%' % (evaluate(X_train, Y_train, parameters)))
    print('Test Accuracy = %f %%' % (evaluate(X_test, Y_test, parameters)))

    
    