#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 20:45:35 2018

@author: gaurish
"""

import numpy as np
import math

def random_minibatches(X, Y, minibatch_size, seed = 0):
    
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))
    
    num_complete_minibatches = math.floor(m / minibatch_size)
    
    for k in range(0, num_complete_minibatches):
        
        minibatch_X = shuffled_X[:, k * minibatch_size : (k+1) * minibatch_size]
        minibatch_Y = shuffled_Y[:, k * minibatch_size : (k+1) * minibatch_size]
        
        minibatch = (minibatch_X, minibatch_Y)
        mini_batches.append(minibatch)
        
    if m % minibatch_size != 0:
        
        minibatch_X = shuffled_X[:, num_complete_minibatches * minibatch_size: m]
        minibatch_Y = shuffled_Y[:, num_complete_minibatches * minibatch_size: m]            
        minibatch = (minibatch_X, minibatch_Y)
        mini_batches.append(minibatch)
        
    return mini_batches

def parameters_to_vector(parameters):
    
    is_empty = True
    keys = {}
    
    L = len(parameters) // 2
    p = []
    for l in range(L):
        p.append('W' + str(l+1))
        p.append('b' + str(l+1))
    
    for k in p:
        S = parameters[k].reshape(-1, 1)
        if is_empty:
            parameter_values = S
            keys[k] = [0, parameter_values.shape[0], parameters[k].shape]
            is_empty = False
        else:
            s = parameter_values.shape[0]
            parameter_values = np.concatenate((parameter_values, S), axis = 0)
            keys[k] = [s, parameter_values.shape[0], parameters[k].shape]
            
    return parameter_values, keys

def vector_to_parameters(parameter_values, keys):
    
    parameters = {}
    
    for k, v in keys.items():
        parameters[k] = (parameter_values[v[0]:v[1], :]).reshape(v[2])
        
    return parameters

def gradients_to_vector(grads):
    
    is_empty = True
    
    L = len(grads) // 2
    p = []
    for l in range(L):
        p.append('dW' + str(l+1))
        p.append('db' + str(l+1))
    
    for k in p:
        S = grads[k].reshape(-1, 1)
        if is_empty:
            gradients = S
            is_empty = False
        else:
            gradients = np.concatenate((gradients, S), axis = 0)
            
    return gradients

def grad_check(cost_function, parameters, grads, epsilon = 1e-7):
    
    parameter_values, keys = parameters_to_vector(parameters)
    gradients = gradients_to_vector(grads)
    
    num_param_values = parameter_values.shape[0]
    gradapprox = np.zeros((num_param_values, 1))
    
    for i in range(num_param_values):
        
        theta_plus = np.copy(parameter_values)
        theta_plus[i][0] = theta_plus[i][0] + epsilon
        parameters_plus = vector_to_parameters(theta_plus, keys)
        J_plus = cost_function(parameters_plus)
        
        theta_minus = np.copy(parameter_values)
        theta_minus[i][0] = theta_minus[i][0] - epsilon
        parameters_minus = vector_to_parameters(theta_minus, keys)
        J_minus = cost_function(parameters_minus)
        
        gradapprox[i] = (J_plus - J_minus) / (2 * epsilon)
    
    numerator = np.linalg.norm(gradients - gradapprox)
    denominator = np.linalg.norm(gradients) + np.linalg.norm(gradapprox)
    
    diff = numerator / float(denominator)
    
    if np.isnan(diff) or diff > 2e-7:
        print('There is a mistake in backward propagation. Difference = ' + str(diff))
    else:
        print('The backward propagation works perfectly fine. Difference = ' + str(diff))
    
    return diff