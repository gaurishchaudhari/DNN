#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 21:21:23 2018

@author: gaurish
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import helpers as hp
import data_utils as du
import matplotlib.pyplot as plt
#from plot_utils import plot_decision_boundary

def create_placeholders(n_x, n_y):
    
    X = tf.placeholder(tf.float32, (n_x, None), name = 'X')
    Y = tf.placeholder(tf.float32, (n_y, None), name = 'Y')
    return X, Y
    
def initialize_parameters(layers_dims):
    
    np.random.seed(2)
    parameters = {}
    L  = len(layers_dims)
    
    for l in range(1, L):
        W = tf.get_variable('W' + str(l), (layers_dims[l], layers_dims[l-1]), initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b = tf.get_variable('b' + str(l), (layers_dims[l], 1), initializer = tf.zeros_initializer())
        parameters['W' + str(l)] = W
        parameters['b' + str(l)] = b
        
    return parameters
    
def forward_propagation(X, parameters, hidden_activation):
    
    L = len(parameters) // 2   
    A = X
    
    for l in range(L - 1):
        
        A_prev = A
        W = parameters['W' + str(l+1)]
        b = parameters['b' + str(l+1)]
        Z = tf.add(tf.matmul(W, A_prev), b)
        
        if hidden_activation == 'relu':
            A = tf.nn.relu(Z)
        elif hidden_activation == 'sigmoid':
            A = tf.nn.sigmoid(Z)
        elif hidden_activation == 'tanh':
            A = tf.nn.tanh(Z)
        else:
            raise ValueError('Invalid value for hidden_activation ' + hidden_activation)

    A_prev = A
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    Z = tf.add(tf.matmul(W, A_prev), b)
    
    return Z
    
def compute_cost(ZL, Y):
    
    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels)) 
    return cost

def model(X_arr, Y_arr,
          hidden_layers_dims,
          learning_rate,
          num_epochs,
          minibatch_size,
          hidden_activation = 'relu',
          print_cost = False,
          show_plot = False):
    
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 10
    costs = []
    
    (n_x, m) = X_arr.shape
    n_y = Y_arr.shape[0]
    
    layers_dims = []
    layers_dims.append(n_x)
    layers_dims.extend(hidden_layers_dims)
    layers_dims.append(n_y)
    
    X, Y = create_placeholders(n_x, n_y)
    
    parameters = initialize_parameters(layers_dims)
    
    ZL = forward_propagation(X, parameters, hidden_activation)
    
    cost = compute_cost(ZL, Y)
            
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        for i in range(num_epochs):
            
            epoch_cost = 0
            seed = seed + 1
            minibatches = hp.random_minibatches(X_arr, Y_arr, minibatch_size, seed)
            
            for minibatch in minibatches:
                
                (minibatch_X, minibatch_Y) = minibatch
            
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost += minibatch_cost
                
            epoch_cost /= minibatch_size
            
            if print_cost and i % 100 == 0:
                print('Cost after epoch %d: %f' %(i, epoch_cost))
                costs.append(epoch_cost)
        
        parameters_out = sess.run(parameters)
        
        if show_plot:    
            plt.plot(costs)
            plt.ylabel('Cost')
            plt.xlabel('# of iterations (per 100)')
            plt.title('Learning rate = ' + str(learning_rate))  
            plt.show()      
            
            # Plot Decision Boundary
            #print('Decision Boundary')
            #plot_decision_boundary(lambda x: predict_plot(x, parameters, hidden_activation), X_arr, Y_arr)
            pass

    return parameters_out
                    

def predict(X, parameters, hidden_activation):
    
    ZL = forward_propagation(X, parameters, hidden_activation)
    AL = tf.nn.sigmoid(ZL)
    return AL, tf.greater(AL, 0.5)

def evaluate(X_arr, Y_arr, parameters, hidden_activation):
    
    (n_x, m) = X_arr.shape
    n_y = Y_arr.shape[0]
    
    X, Y = create_placeholders(n_x, n_y)
    
    AL, predictions = predict(X, parameters, hidden_activation)
    labels = tf.equal(Y, 1)
    correct_predictions = tf.equal(predictions, labels)
    
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float32')) * 100.0
    
    with tf.Session() as sess:
        return accuracy.eval({X: X_arr, Y: Y_arr})    
        
    
if __name__ == '__main__':
    
    X_train, Y_train, X_test, Y_test = du.load_conc_circles_data()
    
    m = Y_train.shape[1]
    print('X_train shape = ' + str(X_train.shape))
    print('Y_train shape = ' + str(Y_train.shape))
    print('# of Training Examples = %d' % (m))
          
    hidden_activation = 'relu'
    parameters = model(X_train, Y_train, 
                       hidden_layers_dims = [10, 5, 3],
                       learning_rate = 0.008,
                       num_epochs = 2000,
                       minibatch_size = 32,
                       hidden_activation = hidden_activation,
                       print_cost = True,
                       show_plot = True)
    
    print('Train Accuracy = %f %%' % (evaluate(X_train, Y_train, parameters, hidden_activation)))
    print('Test Accuracy = %f %%' % (evaluate(X_test, Y_test, parameters, hidden_activation)))
    