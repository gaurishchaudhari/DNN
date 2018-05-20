# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

def load_planar_data():
    
    np.random.seed(1)
    
    m = 400       # number of examples
    N = int(m/2)  # number of points per class
    D = 2         # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4         # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y

def load_noisy_circles_data():
    X, Y = sklearn.datasets.make_circles(n_samples=200, factor=.5, noise=.3)
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    return X, Y

def load_noisy_moons_data():
    X, Y = sklearn.datasets.make_moons(n_samples=200, noise=.2)
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    return X, Y

def load_blobs_data():
    X, Y = sklearn.datasets.make_blobs(n_samples=200, random_state=5, n_features=2, centers=6)
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    Y = Y % 2
    return X, Y
    
def load_guassian_quantiles_data():
    X, Y = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=200, n_features=2, n_classes=2, shuffle=True, random_state=None)
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    return X, Y

def load_conc_circles_data():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y

if __name__ == '__main__':
    
    X, Y, X1, Y1 = load_conc_circles_data()
    
    print('X.shape = ' + str(X.shape))
    print('Y.shape = ' + str(Y.shape))
    
    plt.scatter(X[0, :], X[1, :], c = Y[0, :], s = 40, edgecolors='black', cmap = plt.cm.Spectral)
    
    print('Done!')