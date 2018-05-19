#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 13:03:11 2018

@author: gaurish
"""

import numpy as np

def relu(Z):
    return np.maximum(Z, 0)

def tanh(Z):
    zp = np.exp(Z)
    zn = np.exp(-Z)
    return np.divide(zp - zn , zp + zn)

def sigmoid(Z):
    return 1.0 / (1 + np.exp(-Z))


def relu_derivative(Z):
    return (Z > 0)

def tanh_derivative(Z):
    t = tanh(Z)
    return 1 - t*t

def sigmoid_derivative(Z):
    s = 1.0 / (1 + np.exp(-Z))
    return s * (1 - s)