#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 00:21:40 2019

@author: Ricky Garza-Giron
University of California, Santa Cruz
Earth and Planetary Sciences
Seismology Laboratory
"""

"""
Code for a simple perceptron with a single neuron which learns
at a certain rate
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from copy import deepcopy
import activation_functions
from activation_functions import *


def simple_perceptron(X,d,w,b,max_it=100,alpha=1,aft=0):
    
    """
    :type X: list
    :param X: Input data (usually a matrix) for training
    :type d: list
    :param d: Input results for training
    :type w: list
    :param w: Initial weights of network
    :type b: list or float
    :param b: Initial bias(es) for network
    :type max_it: int
    :param max_it: Maximum number of iterations to be computed
    :type alpha: float
    :param alpha: Learning rate|The amount by which the weighs\
        and bias are changed in each step
    :type aft: float
    :param aft: Activation Function Threshold
    """
    
    t=1
    E=1
    N=len(d)
    Z=list(zip(*X))
    while t<max_it and E>0:
        E=0
        y=[0]*N
        e=[0]*N
        for i in np.arange(0,N):
            y[i]=step(np.dot(w,Z[i])+b,aft,method='Binary')
            e[i]=d[i]-y[i]
            w=w+np.dot(alpha*e[i],Z[i])
            b=b+alpha*e[i]
            E=E+e[i]**2
        t=t+1
    
    return w,b


## EXAMPLES FOR TESTING OUR SIMPLE PERCEPTRON

##Example 1
    
##Train the perceptron to represent the function AND
X=[[0,0,1,1],[0,1,0,1]] #Data
d=[0,0,0,1] #Results

##Case 1
#Weights and biases initiated as zeros
w=[0,0]
b=0
W1,B1 = simple_perceptron(X,d,w,b,max_it=100,alpha=1,aft=0)

##Case 2
#Weights and biases initiated as random numbers drawn from a uniform distribution.
w=[random.uniform(0,1),random.uniform(0,1)]
b=random.uniform(0,1)
W2,B2 = simple_perceptron(X,d,w,b,max_it=100,alpha=1,aft=0)



Z=list(zip(*X))
print('Simple perceptron results for AND function with initial weights = zero')
print([step(np.dot(W1,Z[i])+B1,0) for i in np.arange(0,len(Z))])
print('Simple perceptron results for AND function with initial weights = random')
print([step(np.dot(W2,Z[i])+B2,0) for i in np.arange(0,len(Z))])
#Example 2

##Train the perceptron to represent the function OR
X=[[0,0,1,1],[0,1,0,1]] #Data
d=[0,1,1,1] #Results

##Case 1
w=[0,0]
b=0
W1,B1 = simple_perceptron(X,d,w,b,max_it=100,alpha=1,aft=0)

##Case 2
w=[random.uniform(0,1),random.uniform(0,1)]
b=random.uniform(0,1)
W2,B2 = simple_perceptron(X,d,w,b,max_it=100,alpha=1,aft=0)

Z=list(zip(*X))
print('Simple perceptron results for OR function with initial weights = zero')
print([step(np.dot(W1,Z[i])+B1,0) for i in np.arange(0,len(Z))])
print('Simple perceptron results for OR function with initial weights = random')
print([step(np.dot(W2,Z[i])+B2,0) for i in np.arange(0,len(Z))])
