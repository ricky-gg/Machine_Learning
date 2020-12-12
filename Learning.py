#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 22:44:42 2019

@author: rgarzagi
University of California, Santa Cruz
Earth and Planetary Sciences
Seismology Laboratory
"""

import numpy as np
import activation_functions
import warnings
from functools import partial

class Learning_error(Exception):
    """
    Errors that can come up in the learning part
    """
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return self.value

    def __str__(self):
        return 'Learning_error: ' + self.value
    
#class Learning_warning(Warning):
#    """
#    Warnings that can come up in the learning part
#    """
#    def __init__(self, value):
#        self.value = value
#
#    def __repr__(self):
#        return self.value
#
#    def __str__(self):
#        return 'Learning_WARNING: ' + self.value
#    

def Hebbian(Z,W,alpha=None,max_it=None,activation_function=None,thr=None,method=None):
    '''
    TYPE: UNSUPERVISED; It does not care about a desired behavior, it runs up to a \
    certain amount of iterations
    
    It uses the principles of Hebbian Learning (i.e. associative learning) \
    where the change that happens in the synaptic strength (i.e. the weights) \
    depends on both the presynaptic and postsynaptic neurons being fired up and a learning rate.
    
    “When an axon of cell A is near enough to excite a cell B and repeatedly or \
    persistently  takes  part  in  firing  it,  some  growth  process  or  metabolic  change \
    takes  place  on  one  or  both  cells  such  that  A’s  efficiency  as  one  of  the  cells \
    firing B, is increased.” (Hebb, 1949)
    
    :type Z: list
    :param Z: Input data (usually a matrix) for training
    :type W: list
    :param W: Initial weights of network
    :type max_it: int
    :param max_it: Maximum number of iterations to be computed
    :type alpha: float
    :param alpha: Learning rate|The amount by which the weighs\
        and bias are changed in each step (defaults to 1)
    :type aft: float
    :param aft: Activation Function Threshold
    
    
    Note: This learning type is known to actually happen in the "hippocampus" (a part of the cerebral cortex); \
    neurons show long-term increases in activity with some patterns of stimulation (long-term potentiation)
    '''
    if alpha==None:
        alpha=1
    if max_it==None:
        raise Learning_error('Maximum number of iterations must be specified')
    if activation_function==None:
        raise Learning_error('You must specify the activation function and its parameters')
    activation_function=activation_function.lower()
    if activation_function=='step':
        if thr==None:
            warnings.warn('No threshold is set, the default is thr = 0')
            thr=0
        elif method==None:
            warnings.warn('No method is set, the default is method = binary')
            method='binary'
    #Initiate parameters
    t=1 #Iteration counter
    N=len(Z) #Length of training set
    
    w=W #Start weight matrix
    while t<max_it:
        y=[0]*N #Initiate logit vector
        for i in np.arange(0,N):
            Nz= len(Z[i])#Length of each attribute in training set
            if activation_function=='step' or activation_function=='signum':
                mapfunc = partial(eval(activation_function), thr=thr,method=method)
                u=np.dot(w,Z[i])
                if str(type(u))!="<class 'numpy.ndarray'>":
                    y[i]=mapfunc(u)
                else:
                    y[i]=list(map(mapfunc,u))
            else:
                u=np.dot(w,Z[i])
                if str(type(u))!="<class 'numpy.ndarray'>":
                    u=[u]
                if len(u)==1:
                    u=[u]
                y[i]=list(map(eval(activation_function),u))
            #Generalized Hebb rule
            #W=W+np.matmul(alpha*e[i].reshape(11,1),np.asarray(Z[i]).reshape(1,120))
#            y[i]=np.asarray(y[i]).reshape(N,1)
#            Z[i]=np.asarray(Z[i]).reshape(1,Nz)
            dw=np.matmul(alpha*np.asarray(y[i]).reshape(N,1),np.asarray(Z[i]).reshape(1,Nz))
            w=w+dw
        t=t+1
    return(w,t)
    
def Perceptron():
    '''
    TYPE:
    Rosenblatt (1958)
    '''
    e[i]=d[i]-y[i]
    dw=np.dot(alpha*e[i],Z[i])
    db=alpha*e[i]
    dE=e[i]**2