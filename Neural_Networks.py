#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 02:30:04 2019

@author: Ricky Garza-Giron
University of California, Santa Cruz
Earth and Planetary Sciences
Seismology Laboratory
"""


"""
Classes of Neural Networks with diferent architectures for different purposes
"""

import activation_functions



Z=np.asarray(list(zip(*DATA_X)))
W=np.random.normal(0,1,(Z.shape[0],Z.shape[1]))
b=b=np.random.normal(0,1,(Z.shape[0]))


def MP_simple_Hebbian(X,d,w,b,max_it=100,alpha=1,aft=0):
    
    """
    Function for a simple NN with one neuron following McCulloch and Pitts(1943) :
        -This is the simplest neural net!
    It uses the principles of Hebbian Learning (i.e. associative learning) \
    where the change that happens in the synaptic strength (i.e. the weights and biases) \
    depends on both the presynaptic and postsynaptic neurons and a learning rate
    
    :type X: list
    :param X: Input data (usually a matrix) for training
    :type d: list
    :param d: Input results for training
    :type w: list
    :param w: Initial weights of network
    :type b: list or float
    :param b: Initial bias(es) for network
    :type max_it: int
    :param max_it: Maxi.mum number of iterations to be computed
    :type alpha: float
    :param alpha: Learning rate|The amount by which the weighs\
        and bias are changed in each step
    :type aft: float
    :param aft: Activation Function Threshold
    """
    
    t=1
    E=1
    N=len(d)
    Z=zip(*X)
    while t<max_it and E>0:
        E=0
        y=[0]*N
        e=[0]*N
        for i in np.arange(0,N):
            y[i]=step(np.dot(w,Z[i]),aft,method='Binary')
            #Generalized Hebb rule
            w=w+dw
#            b=b+alpha*e[i]
            E=E+e[i]**2
        t=t+1
    
    return w,b


def perceptron_simple_Rosenblatt(X,d,w,b,max_it=100,alpha=1,aft=0):
    
    """
    Function for a simple perceptron with one neuron :
        -This is the simplest neural net!
    Nota: EL primero en proponer el "Perceptron" fue Rosenblatt (~1958)
    It uses the principles of Hebbian Learning (i.e. associative learning) \
    where the change that happens in the synaptic strength (i.e. the weights and biases) \
    depends on both the presynaptic and postsynaptic neurons and a learning rate
    
    :type X: list
    :param X: Input data (usually a matrix) for training
    :type d: list
    :param d: Input results for training
    :type w: list
    :param w: Initial weights of network
    :type b: list or float
    :param b: Initial bias(es) for network
    :type max_it: int
    :param max_it: Maxi.mum number of iterations to be computed
    :type alpha: float
    :param alpha: Learning rate|The amount by which the weighs\
        and bias are changed in each step
    :type aft: float
    :param aft: Activation Function Threshold
    """
    
    t=1
    E=1
    N=len(d)
    Z=zip(*X)
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





def perceptron_OLMO_Hebbian(X,d,W,b,max_it=100,alpha=1,aft=0):
    
    """
    Function for a perceptron of one layer with multiple outputs (Single Layer Feedforward Network)
    It uses the principles of Hebbian Learning (i.e. associative learning) \
    where the change that happens in the synaptic strength (i.e. the weights and biases) \
    depends on both the presynaptic and postsynaptic neurons and a learning rate

    :type X: list
    :param X: Input data (usually a matrix) for training
    :type d: list
    :param d: Input results for training
    :type W: list
    :param W: Initial weights of network
    :type b: list or float
    :param b: Initial bias(es) for network
    :type max_it: int
    :param max_it: Maximum number of iterations to be computed
    :type alpha: float
    :param alpha: Learning rate|The amount by which the weighs\
        and biases are changed in each step
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
        u=[0]*N
        for i in np.arange(0,N):
            u[i]=np.dot(W,Z[i])+b
            y[i]=list(map(sigmoid,u[i]))
            dSol=np.asarray([0]*N)
            dSol[i]=1
            e[i]=dSol-y[i]
            #Generalized Hebb rule
            W=W+np.matmul(alpha*e[i].reshape(11,1),np.asarray(Z[i]).reshape(1,120))
            b=b+alpha*e[i]
            E=E+sum(e[i]**2)
            t=t+1

    return W,b,t




def ADALINE(X,d,W,b,max_it=100,alpha=1):
    
    """
    Function for an ADALINE (ADAptive LInear NEuron) network (Widrow and Hoff,1960).
    It uses a Least Mean Squared (LMS) or "delta rule" learning rule, which follows the \
    gradient descent of the squared error, and the linear activation function.
    -It allows the network to keep learning even after an input patter has been learnt
    -Less sensitive to noise
    

    :type X: list
    :param X: Input data (usually a matrix) for training
    :type d: list
    :param d: Input results for training
    :type W: list
    :param W: Initial weights of network
    :type b: list or float
    :param b: Initial bias(es) for network
    :type max_it: int
    :param max_it: Maximum number of iterations to be computed
    :type alpha: float
    :param alpha: Learning rate|The amount by which the weighs\
        and biases are changed in each step
    """
    
    t=1
    E=1
    N=len(d)
    Z=list(zip(*X))
    while t<max_it and E>0:
        E=0
        y=[0]*N
        e=[0]*N
        u=[0]*N
        for i in np.arange(0,N):
            u[i]=np.dot(W,Z[i])+b
            y[i]=list(map(sigmoid,u[i]))
            #Delta Rule (LMS)
            dSol=np.asarray([0]*N)
            dSol[i]=1
            e[i]=dSol-y[i]
            W=W+np.matmul(alpha*e[i].reshape(11,1),np.asarray(Z[i]).reshape(1,120))
            b=b+alpha*e[i]
            E=E+sum(e[i]**2)
            t=t+1

    return W,b,t