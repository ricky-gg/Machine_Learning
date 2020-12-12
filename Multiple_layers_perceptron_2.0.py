#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jan 31 17:04:34 2019

@author: Ricky Garza-Giron
University of California, Santa Cruz
Earth and Planetary Sciences
Seismology Laboratory

"""

import os,sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import chain
import copy
from copy import deepcopy
from functools import reduce #Apply function of two arguments cumulatively to the items of sequence, from left to right, so as to reduce the sequence to a single value.
import activation_functions
from activation_functions import *
import Learning
import random


os.chdir('/home/rgarzagi/Machine_Learning/neural-networks-and-deep-learning-master/data')

import pickle
import gzip
import numpy

with open('mnist.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    p = u.load()


##Training data
X=deepcopy(p[0][0])
D=deepcopy(p[0][1])

##Validation data
X_validate=deepcopy(p[1][0])
D_validate=deepcopy(p[1][1])

##Testing data
X_test=deepcopy(p[2][0])
D_test=deepcopy(p[2][1])



class network_architecture(object):
    def __init__(self, N_classes,N_input_neurons,N_hidden_layers,N_neurons_HL,N_subset):
        self.N_classes=N_classes
        self.N_input_neurons=N_input_neurons
        self.N_hidden_layers=N_hidden_layers
        self.N_neurons_HL=N_neurons_HL
        self.N_subset=N_subset


net_arch=network_architecture(N_classes=10,N_hidden_layers=20,N_neurons_HL=[16]*20,N_input_neurons=len(X[0]),N_subset=20)


def MLP(network_architecture,X,D,max_it=100,err_thr=None,alpha=1,aft=0):
    
    """
    Multi-Layer Perceptron
    
    This is a supervised learning algorithm, each input pattern in X must be \
    associated to a class in d.
    
    :type network_architecture: class
    :param network_architecture: design parameters of the ANN
    :type X: list
    :param X: Input data (usually a matrix) for training
    :type d: list
    :param d: Input results for training
    :type w: list
    :param w: Initial weights of network
    :type b: list or float
    :param b: Initial bias(es) for network
    :type max_it: int
    :param max_it: Maximum number of iterations to be computed (defaults to 100)
    :type alpha: float
    :param alpha: Learning rate|The amount by which the weighs\
        and bias are changed in each step
    :type aft: float
    :param aft: Activation Function Threshold
    """


    Nclasses=net_arch.N_classes #Total number of classes
    N_inputNeurons=net_arch.N_input_neurons #Number of neurons in each input pattern (should be the same for all)
    Nhlayers=net_arch.N_hidden_layers #Number of hidden layers
    N_neurons_HL=net_arch.N_neurons_HL #Number of neurons in each hidden layer (list)
    size_subset=net_arch.N_subset #Size of subsets to train network
    
    
    Nneurons=[N_inputNeurons] #Number of neurons in input layer
    for i in range(Nhlayers):
        Nneurons.append(N_neurons_HL[i]) #Number of neurons in each hidden layer
    Nneurons.append(Nclasses) #Number of neurons in output layer
    
    #Initialize the synaptic weights and biases
    W=[0]*(Nhlayers+1)
    b=[0]*(Nhlayers+1)
    for n in np.arange(0,Nhlayers+1):
        W[n]=np.asmatrix([[random.uniform(-1,1) for jj in range(Nneurons[n])] for ii in range(Nneurons[n+1])])
        b[n]=np.asmatrix([random.uniform(-1,1) for ii in range(Nneurons[n+1])]).T
    
    t=1 #Epoch counter (iterations)
    MSE=1 #Start Mean Squared Error
    N=len(X) #Number of patterns in the training set

    while t<=max_it and MSE>err_thr:
        
        #An "epoch" is completed after all training data has been fed to the network
        print('Epoch: '+str(t))
        E=[0 for ii in range(N)]
        u=[[0]*(Nhlayers+1) for ii in range(N)]
        y=[[0]*(Nhlayers+2) for ii in range(N)]
        delta=[[0]*(Nhlayers+1) for ii in range(N)]
        
        #Make random subsets of the training data and feed them until all passed \
        #this should help the gradient descent search for a local minimum
        
        shuffled_D=np.arange(0,N)
        np.random.shuffle(shuffled_D)
        sub_size_index=np.arange(0,N+1,size_subset)
        indx_subset=[shuffled_D[sub_size_index[ii]:sub_size_index[ii+1]] for ii in np.arange(0,len(sub_size_index)-1)]

        for IS in range(int((N/size_subset))):
            for k in range(size_subset):
                
                i=indx_subset[IS][k]
                
                #Get the input layer from pattern i
                y[i][0]=np.asmatrix(X[i]).T
                
                ##Initialize all vectors with zeros
                for n in np.arange(0,Nhlayers+1):
                    u[i][n]=np.asmatrix(np.zeros(Nneurons[n+1])).T
                    y[i][n+1]=np.asmatrix(np.zeros(Nneurons[n+1])).T
                    delta[i][n]=np.asmatrix(np.zeros(Nneurons[n+1])).T
                
                #Calculate inputs and outputs for each layer (feedforward network)
                for n in np.arange(0,Nhlayers+1):
                    N_in=y[i][n]
                    u[i][n]=list(chain(*(np.dot(W[n],N_in)+b[n]).tolist()))
                    y[i][n+1]=np.asmatrix(list(map(ReLU,u[i][n]))).T
                
                #Calculate the synaptic sensitivity (delta) of the last layer
                dSol=np.asmatrix(np.zeros(Nclasses)).T
                cI=D[i] #Get class index
                dSol[cI]=1
                delta[i][Nhlayers]=-2*np.multiply(np.asmatrix(list(map(ReLU_der,u[i][Nhlayers]))).T,dSol-y[i][-1])
                
                #Now we can backpropagate the sensitivities
                M=np.arange(0,Nhlayers)
                for n in M[::-1]:
                     delta[i][n]=np.multiply(np.asmatrix(list(map(ReLU_der,u[i][n]))).T,np.dot(W[n+1].T,delta[i][n+1]))

                #Use each of the sensitivities and the inputs "y" of each layer to adjust W and b
                for n in np.arange(0,Nhlayers+1):
                    W[n]=W[n]-alpha*np.dot(delta[i][n],y[i][n].T)
                    b[n]=b[n]-alpha*delta[i][n]
                    
                #Computation of error for each pattern
                E[i]=float(np.dot((dSol-y[i][-1]).T,dSol-y[i][-1]))
        
        #Calculate the cumulative Mean Squared Error
        MSE=float(sum(E)/N)
        print('MSE= '+str(MSE))
        t=t+1

    return W,b,t,MSE




Wmlp, Bmlp, itmlp, MSEmlp= MLP(net_arch,X,D,max_it=5,err_thr=0.001, alpha=1)



ddd=deepcopy(p[1][1])
TP=0
for xx in np.arange(0,len(ddd)):
    #############################
    Nclasses=net_arch.N_classes #Total number of classes
    N_inputNeurons=net_arch.N_input_neurons #Number of neurons in each input pattern (should be the same for all)
    Nhlayers=net_arch.N_hidden_layers #Number of hidden layers
    N_neurons_HL=net_arch.N_neurons_HL #Number of neurons in each hidden layer (list)
    size_subset=net_arch.N_subset #Size of subsets to train network
    
    Nneurons=[N_inputNeurons] #Number of neurons in input layer
    for i in range(Nhlayers):
        Nneurons.append(N_neurons_HL[i]) #Number of neurons in each hidden layer
    Nneurons.append(Nclasses) #Number of neurons in output layer
    
    y0=np.asmatrix(p[1][0][xx]).T
    
                
    ##Initialize all vectors with zeros
    u=[0]*(Nhlayers+1)
    y=[0]*(Nhlayers+1)
    for n in np.arange(0,Nhlayers+1):
        u[n]=np.asmatrix(np.zeros(Nneurons[n+1]))
        y[n]=np.asmatrix(np.zeros(Nneurons[n+1]))
    
    ##Calculate the first layer output
    u[0]=list(chain(*(np.dot(Wmlp[0],y0)+Bmlp[0]).tolist()))
    y[0]=np.asmatrix(list(map(ReLU,u[0]))).T
    
    #Then the rest
    for n in np.arange(1,Nhlayers+1):
        N_in=y[n-1]
        u[n]=list(chain(*(np.dot(Wmlp[n],N_in)+Bmlp[n]).tolist()))
        y[n]=np.asmatrix(list(map(ReLU,u[n]))).T
    
    if [i for i,y_out in enumerate(list(chain(*(y[-1].tolist())))) if y_out==max(list(chain(*(y[-1].tolist()))))]==ddd[xx]:
        TP=TP+1
#    print('Real result= '+str(ddd[xx]))
#    print(str(np.where(np.asarray(y[-1])==max(y[-1]))[0]))
print("We got %"+str(TP*100/len(ddd))+" of the numbers classified correctly")


