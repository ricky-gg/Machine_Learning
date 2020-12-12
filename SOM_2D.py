#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 21:38:52 2020

@author: Ricky Garza-Giron
University of California, Santa Cruz
Earth and Planetary Sciences
Seismology Laboratory
"""

'''
CODE
'''


data=[]
data.extend((M0,M1,M2,M3,M4,M5,M6,M7,M8,M9,Mdot)) #List


data=np.asarray(data) #Numpy array

DATA_DIMENSION=len(data.shape)-1 #Dimensionality of data (1D,2D...)
data_shape=data[0].shape

N_initial_training=data.shape[0] #### Number of initial training sets (in this case same as number of classes; one per number)

N_classes=11
    
##Get number of neurons in input layer
N_inputNeurons=reduce(lambda x, y: x*y, [np.asarray(data[0]).shape[i] for i in np.arange(0,DATA_DIMENSION)])


DATA=[]
for k in range(N_initial_training):
    DATA.append(data[k].reshape(N_inputNeurons,1))


#Now lets make some NOISE (ha-ha...: )
training_data_noise=deepcopy(DATA)
for k in range(N_initial_training):
    #NumNoise=np.random.randint(0,N_inputNeurons*.3) #Number of samples to be changed in each matrix, not more than 20% of data
    NumNoise=int(np.round(N_inputNeurons*.05)) # Fixed 5% noise for training data
    Index_change=np.random.randint(0,N_inputNeurons,NumNoise) #Indeces of samples to be changed
    for i in Index_change:
        if training_data_noise[k][i]==0:
            training_data_noise[k][i]=1
        else:
            training_data_noise[k][i]=0




testing_data=deepcopy(DATA)
for k in range(N_initial_training):
    #NumNoise=np.random.randint(0,N_inputNeurons*.3) #Number of samples to be changed in each matrix, not more than 20% of data
    NumNoise=np.random.randint(0,120*.3) #Number of samples to be changed in each matrix, not more than 20% of data
    Index_change=np.random.randint(0,N_inputNeurons,NumNoise) #Indeces of samples to be changed
    for i in Index_change:
        if testing_data[k][i]==0:
            testing_data[k][i]=1
        else:
            testing_data[k][i]=0












N=120 #N09mber of training data vectors (len(X), x E X, x=[x1,x2,x3])
# I=len(X[0])
I=3

X=np.random.normal(0,1,(N,I)) #Nxi, i=len of each vector (e.g. number of pixels)


K=3 #rows in map
L=4 #columns in map


d=np.zeros((K,L)) #map

# d=[range(N_classes)] ##Number of classes
# d=np.asarray(list(chain(d)))

##INITIAL VALUES FOR W AND b
#W=np.random.uniform(0,1,(11,120))

#W=np.random.normal(0,1,(16,120))
#b=np.random.normal(0,1,(16))

#W2=np.random.normal(0,1,(11,16))

W=np.random.normal(0,1,(L,I,K)) #Random weights tensor





# b=np.random.normal(0,1,(N_initial_training,1))

#b=b=np.random.uniform(0,1,(11))
#b2=np.random.normal(0,1,(11))

import joblib
from joblib import Parallel, delayed
import multiprocessing
num_cores = 4


def difference(w,x):
    dist = np.linalg.norm(w-x)
    return dist


#Coordinate vector of each neuron in map
r=[]
for k in range(K):
    for l in range(L):
        r.append(np.array([k,l]))
        

        




for n in range(len(X)):
    distances=[]
    for l in range(L):
        distances.extend(Parallel(n_jobs=num_cores)(delayed(difference)(w,x) for w in W[l]))



index_win=np.argmin(distances)
row=np.ceil(index_win/4)-1
col=abs(4*row-index_win)


# reduce(lambda x, y: x*y, [np.asarray(data[0]).shape[i] for i in np.arange(0,DATA_DIMENSION)])


# def perceptron(X,d,W,b,max_it=100,alpha=1,aft=0):
    
#     """
#     :type X: list
#     :param X: Input data (usually a matrix) for training
#     :type d: list
#     :param d: Input results for training
#     :type w: list
#     :param w: Initial weights of network
#     :type b: list or float
#     :param b: Initial bias(es) for network
#     :type max_it: int
#     :param max_it: Maximum number of iterations to be computed
#     :type alpha: float
#     :param alpha: Learning rate|The amount by which the weighs\
#         and bias are changed in each step
#     :type aft: float
#     :param aft: Activation Function Threshold
#     """
    
#     t=1
#     E=1
#     N=d.shape [1]
#     #Z=list(zip(X))
#     while t<max_it and E>0:
#         E=0
#         y=np.zeros(shape=(N,11,1))
#         #y2=[0]*N
#         e=np.zeros(shape=(N,11,1))
#         #u=[0]*N
#         u=np.zeros(shape=(N,11,1))
#         for i in np.arange(0,N):
#             #y[i]=step(sum(np.dot(W,Z[i])+b),aft,method='Binary')
#     #        u[i]=np.dot(W,Z[i])+b
#             u[i]=np.dot(W,X[i])+b
#             y[i]=list(map(sigmoid,u[i]))
#             #y2[i]=np.asarray(map(sigmoid,np.dot(W2,y[i])+b2))
#             dSol=np.zeros(shape=(N,1))
#             dSol[i]=1
#             e[i]=dSol-y[i]
#     #        W=W+np.matmul(alpha*e[i].reshape(11,1),np.asarray(Z[i]).reshape(1,120))
#             W=W+np.matmul(alpha*e[i].reshape(N,1),np.asarray(X[i]).reshape(1,W.shape[1]))
#             b=b+alpha*e[i]
#             E=E+sum(e[i]**2)
#             t=t+1

#     return W,b,t




# W1,b1,it1=perceptron(DATA,d,W,b,max_it=1e5,alpha=1,aft=0)