#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:16:08 2019

@author: rgarzagi
University of California, Santa Cruz
Earth and Planetary Sciences
Seismology Laboratory
"""


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


plt.rc('figure', figsize=(12, 8))
##Make zero
#M0=[[0 for x in range(12)] for y in range(10)]
#M0[1][3:9]=[1]*6
#M0[2][2:10]=[1]*8
#M0[3][1:4]=[1]*3
#M0[3][8:11]=[1]*3
#M0[4][1:4]=[1]*3
#M0[4][8:11]=[1]*3
#M0[5][1:4]=[1]*3
#M0[5][8:11]=[1]*3
#M0[6][1:4]=[1]*3
#M0[6][8:11]=[1]*3
#M0[7][2:10]=[1]*8
#M0[8][3:9]=[1]*6
#M0=np.asarray(M0)
M0=[[0]*12,[0]*3,[1]*6,[0]*3,[0]*2,[1]*8,[0]*2,[0],[1]*3,[0]*4,[1]*3,[0],[0],[1]*3,[0]*4,[1]*3,[0],[0],[1]*3,[0]*4,[1]*3,[0],[0],[1]*3,[0]*4,[1]*3,[0],[0]*2,[1]*8,[0]*2,[0]*3,[1]*6,[0]*3,[0]*12]
M0=np.asarray(list(chain(*M0))).reshape(10,12)

##Make one
M1=[[0]*4,[1]*4,[0]*4]*10
M1=np.asarray(list(chain(*M1))).reshape(10,12)
#Make two
M2=[[0],[1]*9,[0]*2,[0],[1]*9,[0]*2,[0]*8,[1]*2,[0]*2,[0]*8,[1]*2,[0]*2,[0],[1]*9,[0]*2,[0],[1]*9,[0]*2,[0],[1]*3,[0]*8,[0],[1]*3,[0]*8,[0],[1]*9,[0]*2,[0],[1]*9,[0]*2]
M2=np.asarray(list(chain(*M2))).reshape(10,12)
##Make three
M3=[[0]*2,[1]*8,[0]*2,[0]*2,[1]*9,[0]*1,[0]*9,[1]*2,[0]*1,[0]*9,[1]*2,[0]*1,[0]*6,[1]*4,[0]*2,[0]*6,[1]*4,[0]*2,[0]*9,[1]*2,[0]*1,[0]*9,[1]*2,[0]*1,[0]*2,[1]*9,[0]*1,[0]*2,[1]*8,[0]*2]
M3=np.asarray(list(chain(*M3))).reshape(10,12)
#Make four
M4=[[0],[1]*3,[0]*4,[1]*3,[0],[0],[1]*3,[0]*4,[1]*3,[0],[0],[1]*3,[0]*4,[1]*3,[0],[0],[1]*3,[0]*4,[1]*3,[0],[0],[1]*10,[0],[0],[1]*10,[0],[0]*8,[1]*3,[0],[0]*8,[1]*3,[0],[0]*8,[1]*3,[0],[0]*8,[1]*3,[0]]
M4=np.asarray(list(chain(*M4))).reshape(10,12)
#Make five
M5=[[0],[1]*9,[0]*2,[0],[1]*9,[0]*2,[0],[1]*2,[0]*9,[0],[1]*2,[0]*9,[0],[1]*9,[0]*2,[0],[1]*9,[0]*2,[0]*8,[1]*2,[0]*2,[0]*8,[1]*2,[0]*2,[0],[1]*9,[0]*2,[0],[1]*9,[0]*2]
M5=np.asarray(list(chain(*M5))).reshape(10,12)
#Make six
M6=[[0]*3,[1]*6,[0]*3,[0]*3,[1]*2,[0]*7,[0]*3,[1]*2,[0]*7,[0]*3,[1]*2,[0]*7,[0]*3,[1]*2,[0]*7,[0]*3,[1]*6,[0]*3,[0]*3,[1]*6,[0]*3,[0]*3,[1]*2,[0]*2,[1]*2,[0]*3,[0]*3,[1]*2,[0]*2,[1]*2,[0]*3,[0]*3,[1]*6,[0]*3]
M6=np.asarray(list(chain(*M6))).reshape(10,12)
#Make seven
M7=[[0]*3,[1]*6,[0]*3,[0]*3,[1]*6,[0]*3,[0]*7,[1]*2,[0]*3,[0]*7,[1]*2,[0]*3,[0]*7,[1]*2,[0]*3,[0]*7,[1]*2,[0]*3,[0]*7,[1]*2,[0]*3,[0]*7,[1]*2,[0]*3,[0]*7,[1]*2,[0]*3,[0]*7,[1]*2,[0]*3]
M7=np.asarray(list(chain(*M7))).reshape(10,12)
#Make eight
M8=[[0]*2,[1]*8,[0]*2,[0]*1,[1]*10,[0]*1,[0],[1]*2,[0]*6,[1]*2,[0],[0],[1]*2,[0]*6,[1]*2,[0],[0]*2,[1]*8,[0]*2,[0]*2,[1]*8,[0]*2,[0],[1]*2,[0]*6,[1]*2,[0],[0],[1]*2,[0]*6,[1]*2,[0],[0]*2,[1]*9,[0]*1,[0]*2,[1]*8,[0]*2]
M8=np.asarray(list(chain(*M8))).reshape(10,12)
#Make nine
M9=np.rot90(M6,2) #Rotate M6 twice ; )
#Make dot
Mdot=[[[1]*5,[0]*7]*4,[[0]*12]*6]
Mdot=np.asarray(list(chain(*list(chain(*Mdot))))).reshape(10,12)

nrows, ncols = 10, 12

x=np.arange(0,120)
y=np.arange(0,120)

plt.figure('Numbers data')
plt.subplot2grid((1,11), (0,0))
grid=M0
plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest', cmap=cm.gray)
plt.tick_params(axis='both',which='both',left=False, right=False,bottom=False,top=False,
                labelleft=False, labelright=False,labelbottom=False,labeltop=False)
plt.subplot2grid((1,11), (0,1))
grid=M1
plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest', cmap=cm.gray)
plt.tick_params(axis='both',which='both',left=False, right=False,bottom=False,top=False,
                labelleft=False, labelright=False,labelbottom=False,labeltop=False)
plt.subplot2grid((1,11), (0,2))
grid=M2
plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest', cmap=cm.gray)
plt.tick_params(axis='both',which='both',left=False, right=False,bottom=False,top=False,
                labelleft=False, labelright=False,labelbottom=False,labeltop=False)
plt.subplot2grid((1,11), (0,3))
grid=M3
plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest', cmap=cm.gray)
plt.tick_params(axis='both',which='both',left=False, right=False,bottom=False,top=False,
                labelleft=False, labelright=False,labelbottom=False,labeltop=False)
plt.subplot2grid((1,11), (0,4))
grid=M4
plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest', cmap=cm.gray)
plt.tick_params(axis='both',which='both',left=False, right=False,bottom=False,top=False,
                labelleft=False, labelright=False,labelbottom=False,labeltop=False)
plt.subplot2grid((1,11), (0,5))
grid=M5
plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest', cmap=cm.gray)
plt.tick_params(axis='both',which='both',left=False, right=False,bottom=False,top=False,
                labelleft=False, labelright=False,labelbottom=False,labeltop=False)
plt.subplot2grid((1,11), (0,6))
grid=M6
plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest', cmap=cm.gray)
plt.tick_params(axis='both',which='both',left=False, right=False,bottom=False,top=False,
                labelleft=False, labelright=False,labelbottom=False,labeltop=False)
plt.subplot2grid((1,11), (0,7))
grid=M7
plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest', cmap=cm.gray)
plt.tick_params(axis='both',which='both',left=False, right=False,bottom=False,top=False,
                labelleft=False, labelright=False,labelbottom=False,labeltop=False)
plt.subplot2grid((1,11), (0,8))
grid=M8
plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest', cmap=cm.gray)
plt.tick_params(axis='both',which='both',left=False, right=False,bottom=False,top=False,
                labelleft=False, labelright=False,labelbottom=False,labeltop=False)
plt.subplot2grid((1,11), (0,9))
grid=M9
plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest', cmap=cm.gray)
plt.tick_params(axis='both',which='both',left=False, right=False,bottom=False,top=False,
                labelleft=False, labelright=False,labelbottom=False,labeltop=False)
plt.subplot2grid((1,11), (0,10))
grid=Mdot
plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest', cmap=cm.gray)
plt.tick_params(axis='both',which='both',left=False, right=False,bottom=False,top=False,
                labelleft=False, labelright=False,labelbottom=False,labeltop=False)







########################################
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




#nums = {'1':'one',
#        '2':'two',
#        '3':'three',
#        '4':'four',
#        '5':'five',
#        '6':'six',
#        '7':'seven',
#        '8':'eight',
#        '9':'nine'}
#
#plt.figure('Noisy Data')
#
#for k in range(len(data_noise)):
#    plt.subplot2grid((1,11), (0,k))
#    grid=data_noise[k].reshape(10,12)
#    plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
#               interpolation='nearest', cmap=cm.gray)
#    plt.tick_params(axis='both',which='both',left=False, right=False,bottom=False,top=False,
#                    labelleft=False, labelright=False,labelbottom=False,labeltop=False)

#DATA_X=[[0 for x in range(11)] for y in range(120)]
#
#for i in range(len(DATA)):
#    for j in range(120):
#        DATA_X[j][i]=DATA[i][j]
#        
#        
#DATA_X_noise=[[0 for x in range(11)] for y in range(120)]
#for i in range(len(data_noise)):
#    for j in range(120):
#        DATA_X_noise[j][i]=data_noise[i][j]
#Z=zip(*DATA_X)
#d=[range(10),['dot']]
d=[range(N_classes)] ##Number of classes
d=np.asarray(list(chain(d)))

##INITIAL VALUES FOR W AND b
#W=np.random.uniform(0,1,(11,120))

#W=np.random.normal(0,1,(16,120))
#b=np.random.normal(0,1,(16))

#W2=np.random.normal(0,1,(11,16))

W=np.random.normal(0,1,(N_initial_training,N_inputNeurons))
b=np.random.normal(0,1,(N_initial_training,1))

#b=b=np.random.uniform(0,1,(11))
#b2=np.random.normal(0,1,(11))


def perceptron(X,d,W,b,max_it=100,alpha=1,aft=0):
    
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
    N=d.shape [1]
    #Z=list(zip(X))
    while t<max_it and E>0:
        E=0
        y=np.zeros(shape=(N,11,1))
        #y2=[0]*N
        e=np.zeros(shape=(N,11,1))
        #u=[0]*N
        u=np.zeros(shape=(N,11,1))
        for i in np.arange(0,N):
            #y[i]=step(sum(np.dot(W,Z[i])+b),aft,method='Binary')
    #        u[i]=np.dot(W,Z[i])+b
            u[i]=np.dot(W,X[i])+b
            y[i]=list(map(sigmoid,u[i]))
            #y2[i]=np.asarray(map(sigmoid,np.dot(W2,y[i])+b2))
            dSol=np.zeros(shape=(N,1))
            dSol[i]=1
            e[i]=dSol-y[i]
    #        W=W+np.matmul(alpha*e[i].reshape(11,1),np.asarray(Z[i]).reshape(1,120))
            W=W+np.matmul(alpha*e[i].reshape(N,1),np.asarray(X[i]).reshape(1,W.shape[1]))
            b=b+alpha*e[i]
            E=E+sum(e[i]**2)
            t=t+1

    return W,b,t




W1,b1,it1=perceptron(DATA,d,W,b,max_it=1e5,alpha=1,aft=0)
#o=numero de neuronas de salida (i.e. len(d))


#Z=list(zip(*DATA_X_noise))
Z=testing_data



nums = {'0':'zero',
        '1':'one',
        '2':'two',
        '3':'three',
        '4':'four',
        '5':'five',
        '6':'six',
        '7':'seven',
        '8':'eight',
        '9':'nine',
        '10':'dot'}


x_plot=np.arange(0,120)
y_plot=np.arange(0,120)

plt.figure('Noisy data and predictions')

for k in range(len(testing_data)):
    u=np.dot(W1,Z[k])+b1
    y=list(map(sigmoid,u))
    
    if str(np.where(y==max(y))[0][0])=='10':
        print('I think the number is not a number; it is a dot!')
    else:
        print('I think the number is: '+str(np.where(y==max(y))[0][0]))
    plt.subplot2grid((1,11), (0,k))
    grid=testing_data[k].reshape(10,12)
    plt.imshow(grid, extent=(x_plot.min(), x_plot.max(), y_plot.max(), y_plot.min()),
               interpolation='nearest', cmap=cm.gray)
    plt.tick_params(axis='both',which='both',left=False, right=False,bottom=False,top=False,
                    labelleft=False, labelright=False,labelbottom=False,labeltop=False)
    plt.title(nums[str(np.where(y==max(y))[0][0])],FontSize=15)