#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:16:08 2019

@author: rgarzagi
University of California, Santa Cruz
Earth and Planetary Sciences
Seismology Laboratory
"""


import os,sys
os.chdir('/home/rgarzagi/Machine_Learning/')
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



SET={}

SET['Labels']=['Dove','Hen','Duck','Goose','Owl','Hawk','Eagle','Fox','Dog','Wolf','Cat','Tiger','Lion','Horse','Zebra','Cow']
SET['Data']=[[1,0,0,1,0,0,0,0,1,0,0,1,0],[1,0,0,1,0,0,0,0,1,0,0,0,0],[1,0,0,1,0,0,0,0,1,0,0,0,1],[1,0,0,1,0,0,0,0,1,0,0,1,1],[1,0,0,1,0,0,0,0,1,1,0,1,0],[1,0,0,1,0,0,0,0,1,1,0,1,0],[0,1,0,1,0,0,0,0,1,1,0,1,0],[0,1,0,0,1,1,0,0,0,1,0,0,0],[0,1,0,0,1,1,0,0,0,0,1,0,0],[0,1,0,0,1,1,0,1,0,1,1,0,0],[1,0,0,0,1,1,0,0,0,1,0,0,0],[0,0,1,0,1,1,0,0,0,1,1,0,0],[0,0,1,0,1,1,0,1,0,1,1,0,0],[0,0,1,0,1,1,1,1,0,0,1,0,0],[0,0,1,0,1,1,1,1,0,0,1,0,0],[0,0,1,0,1,1,1,0,0,0,0,0,0]]



N_out=16
N_inputNeurons=13

W=np.random.normal(0,1,(N_out,N_inputNeurons))

#W=np.round(W)


sigma0=16
alpha0=0.1
tao1=1e6
tao2=1e6/np.log10(sigma0)



X=SET['Data']




def SOM_1D(X,W,max_it=1e4,alpha0=None,sigma0=None,tao1=None,tao2=None):
    
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
    
    sigma=sigma0
    alpha=alpha0
    
    N=len(X)
    
    t=1
    
    while t<max_it:
        shuffled_D=np.arange(0,N)
        np.random.shuffle(shuffled_D)
        for i in shuffled_D:
            x=X[i]
            #COMPETITION
            #Determine the winner neuron
            i_winner=np.argmin([np.sqrt(sum(abs(x-w)**2)) for w in W])
            #COOPERATION and ADAPTATION
            #Neighborhood function (depends on distance to winner)
            h=np.exp(-abs(np.arange(0,N_out)[i_winner]-np.arange(0,N_out))**2/(2*sigma**2))
            
            Neighborhood=h.tolist()
            del Neighborhood[i_winner]
            #Update weights based on neibghorhood
            W=W+np.asarray(([diff*(h*alpha)[kk] for kk,diff in enumerate((x-W))]))
            
            if t>=100 and alpha>0.00001:
                alpha=alpha*np.exp(-t/tao1)
#                alpha=alpha*0.3
            if t>=100 and any(Neighborhood):
                sigma=sigma*np.exp(-t/tao2)
#                sigma=sigma*0.3
#                print(t)
            t=t+1
            

    return W,t


W,t = SOM_1D(X,W,max_it=5e3,alpha0=alpha0,sigma0=sigma0,tao1=tao1,tao2=tao2)

for jj in np.arange(0,len(X)):
    x=SET['Data'][jj]
    print('ANIMAL: '+SET['Labels'][jj]+' --> '+ 'Neuron: '+str(np.argmin([np.sqrt(sum(abs(x-w)**2)) for w in W])))
    
    
    