#!/usr/bin/env python2
# -*- coding: utf-8 -*-


from __future__ import division



"""
Created on Tue Jan 22 02:36:19 2019

@author: rgarzagi
University of California, Santa Cruz
Earth and Planetary Sciences
Seismology Laboratory
"""

"""
Suite of activation functions for ANNs
Turning input u=Wx+b into a probability
"""

import numpy as np
import matplotlib.pyplot as plt




def linear_activation(u,plot=False):
    
    """
     Activation function step/ defaults to binary method (McCulloch and Pitts neuron)
    :type u: float
    :param u: Linear combination Wx+b (input for postsynaptic neuron)
    :type thr: float
    :param thr: Threshold value
    :type method: str
    :param method: Method of function output: Binary {0,1} or\
        Bipolar {-1,1}
    """
    f=u
    
    if plot==True:
        '''PLOT'''
        x=np.arange(-1,1.1,.01)
        fx=x
        plt.plot(x,fx)
        plt.title('Linear activation function',fontsize=20)
        plt.tick_params('both',labelsize=20)
        plt.yticks(np.arange(min(fx), max(fx), 1.0))
        plt.ylabel('Probability',fontsize=20)  
        plt.xlabel('Input',fontsize=20)
    return(f)
        
    
def step(u,thr=None,method=None,plot=False):
    
    """
     Activation function step (also called threshold) defaults to \
     binary method (McCulloch and Pitts neuron)
    :type u: float
    :param u: Linear combination Wx+b (input for postsynaptic neuron)
    :type thr: float
    :param thr: Threshold value (defaults to 0)
    :type method: str
    :param method: Method of function output: Binary {0,1} or\
        Bipolar {-1,1} (defaults to Binary)
    
    """
    if method==None:
        method='binary'
    method=method.lower()
    if method not in ("binary", "bipolar"):
        raise ValueError("no valid type: %s" % method)
    if thr==None:
        thr=0
    if method=='bipolar':
        if u>=thr:
            f=1
        else:
            f=-1
        if plot==True:
            '''PLOT'''
            x=np.arange(-1,1.1,.01)
            fx=[0]*len(x)
            for i in np.arange(0,len(x)):
                if x[i]==0:
                    fx[i]=0
                elif x[i]>thr:
                    fx[i]=1
                else:
                    fx[i]=-1
            plt.plot(x,fx)
            plt.title('Step activation function with '+method+' method',fontsize=20)
            plt.tick_params('both',labelsize=20)
            plt.yticks(np.arange(min(fx), max(fx)+1, 1.0))
            plt.ylabel('Probability',fontsize=20)  
            plt.xlabel('Input',fontsize=20)
    if method=='binary':
        if u>=thr:
            f=1
        else:
            f=0
        if plot==True:
            '''PLOT'''
            x=np.arange(-1,1.1,.01)
            fx=[0]*len(x)
            for i in np.arange(0,len(x)):
                if x[i]==0:
                    fx[i]=0
                elif x[i]>thr:
                    fx[i]=1
                else:
                    fx[i]=0
            plt.plot(x,fx)
            plt.title('Step activation function with '+method+' method',fontsize=20)
            plt.tick_params('both',labelsize=20)
            plt.yticks(np.arange(min(fx), max(fx)+1, 1.0))
            plt.ylabel('Probability',fontsize=20)  
            plt.xlabel('Input',fontsize=20)
    return(f)


def signum(u,thr=None,method=None,plot=False):
    
    """
     Activation function signum defaults to binary method (McCulloch and Pitts neuron)
     Similar to "Step or threshold function" but f(u)=0 if u=0
    :type u: float
    :param u: Linear combination Wx+b (input for postsynaptic neuron)
    :type thr: float
    :param thr: Threshold value (defaults to 0)
    :type method: str
    :param method: Method of function output: Binary {0,1} or\
        Bipolar {-1,1} (defaults to Binary)
    
    """
    if method==None:
        method='binary'
    method=method.lower()
    if method not in ("binary", "bipolar"):
        raise ValueError("no valid type: %s" % method)
    if thr==None:
        thr=0
    
    if method=='bipolar':
        if u==0:
            f=0
        elif u>thr:
            f=1
        else:
            f=-1
        if plot==True:
            '''PLOT'''
            x=np.arange(-1,1.1,.01)
            fx=[0]*len(x)
            for i in np.arange(0,len(x)):
                if x[i]==0:
                    fx[i]=0
                elif x[i]>thr:
                    fx[i]=1
                else:
                    fx[i]=-1
            plt.plot(x,fx)
            plt.title('Signum activation function with '+method+' method',fontsize=20)
            plt.tick_params('both',labelsize=20)
            plt.yticks(np.arange(min(fx), max(fx)+1, 1.0))
            plt.ylabel('Probability',fontsize=20)  
            plt.xlabel('Input',fontsize=20)
    if method=='binary':
        if u==0:
            f=0
        elif u>thr:
            f=1
        else:
            f=0
        if plot==True:
            '''PLOT'''
            x=np.arange(-1,1.1,.01)
            fx=[0]*len(x)
            for i in np.arange(0,len(x)):
                if x[i]==0:
                    fx[i]=0
                elif x[i]>thr:
                    fx[i]=1
                else:
                    fx[i]=0
            plt.plot(x,fx)
            plt.title('Signum activation function with '+method+' method',fontsize=20)
            plt.tick_params('both',labelsize=20)
            plt.yticks(np.arange(min(fx), max(fx)+1, 1.0))
            plt.ylabel('Probability',fontsize=20)  
            plt.xlabel('Input',fontsize=20)
    return(f)
    
    
def sigmoid(u,B=None,plot=False):
    
    """
     Activation function sigmoid (one of the most used)
    :type u: float
    :param u: Linear combination Wx+b (input for postsynaptic neuron)
    :type B: float
    :param B: Smoothness of curve (defaults to 1)
    
    Note: the larger B, the more similar it gets to the step function and \
    the lower B, the more similar it gets to linear activation
    """
    if B==None:
        B=1
    f=1/(1+np.exp(-B*u))
    if plot==True:
        '''PLOT'''
        plt.plot(np.arange(-10,11,.1),1/(1+np.exp(-B*np.arange(-10,11,.1))))
        plt.title('Sigmoid (logistic) activation function with smoothness= '+str(B),fontsize=20)
        plt.tick_params('both',labelsize=20)
        plt.ylabel('Probability',fontsize=20)  
        plt.xlabel('Input',fontsize=20)  
    return(f)
    

def sigmoid_der(u,B=None,plot=False):
    
    """
    NOTE!: I should instead make each function a Class with their original
    function and their derivative defined (e.g. sigmoid.sigmoid | sigmoid.derivative)
     Derivative of activation function sigmoid (used for backpropagation)
    :type u: float
    :param u: Linear combination Wx+b (input for postsynaptic neuron)
    :type B: float
    :param B: Smoothness of curve (defaults to 1)
    
    Note: the larger B, the more similar it gets to the step function and \
    the lower B, the more similar it gets to linear activation
    """
    if B==None:
        B=1
    f=B*np.exp(-u)/((1+np.exp(-B*u))**2)
    if plot==True:
        '''PLOT'''
        plt.plot(np.arange(-10,11,.1),B*np.exp(-np.arange(-10,11,.1))/(1+np.exp(-B*np.arange(-10,11,.1)))**2)
        plt.title('Derivative of sigmoid (logistic) activation function with smoothness= '+str(B),fontsize=20)
        plt.tick_params('both',labelsize=20)
        plt.ylabel('Probability',fontsize=20)  
        plt.xlabel('Input',fontsize=20)  
    return(f)

 
def RBF(u,B=None,plot=False):
    
    """
     Activation function RBF (Radial Basis Function)
    :type u: float
    :param u: Linear combination Wx+b (input for postsynaptic neuron)
    :type B: float
    :param B: Smoothness of curve (defaults to 1)
    
    Note: the larger B, the more similar it gets to a pulse and \
    the lower B, the more similar it gets to a parabole (or a flat line ultimately)
    """
    if B==None:
        B=1
    f=np.exp(-B*u**2)
    if plot==True:
        '''PLOT'''
        plt.plot(np.arange(-10,11,.1),np.exp(-B*np.arange(-10,11,.1)**2))
        plt.title('Radial Basis Function (Gaussian) activation function with smoothness= '+str(B),fontsize=20)
        plt.tick_params('both',labelsize=20)
        plt.ylabel('Probability',fontsize=20)  
        plt.xlabel('Input',fontsize=20)  
    return(f)


def ReLU(u,thr=None,plot=False):
    
    """
     Activation function ReLU (Rectified Linear Unit)
    :type u: float
    :param u: Linear combination Wx+b (input for postsynaptic neuron)
    :type thr: float
    :param thr: Threshold value (defaults to 0)
    
    """
    if thr==None:
        thr=0
    f=max(thr,u)
    if plot==True:
        '''PLOT'''
        x=np.arange(-1,1.1,.01)
        fx=[0]*len(x)
        for i in np.arange(0,len(x)):
            fx[i]=max(thr,x[i]) 
        plt.plot(x,fx,linewidth=5)
        plt.title('ReLU activation function',fontsize=20)
        plt.tick_params('both',labelsize=20)
        plt.yticks(np.arange(min(fx), max(fx)+1, 1.0))
        plt.ylabel('Probability',fontsize=20)  
        plt.xlabel('Input',fontsize=20)
        plt.ylim([0,1])
    return(f)
    
def ReLU_der(u,thr=None,plot=False):
    
    """
     Activation function ReLU (Rectified Linear Unit)
    :type u: float
    :param u: Linear combination Wx+b (input for postsynaptic neuron)
    :type thr: float
    :param thr: Threshold value (defaults to 0)
    
    """
    if thr==None:
        thr=0
    if u>thr:
        f=1
    else:
        f=0
    if plot==True:
        '''PLOT'''
        x=np.arange(-1,1.1,.01)
        fx=[0]*len(x)
        for i in np.arange(0,len(x)):
            if x[i]>=thr:
                fx[i]=1
            else:
                fx[i]=0
        plt.plot(x,fx,linewidth=5)
        plt.title('Derivative of ReLU activation function',fontsize=20)
        plt.tick_params('both',labelsize=20)
        plt.yticks(np.arange(min(fx), max(fx)+1, 1.0))
        plt.ylabel('Probability',fontsize=20)  
        plt.xlabel('Input',fontsize=20)
        plt.ylim([0,1])
    return(f)