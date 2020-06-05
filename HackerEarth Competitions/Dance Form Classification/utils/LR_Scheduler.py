# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 01:03:07 2020

This script implements the polynomial decay learning rate scheduler for 
training the machine learning model.

@author: Tapas Das
"""


import matplotlib.pyplot as plt


class LearningRateDecay:
    
    def plot(self, epochs, title="Learning Rate Schedule"):
        """
        Plot the learning rate schedule for the training phase.

        Parameters
        ----------
        epochs : Integer
            Number of epochs for training.
        title : String, optional
            Title of the plot figure. The default is "Learning Rate Schedule".

        Returns
        -------
        None.

        """
        
        lrs = [self(i) for i in epochs]
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs, lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")
        plt.grid()
        plt.show()
        
        
class PolynomialDecay(LearningRateDecay):
    
    def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
        """
        Initialize all the variables.

        Parameters
        ----------
        maxEpochs : Integer, optional
            Number of epochs for training. The default is 100.
        initAlpha : Float, optional
            Initial learning rate value. The default is 0.01.
        power : Float, optional
            Decay factor for the LR scheduler. The default is 1.0.

        Returns
        -------
        None.

        """
        
        self.maxEpochs = maxEpochs
        self.initAlpha = initAlpha
        self.power = power


    def __call__(self, epoch):
        """
        Calculates the updated learning rate using decay factor.

        Parameters
        ----------
        epoch : Integer
            Current epoch#.

        Returns
        -------
        Float
            Updated learning rate value.

        """
        
        decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
        alpha = self.initAlpha * decay
        return float(alpha)
