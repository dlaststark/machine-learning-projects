# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:50:21 2020

This script implements the Cyclic Learning Rate schedule for training the
machine learning model.

@author: Tapas Das
"""


from tensorflow.keras.callbacks import *
import tensorflow.keras.backend as K
import numpy as np


class CyclicLR(Callback):

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        """
        Initialize all variables.

        Parameters
        ----------
        base_lr : Float, optional
            Initial learning rate (lower boundary in the cycle). The default is 0.001.
        max_lr : Float, optional
            Upper boundary in the cycle. The default is 0.006.
        step_size : Integer, optional
            Number of training iterations per half cycle. The default is 2000.
        mode : String, optional
            One of {triangular, triangular2, exp_range}. The default is 'triangular'.
        gamma : Float, optional
            Constant in 'exp_range' scaling function: gamma**(cycle iterations). The default is 1.
        scale_fn : String, optional
            Custom scaling policy defined by a single argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0. The default is None.
        scale_mode : TYPE, optional
            One of {'cycle', 'iterations'}. The default is 'cycle'.

        Returns
        -------
        None.

        """
        
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (1.2 ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
            
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()


    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """
        Resets cycle iterations.

        Parameters
        ----------
        new_base_lr : Float, optional
            New base learning rate. The default is None.
        new_max_lr : Float, optional
            New max learning rate. The default is None.
        new_step_size : Integer, optional
            New step size for training. The default is None.

        Returns
        -------
        None.

        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.


    def clr(self):
        """
        Calculates the modified learning rate value.

        Returns
        -------
        Float
            Modified learning rate.

        """
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)


    def on_train_begin(self, logs={}):
        
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())


    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())
