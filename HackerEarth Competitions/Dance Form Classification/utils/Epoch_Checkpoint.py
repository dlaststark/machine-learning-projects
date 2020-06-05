# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:18:44 2020

This script creates a checkpoint for model batch run.

@author: Tapas Das
"""


from tensorflow.keras.callbacks import Callback
import os


class EpochCheckpoint(Callback):
    
    
    def __init__(self, outputPath, every=5, startAt=0):
        """
        Initialize all variables.

        Parameters
        ----------
        outputPath : String
            Full path to save the model checkpoint file.
        every : Integer, optional
            Frequency of checkpoint creation. The default is 5.
        startAt : Integer, optional
            Start value of epoch counter. The default is 0.

        Returns
        -------
        None.

        """

        super(Callback, self).__init__()
        
        self.outputPath = outputPath
        self.every = every
        self.intEpoch = startAt


    def on_epoch_end(self, epoch, logs={}):
        """
        Create model checkpoint file.

        Parameters
        ----------
        epoch : Integer
            Epoch# of the model batch run.
        logs : Dictionary, optional
            Not in use currently. The default is {}.

        Returns
        -------
        None.

        """

        if (self.intEpoch + 1) % self.every == 0:
            p = os.path.sep.join([self.outputPath,
                "epoch_{}.hdf5".format(self.intEpoch + 1)])
            self.model.save(p, overwrite=True)

        self.intEpoch += 1
