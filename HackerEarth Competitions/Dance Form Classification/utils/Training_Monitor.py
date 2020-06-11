# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:10:26 2020

This script stores and plots the training parameters of Dance Form classifier
model.

@author: Tapas Das
"""


from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class TrainingMonitor(BaseLogger):
    
    
    def __init__(self, figPathLoss, figPathMetric, jsonPath=None, startAt=0):
        """
        Initialize the different variables.

        Parameters
        ----------
        figPathLoss : String
            Full path to save the output training loss figure.
        figPathMetric : String
            Full path to save the output training meetrics figure.
        jsonPath : String, optional
            Full path to save the training parameters. The default is None.
        startAt : Integer, optional
            Starting value of epoch. The default is 0.

        Returns
        -------
        None.

        """
        super(TrainingMonitor, self).__init__()
        self.figPathLoss = figPathLoss
        self.figPathMetric = figPathMetric
        self.jsonPath = jsonPath
        self.startAt = startAt


    def on_train_begin(self, logs={}):
        """
        Store the training parameters in dictionary object.

        Parameters
        ----------
        logs : Dictionary, optional
            Not in use currently. The default is {}.

        Returns
        -------
        None.

        """

        self.H = {}

        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                if self.startAt > 0:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]


    def on_epoch_end(self, epoch, logs={}):
        """
        Plot the learning curves for the machine learning model.

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
        
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(float(v))
            self.H[k] = l

        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()

        if len(self.H["loss"]) > 1:
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure(figsize=(15,8))
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.title("Training Loss [Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid()
            plt.savefig(self.figPathLoss)
            plt.close()
            
            plt.style.use("ggplot")
            plt.figure(figsize=(15,8))
            plt.plot(N, self.H["accuracy"], label="train_accuracy")
            plt.plot(N, self.H["val_accuracy"], label="val_accuracy")
            plt.title("Training Accuracy [Epoch {}]".format(len(self.H["accuracy"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid()
            plt.savefig(self.figPathMetric)
            plt.close()
