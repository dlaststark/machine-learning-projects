# -*- coding: utf-8 -*-
"""
Created on Fri May 29 03:11:23 2020

This script plots the learning rate curve for machine learning model.

@author: Tapas Das
"""


from tensorflow.keras.callbacks import LambdaCallback
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile


class LearningRateFinder:
    
    
    def __init__(self, model, stopFactor=4, beta=0.98):
        """
        Initialize all the required variables

        Parameters
        ----------
        model : Model Object
            Keras Model.
        stopFactor : Integer, optional
            Max loss stopping factor. The default is 4.
        beta : Float, optional
            Beta value for smoothing average loss. The default is 0.98.

        Returns
        -------
        None.

        """

        self.model = model
        self.stopFactor = stopFactor
        self.beta = beta

        self.lrs = []
        self.losses = []

        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None


    def reset(self):
        """
        Reinitialize all variables

        Returns
        -------
        None.

        """

        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None


    def is_data_iter(self, data):
        """
        Define list of class types to check for

        Parameters
        ----------
        data : NumPy matrix
            Input data.

        Returns
        -------
        Boolean
            Return whether input data is iterator.

        """

        iterClasses = ["NumpyArrayIterator", "DirectoryIterator",
             "Iterator", "Sequence"]

        return data.__class__.__name__ in iterClasses


    def on_batch_end(self, epoch, logs):
        """
        Increase the learning rate after every epoch

        Parameters
        ----------
        batch : Integer
            Epoch number.
        logs : Dictionary
            Log data after every epoch.

        Returns
        -------
        None.

        """

        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        l = logs["loss"]
        self.batchNum += 1
        self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
        smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
        self.losses.append(smooth)

        stopLoss = self.stopFactor * self.bestLoss

        if self.batchNum > 1 and smooth > stopLoss:
            self.model.stop_training = True
            return

        if self.batchNum == 1 or smooth < self.bestLoss:
            self.bestLoss = smooth

        lr *= self.lrMult
        K.set_value(self.model.optimizer.lr, lr)


    def find(self, trainData, startLR, endLR, epochs=None,
        stepsPerEpoch=None, batchSize=32, sampleSize=2048,
        verbose=1):
        """
        Construct LambdaCallback to update the learning rate after every epoch

        Parameters
        ----------
        trainData : NumPy matrix
            Training dataset.
        startLR : Float
            Starting value of learning rate.
        endLR : Float
            Ending value of learning rate.
        epochs : Float, optional
            Number of epochs to train for. The default is None.
        stepsPerEpoch : Integer, optional
            Number of batch steps in each epoch. The default is None.
        batchSize : Integer, optional
            Mini batch size. The default is 32.
        sampleSize : Integer, optional
            Sample size of training data. The default is 2048.
        verbose : Integer, optional
            Verbosity level. The default is 1.

        Raises
        ------
        an
            useGen and stepsPerEpoch is None.
        Exception
            "Using generator without supplying stepsPerEpoch".

        Returns
        -------
        None.

        """
        
        # Reset class-specific variables
        self.reset()

        # Check if data generator is being used
        useGen = self.is_data_iter(trainData)

        if useGen and stepsPerEpoch is None:
            msg = "Using generator without supplying stepsPerEpoch"
            raise Exception(msg)

        elif not useGen:
            numSamples = len(trainData[0])
            stepsPerEpoch = np.ceil(numSamples / float(batchSize))

        if epochs is None:
            epochs = int(np.ceil(sampleSize / float(stepsPerEpoch)))

        numBatchUpdates = epochs * stepsPerEpoch
        self.lrMult = (endLR / startLR) ** (1.0 / numBatchUpdates)
        self.weightsFile = tempfile.mkstemp()[1]
        self.model.save_weights(self.weightsFile)

        origLR = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, startLR)

        callback = LambdaCallback(on_batch_end=lambda batch, logs:
            self.on_batch_end(batch, logs))

        if useGen:
            self.model.fit_generator(
                trainData,
                steps_per_epoch=stepsPerEpoch,
                epochs=epochs,
                verbose=verbose,
                callbacks=[callback])

        else:
            self.model.fit(
                trainData[0], trainData[1],
                batch_size=batchSize,
                epochs=epochs,
                callbacks=[callback],
                verbose=verbose)

        self.model.load_weights(self.weightsFile)
        K.set_value(self.model.optimizer.lr, origLR)


    def plot_loss(self, skipBegin=10, skipEnd=1, title=""):
        """
        Plot loss values of entire batch run

        Parameters
        ----------
        skipBegin : Integer, optional
            Number of starting points to skip. The default is 10.
        skipEnd : Integer, optional
            Number of ending points to skip. The default is 1.
        title : String, optional
            Title of plotted figure. The default is "".

        Returns
        -------
        None.

        """
        
        # Extract learning rate and losses values to plot
        lrs = self.lrs[skipBegin:-skipEnd]
        losses = self.losses[skipBegin:-skipEnd]

        # Plot learning rate vs loss
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")

        if title != "":
            plt.title(title)
