# -*- coding: utf-8 -*-
"""
Created on Fri May 29 03:11:23 2020

This script prepares and trains the machine learning model for running
predictions on test dataset.

@author: Tapas Das
"""


import argparse
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LearningRateScheduler
from utils.Load_Data import load_data
from utils.Model import cnn_model
from utils.Epoch_Checkpoint import EpochCheckpoint
from utils.Training_Monitor import TrainingMonitor
from utils.LR_Scheduler import PolynomialDecay
from utils.File_Config import out_npz_file, plotPathLoss, plotPathMetric, jsonPath


# Create placeholders for user input
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--max_iterations", type=int, default=100,
	help="Number of epochs")
ap.add_argument("-b", "--mini_batch_size", type=int, default=32,
	help="Mini Batch Size to be used during training")
ap.add_argument("-c", "--checkpoint", required=True,
	help="Full path for model checkpoint")
ap.add_argument("-f", "--checkpoint_freq", type=int, default=5,
	help="Frequency of model checkpoint")
ap.add_argument("-m", "--model", type=str,
	help="Full path for model HDF5 file")
ap.add_argument("-s", "--start_epoch", type=int, default=0,
	help="Epoch to restart training at")
ap.add_argument("-i", "--init_lr", type=float, default=1e-1,
	help="Initial learning rate")
ap.add_argument("-n", "--new_lr", type=float, default=1e-2,
	help="New learning rate")
ap.add_argument("-l", "--lr_decay_factor", type=float, default=2,
	help="Learning rate decay factor")
args = vars(ap.parse_args())


# Define model hyperparameters
max_iterations = args["max_iterations"]
mini_batch_size = args["mini_batch_size"]
checkpoint = args["checkpoint"]
checkpoint_freq = args["checkpoint_freq"]
model = args["model"]
start_epoch = args["start_epoch"]
init_lr = args["init_lr"]
new_lr = args["new_lr"]
lr_decay_factor = args["lr_decay_factor"]


# Load dataset
dataset = load_data(out_npz_file)
Xtrain = dataset['Xtrain']
Ytrain_oh = dataset['Ytrain_oh']
Xtest = dataset['Xtest']
Ytest_oh = dataset['Ytest_oh']


# Configure data augmentation
datagen = ImageDataGenerator(
    #rotation_range=20,
    #zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    #shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")


# Build and compile the model, if no checkpoint specified
if model is None:
    print("\nCompiling model from scratch")
    img_shape = (Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3])
    opt = Adam(lr=init_lr)
    model = cnn_model(img_shape)
    print("\n\nModel Summary:\n")
    print(model.summary())
    
    model.compile(loss=CategoricalCrossentropy(name='categorical_crossentropy'), 
                  optimizer=opt, 
                  metrics=[CategoricalAccuracy(name='accuracy')])
    
    # Configure the learning rate schedule
    lr_schedule = PolynomialDecay(maxEpochs=max_iterations, 
                                  initAlpha=init_lr, 
                                  power=lr_decay_factor)

else:
    print("\n\nLoading model: {}".format(model))
    model = load_model(model)
    
    print("\nOld learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, new_lr)
    print("New learning rate: {}\n\n".format(K.get_value(model.optimizer.lr)))
    
    # Configure the learning rate schedule
    lr_schedule = PolynomialDecay(maxEpochs=max_iterations, 
                                  initAlpha=new_lr, 
                                  power=lr_decay_factor)


# Configure set of callbacks
callbacks = [
    EpochCheckpoint(checkpoint, every=checkpoint_freq, startAt=start_epoch),
    TrainingMonitor(plotPathLoss, plotPathMetric, jsonPath=jsonPath, startAt=start_epoch),
    LearningRateScheduler(lr_schedule)
]


# Fit the model
model.fit_generator(
    datagen.flow(Xtrain, Ytrain_oh, batch_size=mini_batch_size),
    steps_per_epoch=int(np.ceil(Xtrain.shape[0] / float(mini_batch_size))),
    epochs=max_iterations, callbacks=callbacks,
    validation_data=(Xtest, Ytest_oh)
)


# Get score values on test dataset
scores = model.evaluate(x=Xtest, y=Ytest_oh, verbose=0)
print("\n\nTest Loss: {} \nTest Accuracy: {}".format(scores[0], scores[1]))
