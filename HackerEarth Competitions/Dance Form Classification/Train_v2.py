# -*- coding: utf-8 -*-
"""
Created on Fri May 29 03:11:23 2020

This script prepares and trains the machine learning model for running
predictions on test dataset.

@author: Tapas Das
"""


import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from utils.Load_Data import load_data
from utils.Model import cnn_model
from utils.Cyclic_LR_Scheduler import CyclicLR
from utils.Training_Monitor import TrainingMonitor
from utils.File_Config import out_npz_file, out_img_path
from utils.File_Config import plotPathLoss, plotPathMetric, jsonPath


# Create placeholders for user input
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--max_iterations", type=int, default=100,
	help="Number of epochs")
ap.add_argument("-b", "--mini_batch_size", type=int, default=32,
	help="Mini Batch Size to be used during training")
ap.add_argument("-llr", "--min_lr", type=float, default=1e-5,
	help="Min learning rate")
ap.add_argument("-hlr", "--max_lr", type=float, default=1e-1,
	help="Max learning rate")
ap.add_argument("-c", "--checkpoint", required=True,
	help="Full path for model checkpoint")
ap.add_argument("-s", "--start_epoch", type=int, default=0,
	help="Epoch to restart training at")
args = vars(ap.parse_args())


# Define model hyperparameters
max_iterations = args["max_iterations"]
mini_batch_size = args["mini_batch_size"]
min_lr = args["min_lr"]
max_lr = args["max_lr"]
checkpoint = args["checkpoint"]
start_epoch = args["start_epoch"]


# Load dataset
dataset = load_data(out_npz_file)
Xtrain = dataset['Xtrain']
Ytrain_oh = dataset['Ytrain_oh']
Xtest = dataset['Xtest']
Ytest_oh = dataset['Ytest_oh']


# Configure data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    shear_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    vertical_flip=True,
    fill_mode="nearest")


# Build and compile the model, if no checkpoint specified
print("\nCompiling model from scratch")
img_shape = (Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3])
model = cnn_model(img_shape)
print("\n\nModel Summary:\n")
print(model.summary())

model.compile(loss=CategoricalCrossentropy(name='categorical_crossentropy'),
              optimizer=Adam(lr=min_lr),
              metrics=[CategoricalAccuracy(name='accuracy'),
			  		   Precision(name='precision'), Recall(name='recall')])


# Configure the learning rate schedule
clr_method = 'triangular2'
step_size = 4 * (Xtrain.shape[0] // mini_batch_size)
clr = CyclicLR(base_lr=min_lr, max_lr=max_lr,
               mode=clr_method, step_size=step_size)


# Configure set of callbacks
callbacks = [
    TrainingMonitor(plotPathLoss, plotPathMetric,
                    jsonPath=jsonPath, startAt=start_epoch),
    clr
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
print("\n\nTest Loss: {} \nTest Accuracy: {}\n\n".format(scores[0], scores[1]))


# Plot learning rate schedule
plt.plot(clr.history["lr"])
plt.ylabel('Learning Rate')
plt.xlabel('Iteration #')
plt.title("Cyclical Learning Rate (CLR)")
plt.grid()
plt.savefig(out_img_path+"Model_LR_Schedule.png", dpi=300, bbox_inches='tight')
plt.show()


# Save the model
model.save(checkpoint + 'Dance_Form_Classifier_Model.hdf5', overwrite=True)
