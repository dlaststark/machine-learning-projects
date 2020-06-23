# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:57:07 2020

This script plots the Learning Rate curve of Dance Form classifier model.

@author: Tapas Das
"""


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
from utils.Load_Data import load_data
from utils.Learning_Rate_Finder import LearningRateFinder
from utils.Model import cnn_model
from utils.File_Config import out_img_path, out_npz_file


# Load dataset
dataset = load_data(out_npz_file)
Xtrain = dataset['Xtrain']
Ytrain_oh = dataset['Ytrain_oh']


# Build and compile the model
mini_batch_size = 32
img_shape = (Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3])
model = cnn_model(img_shape)
model.compile(loss=categorical_crossentropy, optimizer=Adam(),
              metrics=[CategoricalAccuracy(name='accuracy')])
print("\nModel Summary:\n")
print(model.summary())


# Plot learning rate curve
lrf = LearningRateFinder(model)
lrf.find((Xtrain, Ytrain_oh),
         startLR=1e-10, endLR=1e-0,
         stepsPerEpoch=np.ceil((len(Xtrain) / float(mini_batch_size))),
         batchSize=mini_batch_size, sampleSize=512)
lrf.plot_loss(title="Learning Rate curve of Dance Form classifier")
plt.grid()
plt.savefig(out_img_path+"Model_LR_Curve.png", dpi=300, bbox_inches='tight')
plt.show()
