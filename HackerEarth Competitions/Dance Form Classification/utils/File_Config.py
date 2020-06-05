# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:19:53 2020

This scripts sets the different file locations for building the machine
learning model.

@author: Tapas Das
"""


# Set file paths for train/predict datasets
train_img_path = "Dataset/train"
predict_img_path = "Dataset/test"
train_true_label = "Dataset/train.csv"
prediction_file = "Dataset/test.csv"


# Set file paths for output files
out_img_path = "Output Images/"
out_npz_file = "NPZ File/Dance_Form_dataset.npz"


# Set file paths for training plot and training history
plotPathLoss = "Output Images/Model_Loss_Curve.png"
plotPathMetric = "Output Images/Model_Accuracy_Curve.png"
jsonPath = "Training History/Dance_Form_Classifier_Model.json"


# Set file path for model checkpoint
model_checkpoint = "Model Checkpoint"
