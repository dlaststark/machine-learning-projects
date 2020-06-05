# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:19:53 2020

This scripts sets the different file locations for building the machine
learning model.

@author: Tapas Das
"""


# Set file paths for train/predict datasets
train_img_path = "/content/drive/My Drive/Colab Notebooks/Identify the dance form/Dataset/train"
predict_img_path = "/content/drive/My Drive/Colab Notebooks/Identify the dance form/Dataset/test"
train_true_label = "/content/drive/My Drive/Colab Notebooks/Identify the dance form/Dataset/train.csv"
prediction_file = "/content/drive/My Drive/Colab Notebooks/Identify the dance form/Dataset/test.csv"


# Set file paths for output files
out_img_path = "/content/drive/My Drive/Colab Notebooks/Identify the dance form/Output Images/"
out_npz_file = "/content/drive/My Drive/Colab Notebooks/Identify the dance form/NPZ File/Dance_Form_dataset.npz"


# Set file paths for training plot and training history
plotPathLoss = "/content/drive/My Drive/Colab Notebooks/Identify the dance form/Output Images/Model_Loss_Curve.png"
plotPathMetric = "/content/drive/My Drive/Colab Notebooks/Identify the dance form/Output Images/Model_Accuracy_Curve.png"
jsonPath = "/content/drive/My Drive/Colab Notebooks/Identify the dance form/Training History/Dance_Form_Classifier_Model.json"


# Set file path for model checkpoint
model_checkpoint = "/content/drive/My Drive/Colab Notebooks/Identify the dance form/Model Checkpoint"
