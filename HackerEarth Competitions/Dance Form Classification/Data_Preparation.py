# -*- coding: utf-8 -*-
"""
Created on Fri May 29 01:14:37 2020

This script prepares the data for training the machine learning model and run
predictions on test dataset.

Steps:
    1) Read train/predict images and prepare individual feature matrices
    2) Read true labels for train images and prepare target label vector
    3) Split training data into train/test datasets
    4) Convert true labels into one-hot encoding
    5) Save all matrices in NPZ file for re-usability

@author: Tapas Das
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.utils import to_categorical
from utils.Create_Image_Batch import create_image_batch
from utils.File_Config import train_img_path, predict_img_path
from utils.File_Config import train_true_label, prediction_file
from utils.File_Config import out_img_path, out_npz_file


'''
    Prepare training data for machine learning model
'''

# Extract image names and true labels from training file
train_true_label_df = pd.read_csv(train_true_label)
print("\nSample data from training dataset:\n")
print(train_true_label_df.head())


# Convert categorical classes into numerical representations
train_true_label_df['target_enc'] = train_true_label_df['target'].factorize()[0]
print("\nCount of different target labels:\n")
print(train_true_label_df.groupby(['target','target_enc']).size()
      .reset_index().rename(columns={0:'Count'}))


# Display countplot for different classes in training data
sns.set(style="darkgrid")
sns.countplot(x="target_enc", data=train_true_label_df)
plt.xlabel("Target Class")
plt.ylabel("Count")
plt.title("Countplot of different classes in training data")
plt.savefig(out_img_path+'train_labels_countplot.png',
            dpi=300, bbox_inches='tight')
plt.show()


# Create feature matrix for training images
Xtrain = create_image_batch(train_true_label_df, train_img_path)
Xtrain = np.array(Xtrain, dtype="float") / 255.0
print("\nXtrain shape: {}".format(Xtrain.shape))


# Create true label vector for training images
Ytrain = np.array([train_true_label_df['target_enc'].to_numpy()]).T
print("\nYtrain shape: {}".format(Ytrain.shape))


'''
    Prepare prediction data for machine learning model
'''

# Extract image names from prediction file
prediction_df = pd.read_csv(prediction_file)
print("\nSample data from prediction dataset:\n")
print(prediction_df.head())

# Create feature matrix for predict images
Xpredict = create_image_batch(prediction_df, predict_img_path)
Xpredict = np.array(Xpredict, dtype="float") / 255.0
print("\nXpredict shape: {}".format(Xpredict.shape))


'''
    Split training data into train/test datasets
'''

# Create copy of entire training data before splitting
Xtrain_full = Xtrain.copy()
Ytrain_full = Ytrain.copy()

# Stratified splitting of training data
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=1)
for train_index, test_index in sss.split(Xtrain, Ytrain):
    train_x, test_x = Xtrain[train_index], Xtrain[test_index]
    train_y, test_y = Ytrain[train_index], Ytrain[test_index]

print("\n\n----------------------- Training Dataset -----------------------")
print("train_x shape: {}".format(train_x.shape))
print("train_y shape: {}".format(train_y.shape))

print("\n\n----------------------- Test Dataset -----------------------")
print("test_x shape: {}".format(test_x.shape))
print("test_y shape: {}".format(test_y.shape))


'''
    Convert true labels vector to one-hot encoding
'''

train_y_oh = to_categorical(train_y, 8)
test_y_oh = to_categorical(test_y, 8)
Ytrain_full_oh = to_categorical(Ytrain_full, 8)
print("\n\nTrue label matrices shape after one-hot encoding:\n")
print("train_y_oh shape: {}".format(train_y_oh.shape))
print("test_y_oh shape: {}".format(test_y_oh.shape))
print("Ytrain_full_oh shape: {}".format(Ytrain_full_oh.shape))


'''
    Save dataset in NPZ file for re-usability
'''

np.savez_compressed(out_npz_file,
                    Xtrain_full=Xtrain_full, Ytrain_full=Ytrain_full,
                    Ytrain_full_oh=Ytrain_full_oh,
                    Xtrain=train_x, Ytrain=train_y, Ytrain_oh=train_y_oh,
                    Xtest=test_x, Ytest=test_y, Ytest_oh=test_y_oh,
                    Xpredict=Xpredict)

print("\n\nData saved in NPZ file.")
