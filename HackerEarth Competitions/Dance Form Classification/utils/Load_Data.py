# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:46:47 2020

This script loads the training and prediction dataset from NPZ file.

@author: Tapas Das
"""


import numpy as np


def load_data(npz_file):
    """
    Load training and prediction datasets from NPZ file

    Parameters
    ----------
    npz_file : String
        Full path for NPZ file.

    Returns
    -------
    dataset : Dictionary
        Dictionary containing training and prediction datasets.

    """

    processed_dataset = np.load(npz_file, allow_pickle=True)
    
    Xtrain_full, Ytrain_full, Ytrain_full_oh = processed_dataset['Xtrain_full'],\
        processed_dataset['Ytrain_full'], processed_dataset['Ytrain_full_oh']
    Xtrain, Ytrain, Ytrain_oh = processed_dataset['Xtrain'],\
        processed_dataset['Ytrain'], processed_dataset['Ytrain_oh']
    Xtest, Ytest, Ytest_oh = processed_dataset['Xtest'],\
        processed_dataset['Ytest'], processed_dataset['Ytest_oh']
    Xpredict = processed_dataset['Xpredict']
    
    print("\n----------------------- Training Dataset -----------------------")
    print("Xtrain_full shape: {}".format(Xtrain_full.shape))
    print("Ytrain_full shape: {}".format(Ytrain_full.shape))
    print("Ytrain_full_oh shape: {}".format(Ytrain_full_oh.shape))
    print("\nXtrain shape: {}".format(Xtrain.shape))
    print("Ytrain shape: {}".format(Ytrain.shape))
    print("Ytrain_oh shape: {}".format(Ytrain_oh.shape))
    
    print("\n----------------------- Test Dataset -----------------------")
    print("Xtest shape: {}".format(Xtest.shape))
    print("Ytest shape: {}".format(Ytest.shape))
    print("Ytest_oh shape: {}".format(Ytest_oh.shape))
    
    print("\n----------------------- Prediction Dataset -----------------------")
    print("Xpredict shape: {}\n\n".format(Xpredict.shape))
    
    dataset = {}
    dataset['Xtrain_full'] = Xtrain_full
    dataset['Ytrain_full'] = Ytrain_full
    dataset['Ytrain_full_oh'] = Ytrain_full_oh
    dataset['Xtrain'] = Xtrain
    dataset['Ytrain'] = Ytrain
    dataset['Ytrain_oh'] = Ytrain_oh
    dataset['Xtest'] = Xtest
    dataset['Ytest'] = Ytest
    dataset['Ytest_oh'] = Ytest_oh
    dataset['Xpredict'] = Xpredict
    
    return dataset