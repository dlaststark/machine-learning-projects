#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to train the machine learning model for Programming Language Detection

Created on Mon Nov  2 17:40:38 2020

@author: tapasdas
"""


import argparse
import itertools
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils  import plot_model
from tensorflow_addons.optimizers import AdamW, Lookahead
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils.preprocess_data import preprocess_data
from utils.model import prog_lang_detect_model
from utils.params import artifacts_path, embed_models_fl, npz_fl


# Create placeholders for user input
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset_path", required=True,
	help="Full path for source dataset")
ap.add_argument("-e", "--max_iterations", type=int, default=512,
	help="Number of epochs")
ap.add_argument("-b", "--mini_batch_size", type=int, default=64,
	help="Mini Batch Size to be used during training")
ap.add_argument("-m", "--model_save_path", required=True,
	help="Full path for saving the trained model")
args = vars(ap.parse_args())

dataset_path = args["dataset_path"]
max_iterations = args["max_iterations"]
mini_batch_size = args["mini_batch_size"]
model_save_path = args["model_save_path"]


def plot_confusion_matrix(cm, classes):
    """
    Function to plot the confusion matrix for predictions made by model

    Parameters
    ----------
    cm : Numpy array
        Confusion matrix calculated.
    classes : list
        List of target classes.

    Returns
    -------
    None.

    """
    
    plt.figure(figsize=(15,12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(artifacts_path + '/confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def learning_curves(hist):
    """
    Function to plot the learning curves for model training

    Parameters
    ----------
    hist : Keras History object
        Object containing loss and accuracy values for model training.

    Returns
    -------
    None.

    """
    
    # Model Loss Curve
    plt.plot(hist.history['loss'], label='train_loss')
    plt.plot(hist.history['val_loss'], label='val_loss')
    plt.ylabel('Cost')
    plt.xlabel('Epoch #')
    plt.title("Model Loss Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(artifacts_path + '/model_loss_curve.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Model Accuracy Curve
    plt.plot(hist.history['categorical_accuracy'], label='train_accuracy')
    plt.plot(hist.history['val_categorical_accuracy'], label='val_accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch #')
    plt.title("Model Accuracy Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(artifacts_path + '/model_accuracy_curve.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


# Data Preprocessing
data_dict, class_weight = preprocess_data(dataset_path, 'TRAIN', 
                                          npz_fl, embed_models_fl)

Xtrain = data_dict['Xtrain']
Ytrain = data_dict['Ytrain']
Ytrain_oh = data_dict['Ytrain_oh']
Xtest = data_dict['Xtest']
Ytest = data_dict['Ytest']

print("[{}]  --------- Training Dataset ---------".format(datetime.now()))
print("[{}]      Xtrain shape: {}".format(datetime.now(), Xtrain.shape))
print("[{}]      Ytrain shape: {}".format(datetime.now(), Ytrain.shape))
print("[{}]      Ytrain_oh shape: {}".format(datetime.now(), Ytrain_oh.shape))

print("[{}]  --------- Test Dataset ---------".format(datetime.now()))
print("[{}]      Xtest shape: {}".format(datetime.now(), Xtest.shape))
print("[{}]      Ytest shape: {}".format(datetime.now(), Ytest.shape))

# Create and display the model
model = prog_lang_detect_model(Xtrain.shape[1], Ytrain_oh.shape[1])
print("[{}]  Model Summary...\n".format(datetime.now()))
print(model.summary())
plot_model(model, to_file=artifacts_path + '/prog_lang_detect_model.png', 
           show_shapes=True, show_layer_names=True)

# Compile model to configure the learning process
model.compile(loss='categorical_crossentropy', 
              metrics=['categorical_accuracy'],
              optimizer=Lookahead(AdamW(lr=1e-2, 
                                        weight_decay=1e-5, 
                                        clipvalue=700), 
                                  sync_period=10))

# Early stopping policy
early = EarlyStopping(monitor="val_loss", mode="min", 
                      restore_best_weights=True, 
                      patience=10, verbose=1)

# Reduce LR on plateau policy
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, 
                              min_lr=1e-5, patience=7, 
                              verbose=1, mode='min')

# Fit the model
print("[{}]  Starting model training...".format(datetime.now()))
history = model.fit(x=Xtrain, y=Ytrain_oh, class_weight=class_weight,
                    batch_size=mini_batch_size, epochs=max_iterations, 
                    verbose=1, callbacks=[reduce_lr, early], workers=5,
                    validation_split=0.1)
print("[{}]  Model training completed.".format(datetime.now()))

model.save(model_save_path + '/Prog_Lang_Detect_Model.h5', overwrite=True)
print("[{}]  Model saved at path: {}.".format(datetime.now(), model_save_path))

# Get validation metrics
y_pred = model.predict(Xtest)
y_pred = np.array([np.argmax(y_pred, axis=1)]).T
f1 = f1_score(Ytest, y_pred, average='weighted')
acc = accuracy_score(Ytest, y_pred)
print("[{}]  Metrics for test dataset...\n    F1-Score: {} \n    Accuracy: {}".format(datetime.now(), f1, acc))
print("\n[{}]  Classification Report...\n\n{}\n\n".format(datetime.now(), classification_report(Ytest, y_pred)))

# Plot confusion matrix and learning curves
num_classes = len(np.unique(Ytest))
cnf_matrix = confusion_matrix(Ytest, y_pred, labels=list(range(num_classes)))
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, classes=list(range(num_classes)))
learning_curves(history)
