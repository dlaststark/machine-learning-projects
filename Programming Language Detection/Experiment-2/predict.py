#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:44:19 2020

@author: tapasdas
"""


import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import WeightNormalization
from utils.preprocess_data import preprocess_data
from utils.params import predictions_path, rev_lang_map, embed_models_fl


# Create placeholders for user input
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset_path", required=True,
	help="Full path for source dataset")
ap.add_argument("-m", "--model_save_path", required=True,
	help="Full path where the trained model is saved")
ap.add_argument("-p", "--pred_file_name", required=True,
	help="File to store the predictions made")
args = vars(ap.parse_args())

dataset_path = args["dataset_path"]
model_save_path = args["model_save_path"]
pred_file_name = args["pred_file_name"]

# Data Preprocessing
data_dict, class_weight, fileset = preprocess_data(dataset_path, 'PREDICT', 
                                                   None, embed_models_fl)
Xpredict = data_dict['Xpredict']
fileset_df = pd.DataFrame(list(fileset.items()), columns=['ID', 'File_Name'])

# Load the trained model
model = load_model(model_save_path + '/Prog_Lang_Detect_Model.h5')
print("[{}]  Trained model loaded for making predictions.".format(datetime.now()))

# Make predictions
y_pred = model.predict(Xpredict)
y_pred = np.array([np.argmax(y_pred, axis=1)]).T
fileset_df['Prog_Lang'] = y_pred
fileset_df['Prog_Lang'] = fileset_df['Prog_Lang'].map(rev_lang_map)
fileset_df.to_csv(predictions_path + '/' + pred_file_name, index=False)
print("[{}]  Predictions saved at: {}.".format(datetime.now(), predictions_path + '/' + pred_file_name))
