#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to define the machine learning model for Programming Language Detection

Created on Mon Nov  2 17:34:04 2020

@author: tapasdas
"""


from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, BatchNormalization
from tensorflow.keras.layers import SpatialDropout1D, AveragePooling1D
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.models import Model


def prog_lang_detect_model(input_shape, output_shape):
    
    # Input Layer
    x_input = Input(shape=(input_shape, 1))
    x = BatchNormalization()(x_input)

    # Convolutional Layers
    x = WeightNormalization(
          Conv1D(filters=16, kernel_size=5, strides=2, padding='same', 
                 activation='swish', kernel_regularizer=l2(0.0005),
                 kernel_initializer=he_uniform(seed=1)))(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = SpatialDropout1D(rate=0.4)(x)
    
    x = WeightNormalization(
          Conv1D(filters=32, kernel_size=3, strides=1, padding='same', 
                 activation='swish', kernel_regularizer=l2(0.0005),
                 kernel_initializer=he_uniform(seed=1)))(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = SpatialDropout1D(rate=0.4)(x)
    
    x = WeightNormalization(
          Conv1D(filters=32, kernel_size=3, strides=1, padding='same', 
                 activation='swish', kernel_regularizer=l2(0.0005),
                 kernel_initializer=he_uniform(seed=1)))(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = SpatialDropout1D(rate=0.4)(x)

    x = WeightNormalization(
          Conv1D(filters=64, kernel_size=3, strides=1, padding='same', 
                 activation='swish', kernel_regularizer=l2(0.0005),
                 kernel_initializer=he_uniform(seed=1)))(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = SpatialDropout1D(rate=0.4)(x)
    
    # Fully-connected Layers
    x = Flatten()(x)
    x = WeightNormalization(
          Dense(units=1024, activation='swish', kernel_regularizer=l2(0.001), 
                kernel_initializer=he_uniform(seed=1)))(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    
    x = WeightNormalization(
          Dense(units=1024, activation='swish', kernel_regularizer=l2(0.001), 
                kernel_initializer=he_uniform(seed=1)))(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    
    # Output Layer
    x = WeightNormalization(
          Dense(units=output_shape, activation='softmax', 
                kernel_initializer=he_uniform(seed=1)))(x)

    # Create Keras Model instance
    model = Model(inputs=x_input, outputs=x, name='Prog_Lang_Detect_Model')

    return model
