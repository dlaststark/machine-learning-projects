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
from tensorflow.keras.layers import Activation, Add
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.models import Model


def identity_block(x, f, filters):

    F1, F2, F3 = filters

    x_shortcut = x

    # Main Path
    x = Conv1D(filters=F1, kernel_size=1, padding='same',
               activation='swish', kernel_regularizer=l2(0.0005), 
               kernel_initializer=he_uniform(seed=1))(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(rate=0.4)(x)

    x = Conv1D(filters=F2, kernel_size=f, padding='same',
               activation='swish', kernel_regularizer=l2(0.0005), 
               kernel_initializer=he_uniform(seed=1))(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(rate=0.4)(x)

    x = Conv1D(filters=F3, kernel_size=1, padding='same',
               activation='linear', kernel_regularizer=l2(0.0005), 
               kernel_initializer=he_uniform(seed=1))(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(rate=0.4)(x)

    # Add shortcut value to main path
    x = Add()([x, x_shortcut])
    x = Activation('swish')(x)
    x = BatchNormalization()(x)

    return x


def convolution_block(x, f, filters):
    
    F1, F2, F3 = filters

    x_shortcut = x

    # Main Path
    x = Conv1D(filters=F1, kernel_size=1, padding='same', 
               activation='swish', kernel_regularizer=l2(0.0005), 
               kernel_initializer=he_uniform(seed=1))(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(rate=0.4)(x)

    x = Conv1D(filters=F2, kernel_size=f, strides=2, padding='same', 
               activation='swish', kernel_regularizer=l2(0.0005), 
               kernel_initializer=he_uniform(seed=1))(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(rate=0.4)(x)

    x = Conv1D(filters=F3, kernel_size=1, padding='same', 
               activation='linear', kernel_regularizer=l2(0.0005), 
               kernel_initializer=he_uniform(seed=1))(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(rate=0.4)(x)

    # Shortcut Path
    x_shortcut = Conv1D(filters=F3, kernel_size=1, strides=2, padding='same',
                        activation='linear', kernel_regularizer=l2(0.0005), 
                        kernel_initializer=he_uniform(seed=1))(x_shortcut)
    x_shortcut = BatchNormalization()(x_shortcut)
    x_shortcut = SpatialDropout1D(rate=0.4)(x_shortcut)

    # Add shortcut value to main path
    x = Add()([x, x_shortcut])
    x = Activation('swish')(x)
    x = BatchNormalization()(x)

    return x
    

def prog_lang_detect_model(input_shape, output_shape):
    
    # Input Layer
    x_input = Input(shape=(input_shape, 1))
    x = BatchNormalization()(x_input)

    # Convolutional Layers
    x = Conv1D(filters=8, kernel_size=5, strides=2, padding='same', 
               activation='swish', kernel_regularizer=l2(0.0005),
               kernel_initializer=he_uniform(seed=1))(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = SpatialDropout1D(rate=0.4)(x)
    
    x = convolution_block(x, f=3, filters=[8, 8, 32])
    x = identity_block(x, f=3, filters=[8, 8, 32])
    
    x = convolution_block(x, f=3, filters=[8, 8, 32])
    x = identity_block(x, f=3, filters=[8, 8, 32])
    
    x = convolution_block(x, f=3, filters=[16, 16, 64])
    x = identity_block(x, f=3, filters=[16, 16, 64])
    
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
