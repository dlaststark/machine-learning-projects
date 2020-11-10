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
    """
    Function to define the identity block of Resnet model

    Parameters
    ----------
    x : Numpy array
        Feature matrix.
    f : int
        Kernel size for convolution layers.
    filters : list
        List of filter sizes for different convolution layers.

    Returns
    -------
    x : Numpy array
        Feature matrix.

    """

    F1, F2 = filters

    x_shortcut = x

    # Main Path
    x = Conv1D(filters=F1, kernel_size=1, padding='same',
               activation='swish', kernel_regularizer=l2(0.001), 
               kernel_initializer=he_uniform(seed=1))(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(rate=0.5)(x)

    x = Conv1D(filters=F2, kernel_size=f, padding='same',
               activation='linear', kernel_regularizer=l2(0.001), 
               kernel_initializer=he_uniform(seed=1))(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(rate=0.5)(x)

    # Add shortcut value to main path
    x = Add()([x, x_shortcut])
    x = Activation('swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.25)(x)

    return x


def convolution_block(x, f, filters):
    """
    Function to define the convolution block of Resnet model

    Parameters
    ----------
    x : Numpy array
        Feature Matrix.
    f : int
        Kernel size for convolution layers.
    filters : list
        List of filter sizes for different convolution layers.

    Returns
    -------
    x : Numpy array
        Feature matrix.

    """
    
    F1, F2 = filters

    x_shortcut = x

    # Main Path
    x = Conv1D(filters=F1, kernel_size=1, padding='same', 
               activation='swish', kernel_regularizer=l2(0.001), 
               kernel_initializer=he_uniform(seed=1))(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(rate=0.5)(x)

    x = Conv1D(filters=F2, kernel_size=f, strides=2, padding='same', 
               activation='linear', kernel_regularizer=l2(0.001), 
               kernel_initializer=he_uniform(seed=1))(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(rate=0.5)(x)

    # Shortcut Path
    x_shortcut = Conv1D(filters=F2, kernel_size=1, strides=2, padding='same',
                        activation='linear', kernel_regularizer=l2(0.001), 
                        kernel_initializer=he_uniform(seed=1))(x_shortcut)
    x_shortcut = BatchNormalization()(x_shortcut)
    x_shortcut = SpatialDropout1D(rate=0.5)(x_shortcut)

    # Add shortcut value to main path
    x = Add()([x, x_shortcut])
    x = Activation('swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.25)(x)

    return x
    

def prog_lang_detect_model(input_shape, output_shape):
    """
    Function to define and build the Resnet model for Programming Language Detection

    Parameters
    ----------
    input_shape : int
        Number of features in source dataset.
    output_shape : int
        Number of target labels to be predicted.

    Returns
    -------
    model : Keras Model
        Final resnet model for training and making predictions.

    """
    
    # Input Layer
    x_input = Input(shape=(input_shape, 1))
    x = BatchNormalization()(x_input)

    # Convolutional Layers
    x = Conv1D(filters=16, kernel_size=5, strides=2, padding='same', 
               activation='swish', kernel_regularizer=l2(0.001),
               kernel_initializer=he_uniform(seed=1))(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = SpatialDropout1D(rate=0.5)(x)
    
    x = convolution_block(x, f=3, filters=[16, 32])
    x = identity_block(x, f=3, filters=[16, 32])
    x = identity_block(x, f=3, filters=[16, 32])
    
    x = convolution_block(x, f=3, filters=[32, 64])
    x = identity_block(x, f=3, filters=[32, 64])
    x = identity_block(x, f=3, filters=[32, 64])
    
    x = convolution_block(x, f=3, filters=[64, 128])
    x = identity_block(x, f=3, filters=[64, 128])
    x = identity_block(x, f=3, filters=[64, 128])
    
    x = Conv1D(filters=256, kernel_size=3, strides=2, padding='same', 
               activation='swish', kernel_regularizer=l2(0.001),
               kernel_initializer=he_uniform(seed=1))(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = SpatialDropout1D(rate=0.5)(x)
    
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
