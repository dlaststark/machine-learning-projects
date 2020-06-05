# -*- coding: utf-8 -*-
"""
Created on Fri May 29 04:11:31 2020

This script defines the machine learning model for image classification.

@author: Tapas Das
"""


from tensorflow.keras.layers import Input, Dropout, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense, ZeroPadding2D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2


def cnn_model(input_shape):
    """
    Build the Keras CNN model for classifying the dance forms

    Parameters
    ----------
    input_shape : Tuple
        Dimensions of input image.

    Returns
    -------
    model : Keras Model
        Keras CNN Model object.

    """
    
    # Input Layer
    x_input = Input(shape=input_shape, name='INPUT')
    x = ZeroPadding2D((2, 2))(x_input)

    # Convolution Layers
    x = Conv2D(filters=16, kernel_size=(7, 7), strides=(2, 2), 
               padding='valid', 
               activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-1')(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-1')(x)
    x = Dropout(rate=0.25, name='DROPOUT_CONV-1')(x)
    
    x = Conv2D(filters=32, kernel_size=(3, 3), 
               padding='same', 
               activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-2A')(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-2A')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), 
               padding='same', 
               activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-2B')(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-2B')(x)
    x = Dropout(rate=0.25, name='DROPOUT_CONV-2')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), 
               padding='same', 
               activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-3A')(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-3A')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), 
               padding='same', 
               activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-3B')(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-3B')(x)
    x = Dropout(rate=0.25, name='DROPOUT_CONV-3')(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), 
               padding='same', 
               activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-4A')(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-4A')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), 
               padding='same', 
               activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-4B')(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-4B')(x)
    x = Dropout(rate=0.25, name='DROPOUT_CONV-4')(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), 
               padding='same', 
               activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-5A')(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-5A')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), 
               padding='same', 
               activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-5B')(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-5B')(x)
    x = Dropout(rate=0.25, name='DROPOUT_CONV-5')(x)

    # Fully Connected Layers
    x = Flatten(name='FLATTEN')(x)
    x = Dense(units=512, activation='relu', 
              kernel_initializer='he_normal',
              kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
              name='FC-1')(x)
    x = BatchNormalization(axis=-1, name='BN_FC-1')(x)
    x = Dropout(rate=0.25, name='DROPOUT_FC-1')(x)

    # Output Layer
    x = Dense(units=8, activation='softmax', name='OUTPUT')(x)

    # Create Keras Model instance
    model = Model(inputs=x_input, outputs=x, name='Dance_Form_Classifier')
    return model
