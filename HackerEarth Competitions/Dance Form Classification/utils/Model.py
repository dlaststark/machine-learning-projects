# -*- coding: utf-8 -*-
"""
Created on Fri May 29 04:11:31 2020

This script defines the machine learning model for image classification.

@author: Tapas Das
"""


from tensorflow.keras.layers import Input, Dropout, BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D, SeparableConv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dense, Flatten
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
    x = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), 
               padding='valid', kernel_initializer='he_normal', name='CONV-1A',
               kernel_regularizer=l1_l2(l1=0.1, l2=0.1))(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-1A')(x)
    x = Activation('selu')(x)
    x = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), 
               padding='same', kernel_initializer='he_normal', name='CONV-1B',
               kernel_regularizer=l1_l2(l1=0.1, l2=0.1))(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-1B')(x)
    x = Activation('selu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MAXPOOL-1')(x)
    x = Dropout(rate=0.1, name='DROPOUT_CONV-1')(x)
    
    x = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), 
               padding='same', kernel_initializer='he_normal', name='CONV-2A',
               kernel_regularizer=l1_l2(l1=0.1, l2=0.1))(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-2A')(x)
    x = Activation('selu')(x)
    x = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), 
               padding='same', kernel_initializer='he_normal', name='CONV-2B',
               kernel_regularizer=l1_l2(l1=0.1, l2=0.1))(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-2B')(x)
    x = Activation('selu')(x)
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), 
               padding='same', kernel_initializer='he_normal', name='CONV-2C',
               kernel_regularizer=l1_l2(l1=0.1, l2=0.1))(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-2C')(x)
    x = Activation('selu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MAXPOOL-2')(x)
    x = Dropout(rate=0.1, name='DROPOUT_CONV-2')(x)

    x = SeparableConv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), 
               padding='same', kernel_initializer='he_normal', name='CONV-3A',
               kernel_regularizer=l1_l2(l1=0.1, l2=0.1))(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-3A')(x)
    x = Activation('selu')(x)
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), 
               padding='same', kernel_initializer='he_normal', name='CONV-3B',
               kernel_regularizer=l1_l2(l1=0.1, l2=0.1))(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-3B')(x)
    x = Activation('selu')(x)
    x = SeparableConv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), 
               padding='same', kernel_initializer='he_normal', name='CONV-3C',
               kernel_regularizer=l1_l2(l1=0.1, l2=0.1))(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-3C')(x)
    x = Activation('selu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MAXPOOL-3')(x)
    x = Dropout(rate=0.1, name='DROPOUT_CONV-3')(x)

    x = SeparableConv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), 
               padding='same', kernel_initializer='he_normal', name='CONV-4A',
               kernel_regularizer=l1_l2(l1=0.1, l2=0.1))(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-4A')(x)
    x = Activation('selu')(x)
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), 
               padding='same', kernel_initializer='he_normal', name='CONV-4B',
               kernel_regularizer=l1_l2(l1=0.1, l2=0.1))(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-4B')(x)
    x = Activation('selu')(x)
    x = SeparableConv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), 
               padding='same', kernel_initializer='he_normal', name='CONV-4C',
               kernel_regularizer=l1_l2(l1=0.1, l2=0.1))(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-4C')(x)
    x = Activation('selu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MAXPOOL-4')(x)
    x = Dropout(rate=0.1, name='DROPOUT_CONV-4')(x)

    # Fully Connected Layers
    x = Flatten(name='FLATTEN')(x)
    x = Dense(units=128, name='FC-1', kernel_initializer='he_normal',
              kernel_regularizer=l1_l2(l1=0.1, l2=0.1))(x)
    x = BatchNormalization(axis=-1, name='BN_FC-1')(x)
    x = Activation('selu')(x)
    x = Dropout(rate=0.2, name='DROPOUT_FC-1')(x)
    
    x = Dense(units=128, name='FC-2', kernel_initializer='he_normal',
              kernel_regularizer=l1_l2(l1=0.1, l2=0.1))(x)
    x = BatchNormalization(axis=-1, name='BN_FC-2')(x)
    x = Activation('selu')(x)
    x = Dropout(rate=0.2, name='DROPOUT_FC-2')(x)

    # Output Layer
    x = Dense(units=8, name='OUTPUT', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=-1, name='BN_FC-OUTPUT')(x)
    x = Activation('softmax')(x)

    # Create Keras Model instance
    model = Model(inputs=x_input, outputs=x, name='Dance_Form_Classifier')
    return model
