# -*- coding: utf-8 -*-
"""
Created on Fri May 29 04:11:31 2020

This script defines the machine learning model for image classification.

@author: Tapas Das
"""


from tensorflow.keras.layers import Input, Dropout, BatchNormalization
from tensorflow.keras.layers import GlobalMaxPooling2D, ZeroPadding2D, Conv2D
from tensorflow.keras.layers import Activation, MaxPooling2D
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
    x = Conv2D(filters=16, kernel_size=(5, 5), 
               padding='same', activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-1A')(x)
    x = Conv2D(filters=16, kernel_size=(5, 5), 
               padding='same', activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-1B')(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MAXPOOL-1')(x)
    x = Dropout(rate=0.25, name='DROPOUT_CONV-1')(x)
    
    x = Conv2D(filters=32, kernel_size=(3, 3), 
               padding='same', activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-2A')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), 
               padding='same', activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-2B')(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MAXPOOL-2')(x)
    x = Dropout(rate=0.25, name='DROPOUT_CONV-2')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), 
               padding='same', activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-3A')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), 
               padding='same', activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-3B')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), 
               padding='same', activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-3C')(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-3')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MAXPOOL-3')(x)
    x = Dropout(rate=0.25, name='DROPOUT_CONV-3')(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), 
               padding='same', activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-4A')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), 
               padding='same', activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-4B')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), 
               padding='same', activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-4C')(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-4')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MAXPOOL-4')(x)
    x = Dropout(rate=0.25, name='DROPOUT_CONV-4')(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), 
               padding='same', activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-5A')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), 
               padding='same', activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-5B')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), 
               padding='same', activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-5C')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), 
               padding='same', activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='CONV-5D')(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-5')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MAXPOOL-5')(x)
    x = Dropout(rate=0.25, name='DROPOUT_CONV-5')(x)

    # Fully Connected Layers
    x = Conv2D(filters=512, kernel_size=(1, 1), 
               padding='same', activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='FC-1')(x)
    x = BatchNormalization(axis=-1, name='BN_FC-1')(x)
    x = Dropout(rate=0.25, name='DROPOUT_FC-1')(x)
    
    x = Conv2D(filters=512, kernel_size=(1, 1), 
               padding='same', activation='relu', 
               kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.05, l2=0.05), 
               name='FC-2')(x)
    x = BatchNormalization(axis=-1, name='BN_FC-2')(x)
    x = Dropout(rate=0.25, name='DROPOUT_FC-2')(x)

    # Output Layer
    x = Conv2D(filters=8, kernel_size=(1, 1), name='OUTPUT')(x)
    x = BatchNormalization(axis=-1, name='BN_OUTPUT')(x)
    x = GlobalMaxPooling2D()(x)
    x = Activation('softmax')(x)

    # Create Keras Model instance
    model = Model(inputs=x_input, outputs=x, name='Dance_Form_Classifier')
    return model
