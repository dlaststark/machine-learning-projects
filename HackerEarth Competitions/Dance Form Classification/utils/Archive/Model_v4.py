# -*- coding: utf-8 -*-
"""
Created on Fri May 29 04:11:31 2020

This script defines the machine learning model for image classification.

@author: Tapas Das
"""


from tensorflow.keras.layers import Input, Dropout, BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, Add, Flatten
from tensorflow.keras.layers import Activation, MaxPooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2


def identity_block(x, f, filters, stage, block):
    """
    Identity block implementation for ResNet-50 model.

    Parameters
    ----------
    x : NumPy matrix
        Feature matrix for training data.
    f : Integer
        Kernel dimensions.
    filters : List
        List of filter dimensions.
    stage : Integer
        Stage# of ResNet model.
    block : String
        Block identifier of ResNet model.

    Returns
    -------
    x : NumPy matrix
        Feature matrix after identity transformation.

    """
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    drop_name_base = 'drop' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    x_shortcut = x
    
    x = Conv2D(filters=F1, kernel_size=(1, 1), padding='valid', 
               name=conv_name_base+'2a', kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
    x = BatchNormalization(axis=-1, name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.2, name=drop_name_base+'2a')(x)
    
    x = Conv2D(filters=F2, kernel_size=(f, f), padding='same', 
               name=conv_name_base+'2b', kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
    x = BatchNormalization(axis=-1, name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.2, name=drop_name_base+'2b')(x)

    x = Conv2D(filters=F3, kernel_size=(1, 1), padding='valid', 
               name=conv_name_base+'2c', kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
    x = BatchNormalization(axis=-1, name=bn_name_base+'2c')(x)
    x = Dropout(rate=0.2, name=drop_name_base+'2c')(x)

    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    
    return x


def convolutional_block(x, f, filters, stage, block, s=2):
    """
    Convolution block implementation of ResNet-50 model.

    Parameters
    ----------
    x : NumPy matrix
        Feature matrix for training data.
    f : Integer
        Kernel dimensions.
    filters : List
        List of filter dimensions.
    stage : Integer
        Stage# of ResNet model.
    block : String
        Block identifier of ResNet model.
    s : Integer, optional
        Stride dimensions. The default is 2.

    Returns
    -------
    x : NumPy matrix
        Feature matrix after convolution block transformation.

    """
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    drop_name_base = 'drop' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    x_shortcut = x

    # Main Path
    x = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s,s), 
               name=conv_name_base+'2a', kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
    x = BatchNormalization(axis=-1, name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.2, name=drop_name_base+'2a')(x)

    x = Conv2D(filters=F2, kernel_size=(f, f), padding='same', 
               name=conv_name_base+'2b', kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
    x = BatchNormalization(axis=-1, name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.2, name=drop_name_base+'2b')(x)

    x = Conv2D(filters=F3, kernel_size=(1, 1), padding='valid', 
               name=conv_name_base+'2c', kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
    x = BatchNormalization(axis=-1, name=bn_name_base+'2c')(x)
    x = Dropout(rate=0.2, name=drop_name_base+'2c')(x)

    # Shortcut Path
    x_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s,s), 
                        padding='valid', name=conv_name_base+'1',
                        kernel_initializer='he_normal', 
                        kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x_shortcut)
    x_shortcut = BatchNormalization(axis=-1, name=bn_name_base+'1')(x_shortcut)
    x_shortcut = Dropout(rate=0.2, name=drop_name_base+'1')(x_shortcut)

    # Add shortcut value to main path
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    
    return x


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

	# Stage 1
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), 
               name='conv1', kernel_initializer='he_normal', 
               kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
    x = BatchNormalization(axis=-1, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Stage 2
    x = convolutional_block(x, f=3, filters=[64, 64, 256], 
                            stage=2, block='a', s=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    x = convolutional_block(x, f = 3, filters = [128, 128, 512], 
                            stage = 3, block='a', s = 2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    x = convolutional_block(x, f = 3, filters = [256, 256, 1024], 
                            stage = 4, block='a', s = 2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    x = convolutional_block(x, f = 3, filters = [512, 512, 2048], 
                            stage = 5, block='a', s = 2)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # Maxpool
    x = MaxPooling2D(pool_size=(2,2), name="max_pool")(x)

    # Output Layer
    x = Flatten()(x)
    x = Dense(8, activation='softmax', name='OUTPUT', 
              kernel_initializer='he_normal')(x)

    # Create Keras Model instance
    model = Model(inputs=x_input, outputs=x, name='Dance_Form_Classifier')
    return model
