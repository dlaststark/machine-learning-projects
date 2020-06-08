# -*- coding: utf-8 -*-
"""
Created on Fri May 29 04:11:31 2020

This script defines the machine learning model for image classification.

@author: Tapas Das
"""


from tensorflow.keras.layers import Input, BatchNormalization, Dropout, Add
from tensorflow.keras.layers import ZeroPadding2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Activation, SeparableConv2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2


def identity_block(x, f, filters, stage, block, dr=0.1, lr=0.005, dm=1):
    """
    Identity block implementation for ResNet model.

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
    dr : Float, optional
        Dropout rate.
    lr : Float, optional
        L1/L2 regularization value.
    dm : Integer, optional
        Depth multiplier for Spatial Convolutions.

    Returns
    -------
    x : NumPy matrix
        Feature matrix after identity transformation.

    """
    
    conv_name_base = 'CONV-' + str(stage) + block + '-BRANCH-'
    bn_name_base = 'BN_CONV-' + str(stage) + block + '-BRANCH-'
    drop_name_base = 'DROPOUT-' + str(stage) + block + '-BRANCH-'
    
    F1, F2, F3 = filters
    
    x_shortcut = x
    
    # Main Path
    x = SeparableConv2D(filters=F1, kernel_size=(1, 1), padding='same', 
                        name=conv_name_base+'A', kernel_initializer='he_normal', 
                        kernel_regularizer=l1_l2(l1=lr, l2=lr))(x)
    x = BatchNormalization(axis=-1, name=bn_name_base+'A')(x)
    x = Activation('selu')(x)
    x = Dropout(rate=dr, name=drop_name_base+'A')(x)
    
    x = SeparableConv2D(filters=F2, kernel_size=(f, f), padding='same', 
                        name=conv_name_base+'B', depth_multiplier=dm, 
                        kernel_initializer='he_normal', 
                        kernel_regularizer=l1_l2(l1=lr, l2=lr))(x)
    x = BatchNormalization(axis=-1, name=bn_name_base+'B')(x)
    x = Activation('selu')(x)
    x = Dropout(rate=dr, name=drop_name_base+'B')(x)
    
    x = SeparableConv2D(filters=F3, kernel_size=(1, 1), padding='same', 
                        name=conv_name_base+'C', kernel_initializer='he_normal', 
                        kernel_regularizer=l1_l2(l1=lr, l2=lr))(x)
    x = BatchNormalization(axis=-1, name=bn_name_base+'C')(x)
    x = Dropout(rate=dr, name=drop_name_base+'C')(x)
    
    # Add shortcut value to main path
    x = Add()([x, x_shortcut])
    x = Activation('selu')(x)
    
    return x


def convolutional_block(x, f, filters, stage, block, s=2, dr=0.1, lr=0.005, dm=1):
    """
    Convolution block implementation of ResNet model.

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
    dr : Float, optional
        Dropout rate.
    lr : Float, optional
        L1/L2 regularization value.
    dm : Integer, optional
        Depth multiplier for Spatial Convolutions.

    Returns
    -------
    x : NumPy matrix
        Feature matrix after convolution block transformation.

    """
    
    conv_name_base = 'CONV-' + str(stage) + block + '-BRANCH-'
    bn_name_base = 'BN_CONV-' + str(stage) + block + '-BRANCH-'
    drop_name_base = 'DROPOUT-' + str(stage) + block + '-BRANCH-'
    
    F1, F2, F3 = filters
    
    x_shortcut = x

    # Main Path
    x = SeparableConv2D(filters=F1, kernel_size=(1, 1), padding='same', 
                        name=conv_name_base+'A', kernel_initializer='he_normal', 
                        kernel_regularizer=l1_l2(l1=lr, l2=lr))(x)
    x = BatchNormalization(axis=-1, name=bn_name_base+'A')(x)
    x = Activation('selu')(x)
    x = Dropout(rate=dr, name=drop_name_base+'A')(x)

    x = SeparableConv2D(filters=F2, kernel_size=(f, f), padding='same', 
                        name=conv_name_base+'B', strides=(s, s),
                        kernel_initializer='he_normal', depth_multiplier=dm, 
                        kernel_regularizer=l1_l2(l1=lr, l2=lr))(x)
    x = BatchNormalization(axis=-1, name=bn_name_base+'B')(x)
    x = Activation('selu')(x)
    x = Dropout(rate=dr, name=drop_name_base+'B')(x)
    
    x = SeparableConv2D(filters=F3, kernel_size=(1, 1), padding='same', 
                        name=conv_name_base+'C', kernel_initializer='he_normal', 
                        kernel_regularizer=l1_l2(l1=lr, l2=lr))(x)
    x = BatchNormalization(axis=-1, name=bn_name_base+'C')(x)
    x = Dropout(rate=dr, name=drop_name_base+'C')(x)

    # Shortcut Path
    x_shortcut = SeparableConv2D(filters=F3, kernel_size=(1, 1), padding='same', 
                        name=conv_name_base+'S', strides=(s, s),
                        kernel_initializer='he_normal',
                        kernel_regularizer=l1_l2(l1=lr, l2=lr))(x_shortcut)
    x_shortcut = BatchNormalization(axis=-1, name=bn_name_base+'S')(x_shortcut)
    x_shortcut = Dropout(rate=dr, name=drop_name_base+'S')(x_shortcut)

    # Add shortcut value to main path
    x = Add()([x, x_shortcut])
    x = Activation('selu')(x)
    
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
    x = SeparableConv2D(filters=64, kernel_size=(5, 5), padding='valid', 
                        name='CONV-1A', kernel_initializer='he_normal', 
                        kernel_regularizer=l1_l2(l1=0.005, l2=0.005))(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-1A')(x)
    x = Activation('selu')(x)
    x = SeparableConv2D(filters=64, kernel_size=(5, 5), padding='same', 
                        name='CONV-1B', kernel_initializer='he_normal', 
                        kernel_regularizer=l1_l2(l1=0.005, l2=0.005))(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-1B')(x)
    x = Activation('selu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MAXPOOL-1')(x)
    x = Dropout(rate=0.1, name='DROPOUT_CONV-1')(x)
    
    # Stage 2
    x = convolutional_block(x, f=3, filters=[16, 16, 64], stage=2, block='A', s=1, dm=1)
    x = convolutional_block(x, f=3, filters=[16, 16, 64], stage=2, block='B', s=2, dm=1)
    x = identity_block(x, 3, [16, 16, 64], stage=2, block='C')
    x = identity_block(x, 3, [16, 16, 64], stage=2, block='D')
    
    # Stage 3
    x = convolutional_block(x, f=3, filters=[32, 32, 128], stage=3, block='A', s=1, dm=2)
    x = convolutional_block(x, f=3, filters=[32, 32, 128], stage=3, block='B', s=2, dm=2)
    x = identity_block(x, 3, [32, 32, 128], stage=3, block='C')
    x = identity_block(x, 3, [32, 32, 128], stage=3, block='D')
    
    # Stage 4
    x = convolutional_block(x, f=3, filters=[64, 64, 256], stage=4, block='A', s=1, dm=4)
    x = convolutional_block(x, f=3, filters=[64, 64, 256], stage=4, block='B', s=2, dm=4)
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='C')
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='D')
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='E')
    
    # Stage 5
    x = convolutional_block(x, f=3, filters=[128, 128, 512], stage=5, block='A', s=1, dm=6)
    x = convolutional_block(x, f=3, filters=[128, 128, 512], stage=5, block='B', s=2, dm=6)
    x = identity_block(x, 3, [128, 128, 512], stage=5, block='C')
    x = identity_block(x, 3, [128, 128, 512], stage=5, block='D')
    x = identity_block(x, 3, [128, 128, 512], stage=5, block='E')
    
    # Stage 6
    x = convolutional_block(x, f=3, filters=[256, 256, 1024], stage=6, block='A', s=1, dm=8)
    x = convolutional_block(x, f=3, filters=[256, 256, 1024], stage=6, block='B', s=2, dm=8)
    x = identity_block(x, 3, [256, 256, 1024], stage=6, block='C')
    x = identity_block(x, 3, [256, 256, 1024], stage=6, block='D')
    x = identity_block(x, 3, [256, 256, 1024], stage=6, block='E')
    x = identity_block(x, 3, [256, 256, 1024], stage=6, block='F')
    
    # Stage 7
    x = SeparableConv2D(filters=1024, kernel_size=(3, 3), padding='same', 
                        name='CONV-7', kernel_initializer='he_normal', 
                        kernel_regularizer=l1_l2(l1=0.005, l2=0.005))(x)
    x = BatchNormalization(axis=-1, name='BN_CONV-7')(x)
    x = Activation('selu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MAXPOOL-7')(x)
    x = Dropout(rate=0.1, name='DROPOUT_CONV-7')(x)
    
    # Fully Connected Layers
    x = Flatten(name='FLATTEN')(x)
    
    x = Dense(units=1024, name='FC-1', 
              kernel_initializer='he_normal',
              kernel_regularizer=l1_l2(l1=0.005, l2=0.005))(x)
    x = BatchNormalization(axis=-1, name='BN_FC-1')(x)
    x = Activation('selu')(x)
    x = Dropout(rate=0.2, name='DROPOUT_FC-1')(x)
    
    x = Dense(units=1024, name='FC-2', 
              kernel_initializer='he_normal',
              kernel_regularizer=l1_l2(l1=0.005, l2=0.005))(x)
    x = BatchNormalization(axis=-1, name='BN_FC-2')(x)
    x = Activation('selu')(x)
    x = Dropout(rate=0.2, name='DROPOUT_FC-2')(x)
    
    x = Dense(units=8, name='OUTPUT', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=-1, name='BN_FC-OUTPUT')(x)
    x = Activation('softmax')(x)
    
    # Create Keras Model instance
    model = Model(inputs=x_input, outputs=x, name='Dance_Form_Classifier')
    return model
