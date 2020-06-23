# -*- coding: utf-8 -*-
"""
Created on Fri May 29 01:54:57 2020

This script prepares an image batch object containing data of input images.

@author: Tapas Das
"""


import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array


def create_image_batch(df, img_path, thres=224):
    '''
    Function to prepare image batch object for input images

    Parameters
    ----------
    df : DataFrame
        Input dataframe containing image file names.
    img_path : String
        File path for the input images.
    thres : Integer, optional
        Upper limit for input image dimensions. The default is 224.

    Returns
    -------
    image_batch : NumPy Matrix
        Image batch containing data of all input images.

    '''

    image_group = []

    for idx, file in enumerate(df['Image']):

        # Set file path for input image
        file_path = img_path + "/" + file

        # Read input image using OpenCV and convert to grayscale
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Scale the input image
        (h, w, _) = img.shape
        if (h > thres) or (w > thres):
            img = cv2.resize(img, (thres, thres),
                             interpolation = cv2.INTER_NEAREST)

        img = img_to_array(img)
        image_group.append(img)

    # Get the max image shape
    max_shape = tuple(max(image.shape[x] for image in image_group)
                      for x in range(3))

    # Construct an image batch object
    image_batch = np.zeros((len(image_group),) + max_shape, dtype='float32')

    # Copy all images to the upper left part of the image batch object
    for image_index, image in enumerate(image_group):
        image_batch[image_index,
                    :image.shape[0], :image.shape[1], :image.shape[2]] = image

    return image_batch
