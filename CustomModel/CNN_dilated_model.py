#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:50:33 2019

@author: routhier
"""

from keras.models import Model
from keras.layers import Dropout, LeakyReLU, Concatenate
from keras.layers import Conv2D, Reshape, Input, MaxPooling2D
import keras.backend as K

def cnn_dilated_model(window) :
    """
        Create a convolutional model with 2 convolutional layers with maxpooling before applying 
        a dilated convolutional layer to the model. The result is given by a layer with a convolution
        with size (1x1).
        
        ..notes: the precision of the prediction does not depend strongly on the architecture.
    """        
    inputs = Input(shape=(window, 1, 4))
    
    max_pool_layer = MaxPooling2D((4, 1),padding='same')
    leaky_relu_layer = LeakyReLU(alpha=0.1)
    dropout_layer = Dropout(0.2)
    conv_layer_1 = Conv2D(64, kernel_size=(22, 1), activation='relu',padding='same')
    conv_layer_2 = Conv2D(40, kernel_size=(6, 1),activation='relu',padding='same')
    conv_layer_3 = Conv2D(1, kernel_size=(1, 1),activation='relu',padding='same')
    
    image = conv_layer_1(inputs)
    image = leaky_relu_layer(image)
    image = max_pool_layer(image)
    image = dropout_layer(image)
    
    image = conv_layer_2(image)
    image = leaky_relu_layer(image)
    image =  MaxPooling2D((2, 1),padding='same')(image)
    new_image = dropout_layer(image)

    #dilated_conv_layer = Conv2D(40, kernel_size=(3,1),activation='relu',padding='same', dilation_rate = (1,1))
    #new_image = dilated_conv_layer(image)
    #new_image = leaky_relu_layer(new_image)
    #new_image = dropout_layer(new_image)
    
    for i in range(1,5):
        dilated_conv_layer = Conv2D(16, kernel_size=(3,1),activation='relu',padding='same', dilation_rate = (2**i,1))
        image_1 = dilated_conv_layer(new_image)
        image_1 = leaky_relu_layer(image_1)
        image_1 = dropout_layer(image_1)
        new_image = Concatenate(axis=-1)([new_image, image_1])
        
    output = conv_layer_3(new_image)
    output = Reshape((K.int_shape(output)[1], 1))(output)
   
    return Model(inputs, output), K.int_shape(output)[1]
