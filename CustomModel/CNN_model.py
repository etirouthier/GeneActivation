#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:20:53 2019

@author: routhier
"""

from keras.models import Sequential
from keras.layers import Dropout,Flatten, BatchNormalization
from keras.layers import Dense, Conv2D, MaxPooling2D

def cnn_model() :
    """
        Create a convolutional model with 3 convolutional layers before a final 
        dense a layer with one node used to make the final prediction.
        
        ..notes: the precision of the prediction does not depend strongly with the architecture.
    """
    window = 2001
    num_classes = 1
    
    fashion_model = Sequential()
    
    fashion_model.add(Conv2D(64, kernel_size=(12,4),activation='relu',input_shape=(window,4,1),padding='valid'))
    fashion_model.add(MaxPooling2D((2,1),padding='same'))
    fashion_model.add(BatchNormalization())
    fashion_model.add(Dropout(0.2))

    fashion_model.add(Conv2D(16, kernel_size=(8,1),activation='relu',padding='same'))
    fashion_model.add(MaxPooling2D((2,1),padding='same'))
    fashion_model.add(BatchNormalization())
    fashion_model.add(Dropout(0.2))
    
    fashion_model.add(Conv2D(8, kernel_size=(4,1),activation='relu',padding='same'))
    fashion_model.add(MaxPooling2D((2,1),padding='same'))
    fashion_model.add(BatchNormalization())
    fashion_model.add(Dropout(0.2))
    
    fashion_model.add(Flatten()) 
    
    fashion_model.add(Dense(64, activation = 'relu'))

    fashion_model.add(Dense(num_classes, activation='relu'))

    return fashion_model 

