#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:25:16 2019

@author: routhier
"""

from keras.models import Sequential
from keras.layers import Dropout,Flatten, TimeDistributed, GlobalAveragePooling1D
from keras.layers import Dense, Conv2D, MaxPooling2D, LSTM, Activation

def cnn_lstm_model() :
    """
        Create a convolutional model with 2 convolutional layers, 2 time distributed dense layer, an 
        LSTM layer and finally a global average pooling layer.
        
        ..notes: the precision of the prediction does not depend strongly with the architecture.
    """
    window = 2001
    num_classes = 1
    
    model=Sequential()
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same'), input_shape=(window,10,4,1)))
    model.add(TimeDistributed(Activation('relu')))
   
    model.add(TimeDistributed(Conv2D(32, (3, 3))))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
   
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(32)))
    model.add(TimeDistributed(Dense(35, name="first_dense" )))
    
    model.add(LSTM(20, return_sequences=True, name="lstm_layer"));#%%
    model.add(TimeDistributed(Dense(num_classes, activation='sigmoid'), name="time_distr_dense_one"))
    model.add(GlobalAveragePooling1D(name="global_avg"))

    return model 

