#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:37:39 2019

@author: routhier
"""

from CustomModel.CNN_model import cnn_model as cnn
from CustomModel.CNN_deep_model import cnn_deep_model as cnn_deep
from CustomModel.CNN_LSTM_model import cnn_lstm_model as cnn_lstm
from CustomModel.CNN_dilated_model import cnn_dilated_model as cnn_dilated
from CustomModel.CNN_reversed import cnn_reversed

def model_dictionary(window):
    """
        Dictionary used to import the several model with parse arguments.
    """
    return {'cnn': cnn(window), 'cnn_lstm': cnn_lstm(window), 
            'cnn_dilated': cnn_dilated(window), 'cnn_deep': cnn_deep(window),
            'cnn_reversed' : cnn_reversed(window)}
