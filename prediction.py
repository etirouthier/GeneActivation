#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:02:36 2019

@author: routhier
"""

import argparse
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import load_model
from scipy.stats import pearsonr
import tensorflow as tf
import keras.backend as K

from MyModuleLibrary.array_modifier import rolling_window
from MyModuleLibrary.mykeras.losses import correlate, mse_var
from CustomModel.Models import model_dictionary

def parse_arguments(args=None):
    """
        Parse the arguments to use the module on command line.

        A serie of 6 arguments can be parsed :
            the directory that contains the DNA sequence,
            the .csv file containing the RNA-seq density of reads,
            the name of the output file were the weights of the
            trained model will be stored (in Results_nucleosome directory),
            the model we want to train,
            weither or not this model is a seq2seq model,
            if we want to make prediction on the reverse sequence.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight_file',
                        help='''File containing the trained model with
                        which the prediction will be made.''')
    parser.add_argument('-d', '--directory',
                        help='''Directory containing the DNA sequence
                        chromosome by chromosome in .hdf5 (in seq_sacCer3)''')
    parser.add_argument('-f', '--file',
                        help="""CSV file containing the nucleosome occupancy
                        on the whole genome.""")
    parser.add_argument('-s', '--seq2seq', action='store_true',
                        help='If the model is a seq2seq model')
    parser.add_argument('-m', '--model',
                        help='''Name of the model to predict
                        (only is seq2seq model)''')
    parser.add_argument('--window',
                        default=3001,
                        help='''Size of the input window''')
    parser.add_argument('-ds', '--downsampling', action='store_true',
                        help="""To downsampled the predicted sequence for a 
                        seq2seq model, the length of sampling will be calculated""")
    parser.add_argument("--test", default="4",
                        help="""chromosome on which to make prediction
                        (defaut 16 for S.cerevisiae)""")
    parser.add_argument('-r', '--reversed_seq', action='store_true',
                        help='In order to predict the backward strand')
    return parser.parse_args(args)

def load_data(seq2seq=False, reversed_seq=False, command_line_arguments=None):
    """
        The chromosome 4 and the RNA-seq density on chromosome 4 are prepared
        to be used by the model to predict the RNA-seq density.

        The DNA sequence is cut using a rolling_window with slice one to be
        used as input of the CNN model. The true value associated with every
        input is taken from the .csv file in order to evaluate the predistion.
        If the model is a seq2seq model the output is a sequence of RNA-seq
        density. We can also choose to reverse the sequence to predict the
        RNA-seq density in the negative strand.

        Args:
            seq2seq: boolean, weither the model is a seq2seq model
            reversed_seq: boolean, weither the prediction is made on negative
            strand.
    """
    args = parse_arguments(command_line_arguments)

    window = int(args.window)
    half_wx = window // 2
    # we get the path conducting to seq_chr_sacCer3
    path_to_directory = os.path.dirname(os.path.dirname(args.directory))
    path_to_file = os.path.join(path_to_directory, 'seq_sacCer3',
                                args.directory, 'chr' + args.test + '.hdf5')

    f = h5py.File(path_to_file, 'r')
    nucleotid = np.array(f['data'])
    f.close()

    if reversed_seq:
        nucleotid[nucleotid == 1] = 5
        nucleotid[nucleotid == 2] = 6
        nucleotid[nucleotid == 3] = 7
        nucleotid[nucleotid == 4] = 8
        nucleotid[nucleotid == 5] = 2
        nucleotid[nucleotid == 6] = 1
        nucleotid[nucleotid == 7] = 4
        nucleotid[nucleotid == 8] = 3

        nucleotid = nucleotid[::-1]

    x_one_hot = (np.arange(nucleotid.max()) == nucleotid[..., None]-1).astype(int)
    x_ = x_one_hot.reshape(x_one_hot.shape[0],
                           x_one_hot.shape[1] * x_one_hot.shape[2])

    proba_directory = os.path.dirname(args.file)
    proba_file = os.path.join(proba_directory, 'Start_data', args.file)

    proba = pd.read_csv(proba_file)
    y_true = proba[proba.chr == 'chr' + args.test].value.values

    if reversed_seq:
        y_true = y_true[::-1]

    if seq2seq:
        _, output_len = model_dictionary(window)[args.model]

        if args.downsampling:
            output_len_ = output_len
            output_len = window

        half_len = output_len // 2
        x_slide = rolling_window(x_, window=(window, 4),
                                 asteps=(output_len, 4))
        x_ = x_slide.reshape(x_slide.shape[0], x_slide.shape[2], 1,
                             x_slide.shape[3])
        y_true = y_true[half_wx - half_len :
                        x_slide.shape[0]*output_len + half_wx - half_len]

        if args.downsampling:
            output_len = output_len_

    else:
        x_slide = rolling_window(x_, window=(window, 4))
        x_ = x_slide.reshape(x_slide.shape[0], x_slide.shape[2], 1,
                             x_slide.shape[3])
        y_true = y_true[half_wx : -half_wx]
    
    if args.downsampling:
        return x_, y_true, output_len
    else:
        return x_, y_true

def prepare_session():
    """
        Initializing the Tensorflow session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  
    config.log_device_placement = True 
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)
    
def main(command_line_arguments=None):
    """
        Main function of the module designed to make prediction on chromosome 4
        using the parsed arguments. The prediction are compared using pearson
        correlation to the actual RNA-seq density and the results are displayed
    """
    args = parse_arguments(command_line_arguments)
    results_path = os.path.dirname(os.path.dirname(args.directory))
    path_to_weight = os.path.join(results_path, 'Results_RNA_seq',
                                  os.path.split(args.weight_file)[1])
    prepare_session()
    if args.reversed_seq:
        path_to_results = os.path.join(results_path,
                                       'Results_RNA_seq',
                                       'y_pred_bw' + args.weight_file[6:-5]\
                                       + '_applied_on_chr' + args.test + '.npy')
    else:
        path_to_results = os.path.join(results_path,
                                       'Results_RNA_seq',
                                       'y_pred_fw' + args.weight_file[6:-5]\
                                       + '_applied_on_chr' + args.test + '.npy')

    model = load_model(path_to_weight,
                       custom_objects={'correlate':correlate,
                                       'mse_var':mse_var})
    
    if args.downsampling:
        x_test, y_true, output_len = load_data(args.seq2seq,
                                               args.reversed_seq,
                                               command_line_arguments)
        sample_len = int(args.window) // output_len
    else:
        x_test, y_true = load_data(args.seq2seq,
                                   args.reversed_seq,
                                   command_line_arguments)

    # renormalisation of by applying the log function
    y_true = y_true.astype(np.float32)
    y_true[y_true > 0] = np.log(y_true[y_true > 0])

    y_pred = model.predict(x_test)
    
    if args.downsampling:
        y_pred = np.tile(y_pred, [1, 1, sample_len])
        y_pred = np.concatenate([np.concatenate(row, axis=0) for row in y_pred])
    else:
        y_pred = y_pred.reshape((y_pred.shape[0] * y_pred.shape[1],))
    np.save(path_to_results, y_pred)
    correlation = pearsonr(y_pred, y_true)[0]
    print('Correlation between true and pred :', correlation)

    fig, ax = plt.subplots()
    ax.plot(y_pred, 'b', label='prediction')
    ax.plot(y_true, 'r', label='experimental')
    ax.legend()
    plt.title('Experimental and predicted' +
              'occupancy on chr 16 for model{}'.format(args.weight_file[6:]))
    plt.show()

if __name__ == '__main__':
    main()