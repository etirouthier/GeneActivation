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

from MyModuleLibrary.array_modifier import rolling_window
from MyModuleLibrary.mykeras.losses import correlate, mse_var
from CustomModel.Models import model_dictionary

def parse_arguments():
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
    parser.add_argument('-r', '--reversed_seq', action='store_true',
                        help='In order to predict the backward strand')
    return parser.parse_args()

def load_data(seq2seq=False, reversed_seq=False):
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
    args = parse_arguments()

    window = 2001
    half_wx = window // 2
    args = parse_arguments()
    # we get the path conducting to seq_chr_sacCer3
    path_to_directory = os.path.dirname(os.path.dirname(args.directory))
    path_to_file = os.path.join(path_to_directory, 'seq_sacCer3',
                                args.directory, 'chr16.hdf5')

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
    y_true = proba[proba.chr == 'chr16'].value.values

    if reversed_seq:
        y_true = y_true[::-1]

    if seq2seq:
        _, output_len = model_dictionary()[args.model]

        if output_len % 2 == 0:
            half_len = output_len//2
        else:
            half_len = output_len//2 + 1

        x_slide = rolling_window(x_, window=(window, 4),
                                 asteps=(output_len, 4))
        x_ = x_slide.reshape(x_slide.shape[0], x_slide.shape[2],
                             x_slide.shape[3], 1)
        y_true = y_true[half_wx - half_len :
                        x_slide.shape[0]*output_len + half_wx - half_len]

    else:
        x_slide = rolling_window(x_, window=(window, 4))
        x_ = x_slide.reshape(x_slide.shape[0], x_slide.shape[2],
                             x_slide.shape[3], 1)
        y_true = y_true[half_wx : -half_wx]

    return x_, y_true

def main():
    """
        Main function of the module designed to make prediction on chromosome 4
        using the parsed arguments. The prediction are compared using pearson
        correlation to the actual RNA-seq density and the results are displayed
    """
    args = parse_arguments()
    results_path = os.path.dirname(os.path.dirname(args.directory))
    path_to_weight = os.path.join(results_path, 'Results_RNA_seq',
                                  os.path.split(args.weight_file)[1])

    if args.reversed_seq:
        path_to_results = os.path.join(results_path,
                                       'Results_RNA_seq',
                                       'y_pred_bw' + args.weight_file[6:-5]
                                       + args.file[-23:-4])
    else:
        path_to_results = os.path.join(results_path,
                                       'Results_RNA_seq',
                                       'y_pred_fw' + args.weight_file[6:-5]
                                       + args.file[-23:-4])

    model = load_model(path_to_weight,
                       custom_objects={'correlate':correlate,
                                       'mse_var':mse_var})
    x_test, y_true = load_data(args.seq2seq, args.reversed_seq)

    # renormalisation of by applying the log function
    y_true = y_true.astype(np.float32)
    y_true[y_true > 0] = np.log(y_true[y_true > 0])

    y_pred = model.predict(x_test)
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