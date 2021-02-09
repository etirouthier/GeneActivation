#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:11:31 2019

@author: routhier
"""

import numpy as np
import h5py
import argparse
import os


from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf


from MyModuleLibrary.array_modifier import rolling_window
from MyModuleLibrary.mykeras.losses import correlate, mae_cor
from prediction import main as pred


def _makebatch(nuc_seq, pos, mutation, window):    
    local_seq = np.copy(nuc_seq)
    
    if local_seq[pos] == mutation:
        no_predict=True
    else:
        no_predict=False

    local_seq[pos] = mutation
    seq = local_seq[pos - window + 1 : pos + window]
    
    x_one_hot = (np.arange(seq.max()) == seq[...,None]-1).astype(int)
    x_one_hot = x_one_hot.reshape(x_one_hot.shape[0], 4)

    x_slide = rolling_window(x_one_hot, window=(window,4))

    return x_slide.reshape(x_slide.shape[0],
                           x_slide.shape[2],
                           1,
                           x_slide.shape[3]), no_predict

def _mutation_score(y_wild, y_syn, pos, window):    
    y_true = y_wild[pos - window + 1 : pos + 1]
    y_true = y_true.reshape(len(y_true))
    
    y_syn = y_syn.reshape(len(y_syn))
    return np.mean(np.abs(y_syn - y_true))

def _parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model',
                        help='''model with which the prediction will be made
                        ''')
    parser.add_argument('-d',
                        '--directory',
                        help='''directory that contains the dna seq in hdf5.
                        ''')
    parser.add_argument('-c',
                         '--chromosome',
                         help='''chromosome on which to perform the analysis''')
    parser.add_argument('-s',
                         '--start',
                         help='''where to restart the mutazome in a case of
                         interruption''',
                         default=None)
  
    return parser.parse_args(args)

def main(command_line_arguments=None):
    """
        Compute the mutation score for every single mutation possible along 
        the S.cerevisiae genome.
        
        The mutation score is the mean absolute error between the prediction of
        the model for the sequence surrounding the nucleotid with and without a
        mutation. Surronding means here being at less than a window from the 
        position.
    """
    args = _parse_arguments(command_line_arguments)
    path_to_program = os.path.dirname(__file__)

    if not args.start:
        pred(['--weight_file', os.path.basename(args.model), '--directory',
              args.directory,
              '--file', 'plus_strand_coverage_ypd.csv',
              '--model', 'cnn', '--test', args.chromosome])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  
    config.log_device_placement = True 
    sess = tf.Session(config=config)
    set_session(sess)

    model = load_model(os.path.join(path_to_program,
                                    'Results_RNA_seq',
                                    os.path.basename(args.model)),
                       custom_objects={'correlate' : correlate,
                                       'mae_cor' : mae_cor})

    y_pred = np.load(os.path.join(path_to_program,
                                  'Results_RNA_seq',
                                  'y_pred_fw' + os.path.basename(args.model)[6 : -5]) + '_applied_on_chr' + args.chromosome + '.npy')

    f = h5py.File(os.path.join(path_to_program, 'seq_sacCer3',
                               args.directory,
                               'chr' + args.chromosome + '.hdf5'), 'r')
    nucleotid = np.array(f['data'])
    f.close()

    WINDOW = 3001
    mutazome = np.zeros((len(nucleotid), 4))
    if args.start:
        start = int(args.start)
        mutazome = np.load(os.path.join(path_to_program, 'Results_RNA_seq',
                                 'mutazome' + os.path.basename(args.model)[6 : -5])\
                                 + 'applied_on_chr' + args.chromosome + '.npy')
    else:
        start = WINDOW

    for pos in range(start, len(nucleotid) - WINDOW):
        if pos % 100 == 0:
            print('nucleotid_number : ', pos)
            np.save(os.path.join(path_to_program, 'Results_RNA_seq',
                                 'mutazome' + os.path.basename(args.model)[6 : -5])\
                                 + 'applied_on_chr' + args.chromosome,
                                 mutazome)

        for mutation in [1, 2, 3, 4]:
            x_syn, no_predict = _makebatch(nucleotid, pos, mutation, WINDOW)

            if not no_predict:
                y_syn = model.predict(x_syn)

                mutazome[pos, mutation - 1] = _mutation_score(y_pred,
                                                                 y_syn,
                                                                 pos,
                                                                 WINDOW)

if __name__ == '__main__':
    main()
    
