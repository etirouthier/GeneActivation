#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 09:34:19 2019

@author: routhier
"""
import os
import argparse

import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from MyModuleLibrary.mykeras.losses import correlate
from DataPipeline.generator import generator
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
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory',
                        help='''Directory containing the DNA sequence
                        chromosome by chromosome in .hdf5 (in seq_sacCer3)''')
    parser.add_argument('-f', '--file',
                        help="""CSV file containing the nucleosome
                        occupancy on the whole genome.""")
    parser.add_argument('-o', '--output_file',
                        help="""Name of the file were the weight
                        will be stored (in Results_nucleosome)""")
    parser.add_argument('-m', '--model',
                        help='''Name of the model to be trained''')
    parser.add_argument('-s', '--seq2seq', action='store_true',
                        help="""If the model is a seq2seq model""")
    return parser.parse_args()

def prepare_session():
    """
        Initializing the Tensorflow session.
    """
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
    K.clear_session()
    sess = tf.Session(config=config)
    K.set_session(sess)

def main():
    """
        Training the model with the arguments needed.
    """
    args = parse_arguments()
    # we get the path conducting to seq_chr_sacCer3
    path_to_directory = os.path.dirname(os.path.dirname(args.directory))
    path_to_tensorboard = os.path.join(path_to_directory, 'Tensorboard')
    path_to_output_file = os.path.join(path_to_directory, 'Results_RNA_seq',
                                       os.path.split(args.output_file)[1])
    path_to_file = os.path.join(path_to_directory, 'Start_data',
                                os.path.split(args.file)[1])
    path_to_directory = os.path.join(path_to_directory, 'seq_sacCer3',
                                     args.directory)

    num_epochs = 200

    if args.seq2seq:
        model, output_len = model_dictionary()[args.model]
        generator_train, number_of_set_train, \
        generator_val, number_of_set_val = generator(path_to_directory,
                                                     path_to_file,
                                                     output_len, args.seq2seq)
    else:
        model = model_dictionary()[args.model]
        generator_train, number_of_set_train, \
        generator_val, number_of_set_val = generator(path_to_directory,
                                                     path_to_file,
                                                     args.seq2seq)

    model.compile(optimizer='adam', loss='mae',
                  metrics=['mse', correlate])

    checkpointer = ModelCheckpoint(filepath=path_to_output_file,
                                   monitor='val_loss',
                                   verbose=0,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='min', period=1)
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                          verbose=0, mode='auto')
    tensorboard = TensorBoard(log_dir=path_to_tensorboard, update_freq=200)
    print model.summary()
    model.fit_generator(generator=generator_train,
                        steps_per_epoch=500,
                        epochs=num_epochs,
                        validation_data=generator_val,
                        validation_steps=200,
                        callbacks=[checkpointer, early, tensorboard])


if __name__ == '__main__':
    main()
    