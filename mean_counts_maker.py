#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:26:35 2019

@author: routhier
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os 

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--replica', help = ''' Code of the replica we want to analyse (BY4742_rep01, Syn3C_rep01, ...) ''')
    parser.add_argument('-d', '--count_data', help = ''' File that contains the count data ''')
    return  parser.parse_args()

def chr_size():
    taille_chr={}
    taille_chr['chr1']=230218
    taille_chr['chr2']=813184
    taille_chr['chr3']=316620
    taille_chr['chr4']=1531933
    taille_chr['chr5']=576874
    taille_chr['chr6']=270161
    taille_chr['chr7']=1090940
    taille_chr['chr8']=562643
    taille_chr['chr9']=439888
    taille_chr['chr10']=745751
    taille_chr['chr11']=666816
    taille_chr['chr12']=1078177
    taille_chr['chr13']=924431
    taille_chr['chr14']=784333
    taille_chr['chr15']=1091291
    taille_chr['chr16']=948066
    return taille_chr

def mean_counts_maker(path_to_data, replica):
    '''
        This function returns two .csv files containing the mean reads counts of every gene in each
        strand for a replica.
        
        Those files are made of two columns being chr and value where value stand for the mean 
        read counts of the gene at this position. If we are out of a gene then the value at position 
        is zero whereas if we are in a gene then value at this position is the mean read counts of 
        this particular gene (i.e number of counts divided by the size of the gene).
        
        Args:
            data: csv file, containing the counts data for each gene.
            replica: string, the replica we want to analyse
        returns:
            .csv file: one for each strand.
    '''
    roman_num = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI']
    taille_chr = chr_size()
    signal_fw = pd.DataFrame()
    signal_bw = pd.DataFrame()
    data = pd.read_csv(path_to_data)
    data['name'] = [name[:-4] for name in data[data.columns[0]]]
    data = data.drop(data.columns[0], axis=1)
    
    for (i,j) in zip(range(1,17), roman_num):
        annotation = pd.read_csv('./Start_data/annotation_s_cerevisiae/Saccharomyces_cerevisiae.R64-1-1.95.chromosome.' + j + '.gff3', sep = '\t')
        annotation = annotation[annotation.type == 'gene']
        annotation['name'] = [annotation['4'].iloc[_].split(';')[0][8:] for _ in range(len(annotation))]
        
        signal_fw_ = pd.DataFrame()
        signal_fw_['value'] = np.zeros(taille_chr['chr' + str(i)])
        signal_fw_['chr'] = 'chr' + str(i)
        
        signal_bw_ = pd.DataFrame()
        signal_bw_['value'] = np.zeros(taille_chr['chr' + str(i)])
        signal_bw_['chr'] = 'chr' + str(i)
        
        for gene in annotation[annotation.strand == '+'].name:
            start = annotation[annotation.name == gene].start.values[0]
            stop = annotation[annotation.name == gene].stop.values[0]
            if data[data.name == gene][replica].values :
                signal_fw_.value[int(start) : int(stop)] = data[data.name == gene][replica].values[0]

        for gene in annotation[annotation.strand == '-'].name:
            start = annotation[annotation.name == gene].start.values[0]
            stop = annotation[annotation.name == gene].stop.values[0]
            if data[data.name == gene][replica].values :
                signal_bw_.value[int(start) : int(stop)] = data[data.name == gene][replica].values[0]
        
        signal_fw = signal_fw.append(signal_fw_)
        signal_bw = signal_bw.append(signal_bw_)
        
    return signal_fw, signal_bw
        
def main():
    args = parse_arguments()
    path_to_data = os.path.join(os.path.dirname(args.count_data), os.path.split(args.count_data)[1])

    signal_fw, signal_bw = mean_counts_maker(path_to_data, args.replica)
    
    signal_fw.to_csv('./Start_data/' + args.replica + '_fw_count_per_genes.csv')
    signal_bw.to_csv('./Start_data/' + args.replica + '_bw_count_per_genes.csv')
    
    plt.plot(signal_fw.value.values)
    plt.plot(signal_bw.value.values)
    plt.show()
    
if __name__ == '__main__':
    main()
        