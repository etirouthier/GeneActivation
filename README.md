# GeneActivation

This project is aimed at predicting the gene activation of S.cerevisiae using convolutional neural network. 
Gene activation is determined by the density of RNA-seq reads. The CNN can be trained to predict either the RNA-seq landscape
or the number of RNA-seq reads that match a every genes (which is used in DESeq protocol to detect effect of mutations).

## Predicting  effect of mutations

The goal of this project is to assess the question of predicting the effect of mutation on gene activation. 
We use for that the data coming from RNA-seq experiment that give the gene activation measure on the wild type genome of
S.cerevisiae (BY4742) and on another genome where a synthetic parts was added (YRDG190).

## Data generation

The data are not provided (too large). In seq_sacCer3 the DNA sequences of S.cerevisiae need to be added in a .hdf5 format 
(one directory per assembly). 
samfile_analysis.py and syn_hic_analysis.py can be used to generate .csv file containing the RNA-seq landscape respectively on the
wilde type genome (BY4742) and on a synthetic part of the genome (YRSG190). Those use .bed files resulting from the alignment 
of row RNA-seq results (available with accession numbers SRX4051932 and SRX4051975).
mean_counts_maker.py can be used to count the number of reads that match every gene (data not available).
