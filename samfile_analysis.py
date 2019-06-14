#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:57:40 2019

@author: routhier
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse

import pysam

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bedfile_to_csv',
                        action = 'store_true',
                        help = '''Use it to change a .bed file into two .csv
                        file, one for each strand''')
    parser.add_argument('-c', '--count_coverage', 
                        action = 'store_true',
                        help = '''Use it to count the number of reads into all
                        genes (from a .bam file)''')
    parser.add_argument('-r', '--replica',
                        help = ''' Code SRR of the replica we want to analyse''')
    return  parser.parse_args()


def bedfile_to_csv(replica) :
    '''
        Function that takes the .bed file containing the per base density of
        reads and returns two .csv file with the density of reads on the genes
        of the forward and the backward strands.
        
        The .bed file is obtained using bedtools genomeCoverage and consist of
        3 columns : the chromosome number (with roman number), the position of
        the nucleotid and the number of reads that match this position. The
        .csv files are made of the three same columns (called respectively 'chr',
        'pos' and 'value'). Nevertheless, on the forward csv file the density
        of reads outside genes in a forward strand is set to zeros.

        Args:
            replica: str, code SRR of the replica we want to analyse.
        Returns:
            Two .csv files. 
    '''
    bedfile = pd.read_csv('/home/invites/routhier/Projet_RNA_seq/Start_data/' + \
                          + replica + '/' + replica + '.bed', sep = '\t')
    bedfile_fw = pd.DataFrame()
    bedfile_bw = pd.DataFrame()
    
    roman_num = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI']
    
    for (i,j) in zip(range(1,17), roman_num):
        annotation = pd.read_csv('./Start_data/annotation_s_cerevisiae/' + \
                                 'Saccharomyces_cerevisiae.R64-1-1.95.' + \
                                 'chromosome.' + j + '.gff3', sep = '\t')
        annotation = annotation[annotation.type == 'gene']
        
        forward_start = annotation[annotation.strand == '+'].start.values
        forward_stop = annotation[annotation.strand == '+'].stop.values
        backward_start = annotation[annotation.strand == '-'].start.values
        backward_stop = annotation[annotation.strand == '-'].stop.values
        
        bedfile_fw_ = bedfile[bedfile.chr == 'chr' + j].copy()
        
        bedfile_fw_.chr = 'chr' + str(i)
        
        bedfile_fw_.value.iloc[: int(forward_start[0]) - 1] = 0
        bedfile_fw_.value.iloc[int(forward_stop[-1]) - 1 :] = 0
        for start, stop in zip(forward_start[1:] - 1, forward_stop[:-1] - 1):
            bedfile_fw_.value.iloc[int(stop) : int(start)] = 0
        
        bedfile_bw_ = bedfile[bedfile.chr == 'chr' + j].copy()
        
        bedfile_bw_.chr = 'chr' + str(i)
        
        bedfile_bw_.value.iloc[0 : int(backward_start[0]) - 1] = 0
        bedfile_bw_.value.iloc[int(backward_stop[-1]) - 1:] = 0
        for start, stop in zip(backward_start[1:] - 1, backward_stop[:-1] - 1):
            bedfile_bw_.value.iloc[int(stop) : int(start)] = 0
    
        bedfile_fw = bedfile_fw.append(bedfile_fw_)
        bedfile_bw = bedfile_bw.append(bedfile_bw_)
    
    
    
    bedfile_fw.to_csv('./Start_data/' + replica + '_forward.csv')
    bedfile_bw.to_csv('./Start_data/' + replica + '_backward.csv')
    
    plt.plot(bedfile_fw.value.values, 'b')
    plt.plot(-bedfile_bw.value.values, 'r')
    plt.show()

def count_coverage(replica):
    '''
        Takes the replica code in entry and returns a .csv file with genes and 
        the number of reads that map them.

        This function returns a .csv file with column names being 'chr', 'name'
        and 'counts', standing for the chromosome, the names of genes and the
        number of reads that map every gene. We need for that to pass two file :
        a .csv file with the annotation and especially the gene positions and 
        a sorted .bam file corresponding to the replica. An index file 
        corresponding to the .bam file needs to be present in the same 
        directory as the .bam file (file .bam.bai with the sama name as the .bam
        file otherwise). Such an index can be generated using samtools index 
        file_name.bam in command line.

        Args:
            replica: string, the code SRR for the consider replica
        returns:
            .csv file with the name of genes and their coverage.
    '''
    roman_num = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI']
    
    bamfile = pysam.AlignmentFile('/home/invites/routhier/Projet_RNA_seq/Start_data/' + \
                                  + replica + '/' + replica + '_sorted.bam', "rb")
    count_per_genes = pd.DataFrame()
    
    for (i,j) in zip(range(1,17), roman_num):
        annotation = pd.read_csv('/home/invites/routhier/Projet_RNA_seq/' + \
                                 'Start_data/annotation_s_cerevisiae/' + \
                                 'Saccharomyces_cerevisiae.R64-1-1.95.chromosome.' + \
                                 j + '.gff3', sep = '\t')
        annotation = annotation[annotation.type == 'gene']
        
        count_per_genes_ = pd.DataFrame()
        count_per_genes_['chr'] = ['chr' + str(i)  for _ in range(len(annotation))]
        count_per_genes_['name'] = [annotation['4'].iloc[_].split(';')[0][8:] for _ in range(len(annotation))]
        count_per_genes_['counts'] = [bamfile.count('chr'+j, start=annotation.start.iloc[_], stop=annotation.stop.iloc[_]) for _ in range(len(annotation))]
        
        count_per_genes = count_per_genes.append(count_per_genes_)
        print(str(i) + '...')
        print(len(count_per_genes))
        
    count_per_genes.to_csv('./Start_data/' + replica + '_count_per_genes.csv')

def main():
    args = parse_arguments()
    
    if args.bedfile_to_csv:
        bedfile_to_csv(args.replica)
    elif args.count_coverage:
        count_coverage(args.replica)

if __name__ == '__main__' :
    main()
        