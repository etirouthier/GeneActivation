#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 16:48:42 2019

@author: routhier
"""

import pandas as pd

bedfile = pd.read_csv('/home/invites/routhier/Projet_RNA_seq/Start_data/SRR7131302/SRR7131302_syn_hic.bed', sep = '\t', header = None)

bedfile.columns = ['chr', 'pos', 'value']
bedfile.chr = 'chr4'
print(bedfile.head())

bedfile.to_csv('./Start_data/SRR7131302_syn_hic.csv')
