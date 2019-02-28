#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:58:07 2019

@author: routhier
"""

import matplotlib.pyplot as plt

with open('/home/invites/routhier/Projet_RNA_seq/Start_data/Native_region_before_syn-HiC_design_IV_710,000-860,000.txt', 'r') as f:
    line = f.readline()    
    seq_WT = ''
    
    while line:
        line = f.readline().strip("\r\n")
        seq_WT += line

with open('/home/invites/routhier/Projet_RNA_seq/Start_data/S288C_chrIV_BK006938.2.fsa', 'r') as f:
    line = f.readline()    
    seq_sacCer3 = ''
    
    while line:
        line = f.readline().strip("\n")
        seq_sacCer3 += line

seq_WT = seq_WT[:3184] + seq_WT[3185:]
seq_WT = seq_WT[:8490] + 'T' + seq_WT[8490:]

count = 0
counts = []
for i in range(len(seq_WT)):
    if seq_WT[i] != seq_sacCer3[710006 + i]:
        count += 1
    counts.append(count)
print(seq_WT[8490 : 8520], seq_sacCer3[710006 + 8490 : 710006 + 8520])
plt.plot(counts)
plt.show()
print(len(seq_WT), count)