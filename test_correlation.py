# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 18:13:45 2017

@author: NLS2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

#####
# Load the data
#####

df_input = pd.read_csv('data/training_input.csv')
input_headers = df_input.keys()
header_to_idx = {}
input_matrix = df_input.as_matrix()

df_output = pd.read_csv('data/training_output.csv', sep=';')
output_headers = df_output.keys()
output_matrix = df_output.as_matrix()

#%%

#####
# Test correlation
#####

content_list = [] # list that will contain all the correlation coefficients

nb_scenario = len(output_matrix)
for i in range(2, 23):
    print i+1, '/', 23
    for index_begin in range(8):
        indices = range(index_begin, len(input_matrix), 8)
        var_name = input_headers[i]
        offset = input_matrix[index_begin][1]
        variables = np.transpose(input_matrix)[i][indices]
        trades = np.transpose(output_matrix)[1]
        pearson_coeff = pearsonr(trades, variables)[0]
        content_line = var_name+' '+str(offset)+' : '+str(pearson_coeff)+'\n'
        content_list.append(content_line)

content = ''.join(content_list)

with open('correlations.txt', 'w') as f:
    f.write(content)
    f.close()