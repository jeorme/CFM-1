# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:51:34 2017

@author: NLS2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

#####
# Some stats on the data
#####

#%% Plot the distributions

for i in range(2, 23):
    data = np.transpose(input_matrix)[i]
    plt.hist(data, bins=100)
    plt.title(input_headers[i])
    save_path = 'plots/{}.png'.format(input_headers[i])
    plt.pause(1e-17)
    plt.savefig(save_path)
    plt.show()
    
#%% Investigation on data that are near 0
'''
We can see that a lot of data are near 0: all the squentries, and nb_trade
We need to further investigate those data
'''
indices = [18,19,20,21,22]

for idx in indices:
    data = np.transpose(input_matrix)[idx]
    header = input_headers[idx]
    values = []
    for i in range(len(data)):
        if data[i] > 0 and data[i] < 100:
            values.append(data[i])
    values = np.array(values)
    print np.max(values)
    print np.min(values)
    print header, float(len(values))/float(len(data))
    plt.hist(values, bins=20)
    plt.title(header)
    save_path = 'plots/{}_filtered.png'.format(header)
    plt.pause(1e-17)
    plt.savefig(save_path)
    plt.show()