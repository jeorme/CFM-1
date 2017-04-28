# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:25:17 2017

@author: NLS2
"""

'''
Script to train a neural network on the data
The entries of the neural network: 56 entries (variables that are the most 
correlated to the presence of a trade after t=0)
    > bid_size_2 : for all 8 offsets
    > ask_size_2 : for all 8 offsets
    > bid_entry_2 : for all 8 offsets
    > ask_entry_2 : for all 8 offsets
    > bid_entropy_2 : for all 8 offsets
    > ask_entropy_2 : for all 8 offsets
    > nb_trades : for all 8 offsets

The output of the neural network if a 1-dimensional variable indicating
whether or not there is a trade after t=0
'''

import numpy as np
import theano
import theano.tensor as T
import lasagne
import cPickle
import pandas as pd
from random import shuffle
import os
import os.path

from MLP import MLP
from public_accuracy import score_function


def load_dataset():
    '''
    Loads the dataset from the CSV files and return the dataset
    Splits the dataset with 80% training, 20% test
    X_train, X_test -- np.array of shape (nb_users, 56)
    y_train, y_test -- np.array of shape (nb_users, 1)
    '''
    if not os.path.isfile('data/dataset.npz'):
        # Load the matrices from excel files:
        df_input = pd.read_csv('data/training_input.csv')
        input_matrix = df_input.as_matrix()
        
        df_output = pd.read_csv('data/training_output.csv', sep=';')
        output_matrix = df_output.as_matrix()
        
        # indices of the variables of interest
        indices_var = [8, 9, 12, 13, 16, 17, 22] 
        # splits the users indices between train and test:
        nb_users = len(output_matrix)
        user_indices = range(nb_users)
        shuffle(user_indices)
        indices_train = user_indices[0: int(0.8*nb_users)]
        indices_test = user_indices[len(indices_train):]
        
        # Build the arrays:
        X_train_list = []
        X_test_list = []
        y_train_list = []
        y_test_list = []
        
        done_train = 0
        for idx in indices_train:
            done_train+=1
            print 'building train: ', done_train, '/', len(indices_train)
            input_values = []
            for idx_var in indices_var:
                for j in range(8):
                    input_values.append(input_matrix[8*idx+j][idx_var])
            X_train_list.append(input_values)
            trade_val = output_matrix[idx][1]
            y_train_list.append([trade_val])
        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        
        done_test = 0
        for idx in indices_test:
            done_test+=1
            print 'building test: ', done_test, '/', len(indices_test)
            input_values = []
            for idx_var in indices_var:
                for j in range(8):
                    input_values.append(input_matrix[8*idx+j][idx_var])
            X_test_list.append(input_values)
            trade_val = output_matrix[idx][1]
            y_test_list.append([trade_val])
        X_test = np.array(X_test_list)
        y_test = np.array(y_test_list)
        
        # Save in .npz file
        np.savez('data/dataset.npz', X_train=X_train, 
                                     y_train=y_train,
                                     X_test=X_test,
                                     y_test=y_test)
     
    f = np.load('data/dataset.npz')
    X_train = f['X_train']
    y_train = f['y_train']
    X_test = f['X_test']
    y_test = f['y_test']
    
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_dataset()
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    mlp = MLP(architecture=[56,25,1])
    mlp.train(X_train, 
              y_train, 
              X_test, 
              y_test,
              learning_rate=0.01,
              batch_size=50,
              num_epochs=20)