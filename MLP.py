# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:00:14 2017

@author: NLS2
"""

'''
Implementation of an MLP class that handles the building of the MLP and its 
training
'''

import numpy as np
import theano
import theano.tensor as T
import lasagne
import cPickle

from public_accuracy import score_function

class MLP(object):
    
    def __init__(self, architecture=[56,100,1]):
        '''
        Initialization of the network, which is a succession of dense layers
        architecture -- list
            the size of the successive layers in the network
            first one is the input size
            last one is the output size
            by default, builds a single hidden layer network
        '''
        self.build(architecture)
    
    def build(self, architecture):
        '''
        Builds the MLP whose architecture is defined in the parameters
        '''
        self.l_in = lasagne.layers.InputLayer(shape = (None,architecture[0]))
        
        self.layers = {}
        self.layers[0] = self.l_in
        
        for i in range(len(architecture)-1):
            self.layers[i+1] = lasagne.layers.DenseLayer(
                self.layers[i],
                architecture[i+1],
                nonlinearity=lasagne.nonlinearities.tanh)
        self.network = self.layers[len(self.layers)-1]
        self.network.nonlinearity = lasagne.nonlinearities.sigmoid
        print 'MLP built!'
        
        # compile the prediction function:
        test_prediction = lasagne.layers.get_output(self.network)
        self.predict_fn = theano.function([self.l_in.input_var], test_prediction)
        
    def load_weights(self, file_path):
        '''
        Load weights
        file_path -- path to the .npz file
        '''
        weights = np.load(file_path)
        nb_params = len(weights.keys())
        weights_list = []
        for k in range(nb_params):
            weights_list.append(weights['arr_'+str(k)])
        lasagne.layers.set_all_param_values(self.network, weights_list)
    
    def save_weights(self, file_path):
        '''
        save weights
        file_path -- path to .npz file
        '''
        nb_params = len(lasagne.layers.get_all_params(self.network))
        params_values = lasagne.layers.get_all_param_values(self.network)
        np.savez(file_path, *[params_values[k] for k in range(nb_params)])
        
    def train(self, X_train, 
                    y_train, 
                    X_test, 
                    y_test,
                    learning_rate = 0.001,
                    batch_size=20,
                    num_epochs=20):
        
        '''
        Launches the training of the network, with datasets (X_train,Y_train)
        and (X_test, Y_test) for testing
        The objective function is always the mean square error in this 
        regression problem.
        The optimizer used is defined by the self.updateFunction given at the 
        constructor step
        X_train, X_test: np.array of shape (n_batch, nb_features)
        Y_train, Y_test: np.array of shape (n_batch, 1)
        The predictions of the MLP are of size (n_batch, 1)
        '''
        
        target_values = T.matrix('target_output')
        predicted_values = lasagne.layers.get_output(self.network)
        predicted_values.astype(theano.config.floatX)
        
        #cost function: binary cross entropy
        cost = T.mean(lasagne.objectives.binary_crossentropy(predicted_values, 
                                                             target_values))
        all_params = lasagne.layers.get_all_params(self.network)
        
        # Compute SGD updates for training
        print("Computing updates ...")
        updates = lasagne.updates.sgd(cost, all_params, learning_rate)
        
        # Theano functions for training and computing cost
        print("Compiling functions ...")
        train = theano.function([self.l_in.input_var, target_values],
                                cost, updates=updates, mode = 'FAST_COMPILE')
        compute_cost = theano.function(
            [self.l_in.input_var, target_values], 
            cost, mode = 'FAST_COMPILE')
        
        print("Training ...")
        
        try:
            for epoch in range(num_epochs):
                # Training
                print '*'*32
                print 'Epoch', epoch+1, '/', num_epochs
                for j in range(len(X_train)/batch_size):
                    train(X_train[j:j+batch_size],
                          y_train[j:j+batch_size])
                
                # Evaluation on train:
                y_train_pred = self.predict_fn(X_train)
                for k in range(len(y_train_pred)):
                    if y_train_pred[k][0] > 0.5:
                        y_train_pred[k][0] = 1
                    else:
                        y_train_pred[k][0] = 0
                score_train = score_function(y_train, y_train_pred)
                print 'Accuracy on train: ', score_train
                
                # Evaluation on train:
                y_test_pred = self.predict_fn(X_test)
                for k in range(len(y_test_pred)):
                    if y_test_pred[k][0] > 0.5:
                        y_test_pred[k][0] = 1
                    else:
                        y_test_pred[k][0] = 0
                score_test = score_function(y_test, y_test_pred)
                print 'Accuracy on test: ', score_test
                
        except KeyboardInterrupt:
            pass
        
        
        