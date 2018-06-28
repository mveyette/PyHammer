from . import Classifier
from .reader import read_all_spectra

import numpy as np
import tensorflow as tf
from scipy.ndimage import uniform_filter1d 

class NNClassifier(Classifier):

    def __init__(self, hidden_units=[128,128], **kwargs):
        """
        Simple feed forward neurnal network model
        
        inputs:
            hidden_units - 1d array-like containing the number of nodes in each hiden layer
            **kwargs - any other keyword arguments passed to tf.estimator class
        """
        
        ## First, read in training the data so we know what we are working with
        ## TODO: this is slow, need to skip if reloading for prediction only
        labels, spectra = read_all_spectra('', quick=True)
        self.labels_full  = np.array(labels)
        ## We don't need the full-res data, downsample
        self.spectra_full = np.array([self.__downsample(s) for s in spectra])
        ## Clean up the NaNs
        for spectrum in self.spectra_full: self.__clean(spectrum)
        
        ## Whitening training data and save whitening parameters
        self.spectra_full /= np.nanmedian(self.spectra_full,1)[:,np.newaxis]
        self.whiten_mean = np.mean(self.spectra_full, 0)
        self.whiten_std  =  np.std(self.spectra_full, 0)
        self.spectra_full = self.__whiten(self.spectra_full)
        
        ## Create list of unique labels and a dictionary to lookup the index of a label
        self.unique_labels = np.unique(self.labels_full)
        self.label_index_lookup = {}
        for i,label in enumerate(self.unique_labels):
            self.label_index_lookup[label] = int(i)
        
        ## Create the tf estimator
        ## First define the features, a 1D vector same length as a spectrum
        feature_columns = [tf.feature_column.numeric_column('flux',
                           shape=np.shape(self.spectra_full[0]))]
        ## Get number of labels
        nlabels = len(self.unique_labels)
        ## Define estimator
        self.estimator = self.__create_estimator(hidden_units, feature_columns,
                                                nlabels, **kwargs)
                                                    
    def __create_estimator(self, hidden_units, feature_columns, nlabels, **kwargs):
        """Function to create the estimator. Can be overridden by extended classes"""
        return tf.estimator.DNNClassifier(hidden_units, feature_columns,
                                            n_classes=nlabels, **kwargs)
                                            
    def __train_input_fn(self):
        """Returns a random batch of training data"""
        ## To ensure unbiased training, grab random labels to define batch
        labels = np.random.choice(np.unique(self.labels_train), self.batch_size)
        ## Then grab a random spectrum from each label
        spectra = np.zeros((self.batch_size, len(self.spectra_full[0])))
        for i,l in enumerate(labels):
            good = self.labels_train == l
            idx = np.random.choice(np.sum(good))
            spectra[i] = self.spectra_train[good][idx]
        ## Recast into dictionary for estimator
        features = {'flux': spectra}
        ## Convert labels to integers
        ilabels = [self.label_index_lookup[l] for l in labels]
        return features, ilabels
        
    def __test_input_fn(self):
        """Returns the test data"""
        ## Test labels
        labels = self.labels_test
        ## Recast spectra into dictionary for estimator
        features = {'flux': self.spectra_test}
        ## Convert labels to integers
        ilabels = [self.label_index_lookup[l] for l in labels]
        return features, ilabels
        
    def train(self, steps=1e3, batch_size=64, training_fraction=0.8):
        """Function to start the training routine"""
        
        self.batch_size = batch_size
        
        ## Separate into training and testing sets
        shuffled_index = np.random.permutation(np.arange(len(self.labels_full)))
        split_index = int(training_fraction*len(self.labels_full))
        self.labels_train  =  self.labels_full[shuffled_index[:split_index]]
        self.spectra_train = self.spectra_full[shuffled_index[:split_index]]
        self.labels_test   =  self.labels_full[shuffled_index[split_index:]]
        self.spectra_test  = self.spectra_full[shuffled_index[split_index:]]
        
        self.estimator.train(input_fn=self.__train_input_fn, steps=steps)
        
        eval_result = self.estimator.evaluate(input_fn=self.__test_input_fn, steps=1)

        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
        
    ######################################################################
    ##
    ## Helper functions for handling spectra
    ##
    ######################################################################
    
    def __clean(self,spectrum,size=10):
        mask = ~np.isfinite(spectrum)
        while np.sum(mask) > 0:
            masked = np.where(mask, 0, spectrum)
            weights = 1.  / (1. - uniform_filter1d(mask.astype(float), size))
            filtered = weights*uniform_filter1d(masked, size)
            spectrum[mask] = filtered[mask]
            mask = ~np.isfinite(spectrum)
        
    def __downsample(self,spectrum):
        return spectrum[::10]
    
    def __whiten(self, spectrum):
        return((spectrum - self.whiten_mean)/self.whiten_std)
    
    def __unwhiten(self, spectrum):
        return(self.whiten_std * spectrum + self.whiten_mean)
    
        