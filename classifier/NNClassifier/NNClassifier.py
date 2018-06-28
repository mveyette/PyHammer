from .. import Classifier
from .. import PATH_TO_ALL_SPECTRA
from ..reader import read_all_spectra

import numpy as np
import tensorflow as tf
from tensorflow.estimator import DNNClassifier

class NNClassifier(Classifier):

    def __init__(self, hidden_units, **kwargs):
        """
        Simple feed forward neurnal network model
        
        inputs:
            hidden_units - 1d array-like containing the number of nodes in each hiden layer
            **kwargs - any other keyword arguments passed to tf.estimator class
        """
        
        ## First, read in training the data so we know what we are working with
        labels, spectra = read_all_spectra(PATH_TO_ALL_SPECTRA)
        self.train_labels_full  = labels
        self.train_spectra_full = spectra
        
        ## Save whitening parameters
        self.whiten_mean = np.mean(self.train_specra_full, 0)
        self.whiten_std  = np.std(self.train_specra_full, 0)
        
        ## Create the tf estimator
        ## First define the features, a 1D vector same length as a spectrum
        feature_columns = [tf.feature_column.numeric_column('x',
                           shape=np.shape(self.train_spectra_full[0])]
        ## Get number of classes
        nclasses = len(np.unique(self.train_labels_full))
        ## Define estimator
        self.estimator = tf.DNNClassifier(hidden_units, feature_columns, **kwargs)  
        
    def _whiten(self, spectrum):
        return((spectrum - self.whiten_mean)/self.whiten_std)
    
    def _unwhiten(self, spectrum):
        return(self.whiten_std * spectrum + self.whiten_mean)
    
        