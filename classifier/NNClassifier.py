from . import Classifier
from .reader import read_all_spectra

import numpy as np
import tensorflow as tf

class NNClassifier(Classifier):

    def __init__(self, load_model_dir=None):
        """
        Simple feed forward neurnal network model
        
        inputs
            load_model_dir - if passed, will load a model from given directory
            
        If starting a new model, run NNClassifier.new_model(...) after initializing
        """
        
        if load_model_dir:
            raise RuntimeError('Whoops. Not implemented yet')
            
            ## Load pickeled preprocessing function (applied to raw features)
            ## Load pickeled postprocessing function (applied to labels before output)
            ## Load tf model
       
    def new_model(self, nfeatures, labels, model_dir='.', hidden_units=[128], **kwargs):
    
        ## Create list of labels and a dictionary to lookup the index of a label
        self.labels = labels
        self.label_index_lookup = {}
        for i,label in enumerate(labels):
            self.label_index_lookup[label] = int(i)
        
        ## Create the tf estimator
        ## First define the features, a 1D vector same length as a spectrum
        feature_columns = [tf.feature_column.numeric_column('flux', shape=nfeatures)]
        ## Get number of labels
        nlabels = len(labels)
        ## Define estimator
        self.estimator = self.__create_estimator(hidden_units, feature_columns,
                                                 nlabels, model_dir, **kwargs)
                                                    
    def __create_estimator(self, hidden_units, feature_columns, nlabels, model_dir, **kwargs):
        """Function to create the estimator. Can be overridden by extended classes"""
        return tf.estimator.DNNClassifier(hidden_units, feature_columns,
                                          n_classes=nlabels, model_dir=model_dir, **kwargs)
                                            
    def __train_input_fn(self):
        """Returns a random batch of training data"""
        ## To ensure unbiased training, grab random labels to define batch
        labels = np.random.choice(np.unique(self.labels_train), self.batch_size)
        ## Then grab a random spectrum from each label
        spectra = np.zeros((self.batch_size, len(self.spectra_train[0])))
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
        
    def __predict_input_fn(self):
        """Returns the test data"""
        ## Recast spectra into dictionary for estimator
        features = {'flux': self.spectra_test}
        return features
        
    def train(self, spectra, labels, steps=1e4, batch_size=128):
        """Function to start the training routine"""
        self.batch_size = batch_size
        self.spectra_train = np.array(spectra)
        self.labels_train  = np.array(labels)
        self.estimator.train(input_fn=self.__train_input_fn, steps=steps)
        
    def evaluate(self, spectra, labels):
        """Evaluate the model"""
        self.spectra_test = np.array(spectra)
        self.labels_test  = np.array(labels)
        eval_result = self.estimator.evaluate(input_fn=self.__test_input_fn, steps=1)
        return eval_result
        
    def predict(self, spectra, preprocess=True, postprocess=True):
        if preprocess:
            ## Do preprocessing here
            pass
        self.spectra_test = np.array(spectra)
        predictions = self.estimator.predict(input_fn=self.__predict_input_fn)
        if postprocess:
            ## Do post processing here
            pass
        return predictions