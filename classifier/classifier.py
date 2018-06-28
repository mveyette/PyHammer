import numpy as np

class Classifier():
    """Base class to do the classification"""
    
    def __init__(self):
        """Initialization"""
        raise RuntimeError('Can\'t initialize base class. Used an extended class')
        
    def train(self, features,labels):
        """Train the model"""
        raise RuntimeError('No training method defined.')
    
    def predict(self,features):
        """Predict labels given a set fo features"""
        raise RuntimeError('No predicition method defined.')
        
    def test(self,features,labels):
        """
        Test the model
        
        inputs:
            features - NxM array of features were N is the number of 
                       test cases and M is the number of features
            labels - NxM array of labels
        
        """
        guesses = self.predict(features)
        correct = [l == g for l,g in zip(labels,guesses)]
        print(np.sum(correct)/len(labels))