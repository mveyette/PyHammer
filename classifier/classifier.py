import numpy as np

class Classifier():
    """Base class to do the classification"""
    
    def __init__(self):
        """Initialization"""
        raise RuntimeError('Can\'t initialize base class. Used an extended class')
        
    def train(self):
        """Train the model"""
        raise RuntimeError('No training method defined.')
    
    def predict(self):
        """Predict labels given a set of features"""
        raise RuntimeError('No predicition method defined.')
        
    def test(self):
        """Test predicitions"""
        raise RuntimeError('No testing method defined.')