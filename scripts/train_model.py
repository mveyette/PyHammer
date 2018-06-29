import numpy as np
import pickle
import os, sys
import time
from scipy.ndimage import uniform_filter1d

sys.path.append(os.path.split(os.path.dirname(__file__))[0])
import classifier

###############################################################################
##
## Data Preprocessing
##
###############################################################################
  
def clean_spectrum(spectrum,size=10):
    """Replaces NaNs with the average of surrounding points"""
    mask = ~np.isfinite(spectrum)
    while np.sum(mask) > 0:
        masked = np.where(mask, 0, spectrum)
        weights = 1.  / (1. - uniform_filter1d(mask.astype(float), size))
        filtered = weights*uniform_filter1d(masked, size)
        spectrum[mask] = filtered[mask]
        mask = ~np.isfinite(spectrum)
    
def downsample_spectrum(spectrum):
        return spectrum[::10]

def read_data():
    """Read in the full training set and clean it up"""
    labels, spectra = classifier.reader.read_all_spectra('', quick=True)
    labels  = np.array(labels)
    ## We don't need the full-res data, downsample
    spectra = np.array([downsample_spectrum(s) for s in spectra])
    ## Clean up the NaNs
    for spectrum in spectra: clean_spectrum(spectrum)
    return labels, spectra
    
def whiten_spectra(spectra):
    """Whiten pixel-by-pixel"""
    mean = np.mean(spectra, 0)
    std  = np.std(spectra, 0)
    spectra -= mean
    spectra /= std 
    return mean, std
    
###############################################################################
##
## Training/Evaluation Functions
##
###############################################################################
    
def split_train_test(labels, spectra, training_fraction=0.8):
    """
    Randomly splits the data into training and testing sets.
    The split is done by-label. For each label, training_fraction of
    the samples are put in the training set and the rest in the test set
    """
    labels_train, spectra_train, labels_test, spectra_test = [], [], [], []
    for label in np.unique(labels):
        good = labels == label
        shuffled_index = np.random.permutation(np.arange(np.sum(good)))
        split_index = int(training_fraction*np.sum(good))
        labels_train.extend(  labels[good][shuffled_index[:split_index]])
        spectra_train.extend(spectra[good][shuffled_index[:split_index]])
        labels_test.extend(   labels[good][shuffled_index[split_index:]])
        spectra_test.extend( spectra[good][shuffled_index[split_index:]])

    # Old way - not by label
    #shuffled_index = np.random.permutation(np.arange(len(labels)))
    #split_index = int(training_fraction*len(labels))
    #labels_train  =  labels[shuffled_index[:split_index]]
    #spectra_train = spectra[shuffled_index[:split_index]]
    #labels_test   =  labels[shuffled_index[split_index:]]
    #spectra_test  = spectra[shuffled_index[split_index:]]
     
    return labels_train, spectra_train, labels_test, spectra_test

###############################################################################
##
## Main
##
###############################################################################
    
def main():

    ## Directory to save model in
    model_dir = os.path.join(os.path.split(os.path.dirname(__file__))[0],
                             'classifier','models','temp_'+str(time.time()))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        os.makedirs(os.path.join(model_dir,'figures'))
    else:
        print("OUTPUT DIRECTORY ALREADY EXISTS - overwriting any data there")

    ## Read in the training data
    labels_full, spectra_full = read_data()
    unique_labels = np.unique(labels_full)
    
    ## Whiten the data, save whitening parameters
    whiten_mean, whiten_std = whiten_spectra(spectra_full)
    np.savetxt(os.path.join(model_dir, 'whitening_parameters.txt'),
               [whiten_mean, whiten_std])
    
    ## Separate into training and testing sets
    labels_train, spectra_train, labels_test, spectra_test = split_train_test(labels_full, spectra_full)
    
    ## Create the classifier
    c = classifier.NNClassifier()
    c.new_model(len(spectra_full[0]), unique_labels,
                model_dir=model_dir,
                hidden_units=[128,128,128], dropout=0.1)

    ## Train it
    c.train(spectra_train, labels_train, batch_size=64, steps=1e3)
    
    ## Simple bulk evaluation
    eval_result = c.evaluate(spectra_test, labels_test)
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    
    ## Get predictions of test set to do more diagnostics
    predictions = c.predict(spectra_test, preprocess=False, postprocess=False)
    
    for pred_dict, expec in zip(predictions, labels_test):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
              unique_labels[class_id], 100*probability, expec))
    
if __name__ == '__main__':
    main()
