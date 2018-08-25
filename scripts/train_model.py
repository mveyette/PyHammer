import numpy as np
import pickle
import os, sys
import time
from scipy.ndimage import uniform_filter1d
import tensorflow as tf
import re

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
     
    return np.array(labels_train), np.array(spectra_train), np.array(labels_test), np.array(spectra_test)
    
def evaluate_predictions(test, pred, labels, plotdir=None):
    """
    Calculate and print some summary stats for the predictions.
    """
    TPRs = []
    FPRs = []
    precisions = []
    F1s = []
    confusion_matrix = np.zeros(2*[len(labels)], dtype=int)
    for i,label in enumerate(labels):
        w_test = test == label
        w_pred = pred == label
        confusion_matrix[i] = [np.sum(pred[w_test] == lbl) for lbl in labels]
        P = np.sum(w_test)
        N = np.sum(~w_test)
        TP = np.sum( w_test &  w_pred)
        FP = np.sum(~w_test &  w_pred)
        TPR = TP/P # true positive rate
        FPR = FP/N # true positive rate
        precision = TP/(TP+FP)
        recall    = TPR # same as TPR
        F1 = 2. * (precision*recall) / (precision+recall)
        print('\n')
        print('For label {}:'.format(label))
        print('Number in test sample = {:.0f}'.format(P))
        print('TPR       = {:.2f}'.format(TPR))
        print('FPR       = {:.2f}'.format(FPR))
        print('Precision = {:.2f}'.format(precision))
        print('F1        = {:.2f}'.format(F1))
        TPRs.append(TPR)
        FPRs.append(FPR)
        precisions.append(precision)
        F1s.append(F1)
    print('\n')
    print('Average TPR       = {:.2f} +\- {:.2f}'.format(np.nanmean(TPRs),np.nanstd(TPRs)))
    print('Average FPR       = {:.2f} +\- {:.2f}'.format(np.nanmean(FPRs),np.nanstd(FPRs)))
    print('Average precision = {:.2f} +\- {:.2f}'.format(np.nanmean(precisions),np.nanstd(precisions)))
    print('Average F1s       = {:.2f} +\- {:.2f}'.format(np.nanmean(F1s),np.nanstd(F1s)))
    
    print('\n')
    print('Confusion matrix:')
    print(np.array_str(confusion_matrix))
    print('A ^')
    print('c |')
    print('t |')
    print('u |')
    print('a |')
    print('l |___________>')
    print('   Predicted')
        

###############################################################################
##
## Main
##
###############################################################################
    
def main():

    ## Directory to save model in
    ## Weird permission issue are happening and I can't figure out.
    ## Maybe related to the save dir.
    ## Probably some issue with files not being closed and running in IPython.
    ## OMG it was a stupid dropbox issue. Saving to a non-dropbox dir fixed it.
    ## Sometimes tf needs to just chill the f out.
    #model_dir = os.path.join(os.path.split(os.path.dirname(__file__))[0],
    #                         'classifier','models','temp_'+str(int(time.time())))
    model_dir = os.path.join(r'C:\Users\Mark\HAMMER_TEMP_MODELS_DIR','temp_'+str(int(time.time())))
    figdir = os.path.join(model_dir,'figures')
    print('Model dir is {}'.format(model_dir))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        os.makedirs(figdir)
    else:
        print("OUTPUT DIRECTORY ALREADY EXISTS - overwriting any data there")

    ## Read in the training data
    labels_full, spectra_full = read_data()
    unique_labels = np.unique(labels_full)
    
    ## Whiten the data, save whitening parameters
    ## NOTE - really, the whitening parameters should come from only the train data
    whiten_mean, whiten_std = whiten_spectra(spectra_full)
    np.savetxt(os.path.join(model_dir, 'whitening_parameters.txt'),
               [whiten_mean, whiten_std])
    
    ## Separate into training and testing sets
    labels_train, spectra_train, labels_test, spectra_test \
      = split_train_test(labels_full, spectra_full, training_fraction=0.5)
    
    ## Alright tf, let it out
    tf.logging.set_verbosity(tf.logging.INFO)
    
    ## Create the classifier
    c = classifier.NNClassifier()
    ## Run Config, save every N steps, no time listener
    run_config = tf.estimator.RunConfig(save_checkpoints_steps = 1000,
                                        keep_checkpoint_max=100)
    c.new_model(len(spectra_full[0]), unique_labels, model_dir=model_dir,
                hidden_units=[128,128,128], dropout=0.1, config=run_config,
                optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.001,
                                                            l1_regularization_strength=0.5))

    ## Train it
    c.train(spectra_train, labels_train, batch_size=128, steps=5e3)
    
    ## Give tf a chill pill
    tf.logging.set_verbosity(tf.logging.WARN)
    
    ## At each checkpoint, evaluate the model
    evals_train = []  ## accuracy score at each checkpoint
    evals_test  = []
    with open(os.path.join(model_dir,'checkpoint'), 'r') as checkpoints_file:
        for line in checkpoints_file:
            m = re.match(r'all_model_checkpoint_paths: "(.*)"', line)
            if m is not None:
                checkpoint = m.group(1)
                checkpoint_path = os.path.join(model_dir, checkpoint)
                evals_train.append(c.evaluate(spectra_train, labels_train,
                                              checkpoint_path=checkpoint_path,
                                              name='{}_train'.format(checkpoint)))
                evals_test.append( c.evaluate(spectra_test , labels_test ,
                                              checkpoint_path=checkpoint_path,
                                              name='{}_test'.format(checkpoint)))
                print('\n')
                print('At checkpoint {}:'.format(checkpoint))
                print('Train set accuracy: {accuracy:0.3f}'.format(**evals_train[-1]))
                print('Test  set accuracy: {accuracy:0.3f}'.format(**evals_test[-1] ))
    
    ## Get predictions of test set to do more diagnostics
    pred_result = c.predict(spectra_test, preprocess=False, postprocess=False)
    
    ## Check each prediction
    pred_test = labels_test.copy()
    pred_test.fill('')
    print('\n')
    for i, (pred_dict, expec) in enumerate(zip(pred_result, labels_test)):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        pred_test[i] = unique_labels[class_id]
        #print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        #      unique_labels[class_id], 100*probability, expec))
              
    ## Check predictions in each class
    evaluate_predictions(labels_test, pred_test, unique_labels, plotdir=figdir)
    
    
if __name__ == '__main__':
    main()
