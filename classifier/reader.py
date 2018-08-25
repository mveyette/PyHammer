"""Helper functions to read in the spectra"""

import numpy as np
from glob import glob
import os
from astropy.io import fits
import pickle

def read_all_spectra(path, quick=True):
    """Reads in all the spectra"""
    
    if quick:
        ## Read in pickles
        labels, spectra = [], []
        files = glob(os.path.join(os.path.dirname(__file__),'all_spectra_pickles','*.pkl'))
        ###################################################################
        ##
        ## using a smaller set for development, eventually remove the indexing
        ##
        ###################################################################
        for file in files:
            with open(file, 'rb') as pklfile:
                l, s = pickle.load(pklfile)
            if l[0] == 'M' and int(l[1]) <= 8:
                labels.append(l)
                spectra.append(s)
    
    else:
        ## Grab files, ignore giants
        files = glob(os.path.join(path,'*','spec-*.fits'))
        files = np.array([file for file in files if 'Giant' not in file])
        
        ## Create a Spectrum object from PyHammer
        cwd = os.getcwd() ## Save cwd
        ## Need to be in PyHammer directory for Spectrum class to function properly
        os.chdir(os.path.split(os.path.dirname(__file__))[0])
        from spectrum import Spectrum
        spec = Spectrum()
        
        labels = []
        spectra = []
        for file in files:
            
            ## Directory is label
            label = os.path.split(os.path.dirname(file))[1]
            
            ## Process spectrum just like in PyHammer
            message, ftype = spec.readFile(file, filetype='sdssdr12')
            spec.normalizeFlux()
            spec.guessSpecType()
            shift = spec.findRadialVelocity()
            spec.shiftToRest(shift)
            spec.interpOntoGrid()
            
            labels.append(label)
            spectra.append(spec.flux)
        
        ## Go back to previous directory
        os.chdir(cwd)
        
    return labels, spectra
    
def create_spectra_pickle(path):
    """
    Reads in all the spectra and pickles them from quicker reading later.
    Only needs to be run once. After, spectra can be read with read_all_spectra.
    """
    
    labels, spectra = read_all_spectra(path, quick=False)
    
    ## Pickle it
    for i,(l,s) in enumerate(zip(labels,spectra)):
        savefile = os.path.join(os.path.dirname(__file__),'all_spectra_pickles','{}.pkl'.format(i))
        with open(savefile, 'wb') as pklfile:
            pickle.dump((l,s), pklfile)