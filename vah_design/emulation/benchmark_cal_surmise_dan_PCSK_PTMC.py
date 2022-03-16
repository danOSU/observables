#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 19:37:50 2022

@author: ozgesurer
"""

import os, sys
os.chdir('/Users/dananjayaliyanage/git/observables/vah_design/emulation')
sys.path.append('/Users/dananjayaliyanage/git/surmise/')
import dill as pickle
import numpy as np
import time
from split_data import generate_split_data
from surmise.emulation import emulator
from surmise.calibration import calibrator
from plotting import plot_UQ, plot_R2, plot_hist, plot_density
from priors import prior_VAH
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

####################################################
# Note: Total emu time: 390 sec.  Total cal time: 500
####################################################
seconds_st = time.time()

# Note: Here we can create different funcs to split data into training and test
f_train, f_test, theta_train, theta_test, sd_train, sd_test, y, thetanames = generate_split_data()

# Combine all for calibration
fcal = np.concatenate((f_train, f_test), axis=0)
thetacal = np.concatenate((theta_train, theta_test), axis=0)
sdcal = np.concatenate((sd_train, sd_test), axis=0)

x_np = np.arange(0, fcal.shape[1])[:, None]
x_np = x_np.astype('object')

##########################################################
# Note: Pick method_name = 'PCGPwM' or 'PCGPR' or 'PCSK'
##########################################################
method_name = 'PCSK'
is_train = False
emu_path = 'VAH_' + method_name + '.pkl' 
        
prior_min = [10, -0.7, 0.5, 0, 0.3, 0.135, 0.13, 0.01, -2, -1, 0.01, 0.12, 0.025, -0.8, 0.3]
prior_max = [30, 0.7, 1.5, 1.7, 2, 0.165, 0.3, 0.2, 1, 2, 0.25, 0.3, 0.15, 0.8, 1]
prior_dict = {'min': prior_min, 'max': prior_max}

if (os.path.exists(emu_path)) and (is_train==False):
    print('Saved emulators exists and overide is prohibited')
    with open(emu_path, 'rb') as file:
        emu_tr = pickle.load(file)    
else:
    print('training emulators')
    if method_name == 'PCGPwM':
        emu_tr = emulator(x=x_np,
                          theta=thetacal,
                          f=fcal.T,
                          method='PCGPwM',
                          args={'epsilon': 0.05})
        
    elif method_name == 'PCGPR':
        emu_tr = emulator(x=x_np,
                          theta=thetacal,
                          f=fcal.T,
                          method='PCGPR',
                          args={'epsilon': 0.02,
                                'prior': prior_dict})
    elif method_name == 'PCSK':
        emu_tr = emulator(x=x_np,
                          theta=thetacal,
                          f=fcal.T,
                          method='PCSK',
                          args={'epsilonPC': 0.01, # this does not mean full errors
                                'simsd': np.absolute(sdcal.T)})


    if (is_train==True) or not(os.path.exists(emu_path)):
        with open(emu_path, 'wb') as file:
            pickle.dump(emu_tr, file)


seconds_end = time.time()
print('Total emu time:', seconds_end - seconds_st)


seconds_st = time.time()

####################################################
# CALIBRATOR
####################################################

calibrate = True
cal_path = 'VAH_' + method_name + '_calibrator_PTMC' + '.pkl' 



if (os.path.exists(cal_path)) and (calibrate==False):
    print('Saved Calibrators exists and overide is prohibited')
    with open(cal_path, 'rb') as file:
        cal = pickle.load(file)    
else:
    y_mean = np.array(y.iloc[0])
    obsvar = np.array(y.iloc[1])
    obsvar[obsvar < 10**(-6)] = 10**(-6)
    if calibrate:
        cal = calibrator(emu=emu_tr,
                         y=y_mean,
                         x=x_np,
                         thetaprior=prior_VAH,
                         method='directbayes',
                         args={'sampler': 'PTMC'},
                         yvar=obsvar)
    
    if (calibrate==True) or not(os.path.exists(cal_path)):
        with open(cal_path, 'wb') as file:
            pickle.dump(cal, file)

#plot_hist(theta_prior, theta_post,method_name)
#plot_density(theta_prior, theta_post, thetanames, method_name)

seconds_end = time.time()
print('Total cal time:', seconds_end - seconds_st)

