#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 21:42:11 2022

@author: ozgesurer
"""

import os,sys
import dill as pickle
import numpy as np
import time
from split_data import generate_split_data

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

os.chdir('/Users/dananjayaliyanage/git/observables/vah_design/emulation')
sys.path.append('/Users/dananjayaliyanage/git/surmise/')
#sys.path.append('/Users/dananjayaliyanage/git/surmise/surmise/emulationsupport')

from surmise.emulation import emulator
#from surmise.emulationsupport.matern_covmat import covmat as __covmat
from plotting import plot_UQ, plot_R2
#import pyximport
#pyximport.install(setup_args={"include_dirs":np.get_include()},
#                  reload_support=True)

####################################################
# Note: Each method takes on avg. 350 sec.
####################################################
seconds_st = time.time()

# Note: Here we can create different funcs to split data into training and test
f_train, f_test, theta_train, theta_test, sd_train, sd_test = generate_split_data()

x_np = np.arange(0, f_train.shape[1])[:, None]
x_np = x_np.astype('object')

##########################################################
# Note: Pick method_name = 'PCGPwM' or 'PCGPR' or 'PCSK'
##########################################################
method_name = 'PCGPwM'
is_train = False
emu_path = 'VAH_' + method_name + '.pkl' 

if (os.path.exists(emu_path)) and (is_train==False):
    print('Saved emulators exists and overide is prohibited')
    with open(emu_path, 'rb') as file:
        emu_tr = pickle.load(file)    
else:
    print('training emulators')
    if method_name == 'PCGPwM':
        emu_tr = emulator(x=x_np,
                          theta=theta_train,
                          f=f_train.T,
                          method='PCGPwM',
                          args={'epsilon': 0.05})
        
    elif method_name == 'PCGPR':
        prior_min = [10, -0.7, 0.5, 0, 0.3, 0.135, 0.13, 0.01, -2, -1, 0.01, 0.12, 0.025, -0.8, 0.3]
        prior_max = [30, 0.7, 1.5, 1.7, 2, 0.165, 0.3, 0.2, 1, 2, 0.25, 0.3, 0.15, 0.8, 1]
        prior_dict = {'min': prior_min, 'max': prior_max}
        emu_tr = emulator(x=x_np,
                          theta=theta_train,
                          f=f_train.T,
                          method='PCGPR',
                          args={'epsilon': 0.05,
                                'prior': prior_dict})
    elif method_name == 'PCSK':
        emu_tr = emulator(x=x_np,
                          theta=theta_train,
                          f=f_train.T,
                          method='PCSK',
                          args={'epsilonPC': 0.01, # this does not mean full errors
                                'simsd': np.absolute(sd_train.T)})


    if (is_train==True) or not(os.path.exists(emu_path)):
        with open(emu_path, 'wb') as file:
            pickle.dump(emu_tr, file)
            
pred_test = emu_tr.predict(x=x_np, theta=theta_test)
pred_test_mean = pred_test.mean()
pred_test_var = pred_test.var()

# Plotting diagnostics
plot_UQ(f_test, pred_test_mean.T, np.sqrt(pred_test_var.T), method=method_name)
plot_R2(pred_test_mean, f_test.T, method=method_name)
# print(emu_tr._info['emulist'])



seconds_end = time.time()
print('Total time:', seconds_end - seconds_st)