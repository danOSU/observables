#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 14:27:21 2022

@author: ozgesurer
"""

import os
import dill as pickle
import numpy as np
import time
from split_data import generate_split_data
from surmise.emulation import emulator
from plotting import plot_UQ, plot_R2
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

####################################################
# Note: This script takes on avg. 512 sec.
####################################################
seconds_st = time.time()

# Note: Here we can create different funcs to split data into training and test
f_train, f_test, theta_train, theta_test, sd_train, sd_test = generate_split_data()

x_np = np.arange(0, f_train.shape[1])[:, None]
x_np = x_np.astype('object')

####################################################
# Construct your own PCA-here independent columns
a1 = np.mean(f_train, 0)
a2 = np.std(f_train, 0)
fs = (f_train-a1) / a2

cov_mat = np.diag(np.diag(np.cov(fs.T)))#np.cov(fs.T) - np.diag(np.mean(np.square(dsdh / a2),0))
W, V = np.linalg.eigh(cov_mat)
U = np.flip(V, 1)

standardpcinfo = {'offset': a1,
                  'scale': a2,
                  'extravar': 0*a2,
                  'U': U,  # optional
                  'S': np.sqrt(np.flip(W)),  # optional
                  }
####################################################

is_train = False
emu_path = 'VAH_PCSK_ind.pkl'
method_name = 'PCSK'

if (os.path.exists(emu_path)) and (is_train==False):
    print('Saved emulators exists and overide is prohibited')
    with open(emu_path, 'rb') as file:
        emu_tr = pickle.load(file)    
else:
    emu_tr = emulator(x=x_np,
                      theta=theta_train,
                      f=f_train.T,
                      method=method_name,
                      args={'standardpcinfo': standardpcinfo,
                            'verbose': True,
                            'simsd': sd_train.T})
    
    if (is_train==True) or not(os.path.exists(emu_path)):
        with open(emu_path, 'wb') as file:
            pickle.dump(emu_tr, file)
            
    
pred_test = emu_tr.predict(x=x_np, theta=theta_test)
pred_test_mean = pred_test.mean()
pred_test_var = pred_test.var()

# Plotting diagnostics
plot_UQ(f_test, pred_test_mean.T, np.sqrt(pred_test_var.T), method=method_name)
plot_R2(pred_test_mean, f_test.T)

seconds_end = time.time()
print('Total time:', seconds_end - seconds_st)