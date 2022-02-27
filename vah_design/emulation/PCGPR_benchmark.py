#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 17:29:55 2022

@author: ozgesurer
"""
import os
import pickle
import numpy as np
import time
from split_data import generate_split_data
from surmise.emulation import emulator
from plotting import plot_UQ, plot_R2

####################################################
# Note: This script takes on avg. 460 sec.
####################################################
seconds_st = time.time()

# Note: Here we can create different funcs to split data into training and test
f_train, f_test, theta_train, theta_test, sd_train, sd_test = generate_split_data()

print(f_train.shape)
print(f_test.shape)


prior_min = [10, -0.7, 0.5, 0, 0.3, 0.135, 0.13, 0.01, -2, -1, 0.01, 0.12, 0.025, -0.8, 0.3]
prior_max = [30, 0.7, 1.5, 1.7, 2, 0.165, 0.3, 0.2, 1, 2, 0.25, 0.3, 0.15, 0.8, 1]
prior_dict = {'min': prior_min, 'max': prior_max}

x_np = np.arange(0, f_train.shape[1])[:, None]
x_np = x_np.astype('object')

emu_tr = emulator(x=x_np,
                  theta=theta_train,
                  f=f_train.T,
                  method='PCGPR',
                  args={'epsilon': 0.1,
                        'prior': prior_dict})


pred_test = emu_tr.predict(x=x_np, theta=theta_test)
pred_test_mean = pred_test.mean()
pred_test_var = pred_test.var()

# Plotting diagnostics
plot_UQ(f_test, pred_test_mean.T, np.sqrt(pred_test_var.T), method='PCGPR')
plot_R2(pred_test_mean, f_test.T)
# print(emu_tr._info['emulist'])

seconds_end = time.time()
print('Total time:', seconds_end - seconds_st)