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
from plotting import plot_UQ
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

####################################################
# Note: This script takes on avg. 354 sec.
####################################################
seconds_st = time.time()

# Note: Here we can create different funcs to split data into training and test
f_train, f_test, theta_train, theta_test = generate_split_data()

print(f_train.shape)
print(f_test.shape)


x_np = np.arange(0, f_train.shape[1])[:, None]
x_np = x_np.astype('object')

emu_tr = emulator(x=x_np,
                  theta=theta_train,
                  f=f_train.T,
                  method='PCGPwM',
                  args={'epsilon': 0.1})


pred_test = emu_tr.predict(x=x_np, theta=theta_test)
pred_test_mean = pred_test.mean()
pred_test_var = pred_test.var()

plot_UQ(f_test, pred_test_mean.T, np.sqrt(pred_test_var.T), method='PCGP')
# print(emu_tr._info['emulist'])

seconds_end = time.time()
print('Total time:', seconds_end - seconds_st)