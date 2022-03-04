#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 17:29:55 2022

@author: ozgesurer
"""
import os,sys

os.chdir('/Users/dananjayaliyanage/git/observables/vah_design/emulation')
sys.path.append('/Users/dananjayaliyanage/git/surmise/')
import math
import dill as pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
from split_data import generate_split_data
from surmise.emulation import emulator
from plotting import plot_UQ, plot_R2, st_index
from scipy.linalg import block_diag

####################################################
# Note: This script takes on avg. 11348 sec.
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

####################################################
# Construct your own PCA-here independent columns

# Let's split the observables into two.
subset = ['v22_[0 5]','v22_[ 5 10]', 'v22_[10 20]', 'v22_[20 30]', 'v22_[30 40]',
       'v22_[40 50]', 'v22_[50 60]', 'v22_[60 70]', 'v32_[0 5]', 'v32_[ 5 10]',
       'v32_[10 20]', 'v32_[20 30]', 'v32_[30 40]', 'v32_[40 50]', 'v42_[0 5]',
       'v42_[ 5 10]', 'v42_[10 20]', 'v42_[20 30]', 'v42_[30 40]',
       'v42_[40 50]']

f_train_0, f_test_0, theta_train_0, theta_test_0, sd_train_0, sd_test_0 = generate_split_data(drop_list=subset)

f_train_1, f_test_1, theta_train_1, theta_test_1, sd_train_1, sd_test_1 = generate_split_data(keep_list=subset)

print(f'Shape of first set {f_train_0.shape} and Shape of the second set {f_train_1.shape}')

def return_usv(train_data, epsilon):
    SS = StandardScaler(copy=True)
    fs = SS.fit_transform(train_data)
    epsilon = 1 - epsilon
    u, s, vh = np.linalg.svd(fs, full_matrices=True)
    importance = np.square(s/math.sqrt(u.shape[0] - 1))
    cum_importance = np.cumsum(importance)/np.sum(importance)
    pc_no = [c_id for c_id, c in enumerate(cum_importance) if c > epsilon][0]
    S = s[0:pc_no]
    U = u[:, 0:pc_no]
    V = vh[0:pc_no, :]
    return U, S, V, pc_no

u0, s0, v0, pc0 = return_usv(f_train_0,0.05)
u1, s1, v1, pc1 = return_usv(f_train_1,0.05)
print(f'PC number for first set {pc0} second set {pc1}')

# Combine these u,s,v matrices to make the U,S,V
U = np.append(u0, u1, axis=1)
S = np.diag(np.append(s0,s1))
V = block_diag(v0,v1)
print(f'U {U.shape} S {S.shape} V {V.shape}')


standardpcinfo = {'U': U,
                  'S': S,
                  'V': V}
####################################################
method_name = 'PCGPR_split'
is_train = False
emu_path = 'VAH_PCGPR_ind.pkl'

if (os.path.exists(emu_path)) and (is_train==False):
    print('Saved emulators exists and overide is prohibited')
    with open(emu_path, 'rb') as file:
        emu_tr = pickle.load(file)    
else:
    emu_tr = emulator(x=x_np,
                      theta=theta_train,
                      f=f_train.T,
                      method='PCGPR',
                      args={'epsilon': 0.1,
                            'prior': prior_dict,
                            'standardpcinfo': standardpcinfo})

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