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
from sklearn.preprocessing import StandardScaler
import math
from surmise.emulation import emulator
from surmise.calibration import calibrator
from plotting import plot_UQ, plot_R2, plot_hist, plot_density
from priors import prior_VAH
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.linalg import block_diag

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

####################################################
# Note: Total emu time: 390 sec.  Total cal time: 500
####################################################
seconds_st = time.time()

# Note: Here we can create different funcs to split data into training and test
f_train, f_test, theta_train, theta_test, sd_train, sd_test, y, thetanames = generate_split_data()

subset= ['v22_[0 5]',
       'v22_[ 5 10]', 'v22_[10 20]', 'v22_[20 30]', 'v22_[30 40]',
       'v22_[40 50]', 'v22_[50 60]', 'v22_[60 70]', 'v32_[0 5]', 'v32_[ 5 10]',
       'v32_[10 20]', 'v32_[20 30]', 'v32_[30 40]', 'v32_[40 50]', 'v42_[0 5]',
       'v42_[ 5 10]', 'v42_[10 20]', 'v42_[20 30]', 'v42_[30 40]',
       'v42_[40 50]']

obs_type = ['v22','v32','v42']

f_train_0, f_test_0, theta_train_0, theta_test_0, sd_train_0, sd_test_0,_,_ = generate_split_data(drop_list=subset)

f_train_1, f_test_1, theta_train_1, theta_test_1, sd_train_1, sd_test_1,_,_ = generate_split_data(keep_list=subset)

print(f'Shape of first set {f_train_0.shape} and Shape of the second set {f_train_1.shape}')

# Perform a closure test or not. If True will use pseudo-experimental data to test the accuracy of the inference. 
closure = True
# Combine all for calibration
if closure == False:
    fcal = np.concatenate((f_train, f_test), axis=0)
    thetacal = np.concatenate((theta_train, theta_test), axis=0)
    sdcal = np.concatenate((sd_train, sd_test), axis=0)
else:
    # Get one of the most accurate simulations as pseudo-experimental data. So pick one from last 50 simulation points.
    # last 75 simulation data has 1600 events per design (ignoring failure rates)
    np.random.seed(19)
    num = np.random.randint(0,50,1)
    print(f'Using {75-num} simulation as psuedo-experimental data')
    y_temp = np.vstack([f_train[-1*num,:], np.square(sd_train[-1*num,:])])
    y = pd.DataFrame(y_temp, index=['mean', 'variance'])
    closure_params = theta_train[-1*num,:]
    print(closure_params)
    f_train = np.delete(f_train, -1*num, 0)
    sd_train = np.delete(sd_train, -1*num, 0)
    theta_train = np.delete(theta_train, -1*num, 0)

    f_train_0 = np.delete(f_train_0, -1*num, 0)
    sd_train_0 = np.delete(sd_train_0, -1*num, 0)

    f_train_1 = np.delete(f_train_1, -1*num, 0)
    sd_train_1 = np.delete(sd_train_1, -1*num, 0)

    f_train_0 = np.concatenate((f_train_0, f_test_0), axis=0)
    f_train_1 = np.concatenate((f_train_1, f_test_1), axis=0)



    fcal = np.concatenate((f_train, f_test), axis=0)
    thetacal = np.concatenate((theta_train, theta_test), axis=0)
    sdcal = np.concatenate((sd_train, sd_test), axis=0)
    print(f'f_train_0 {f_train_0.shape} f_train_1 {f_train_1} fcal {fcal.shape}')


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
    fig, ax = plt.subplots()
    ax.bar(np.arange(0,pc_no),cum_importance[0:pc_no])
    ax.set_xlabel('PC')
    ax.set_ylabel('Explained Variance')
    plt.show()
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






x_np = np.arange(0, fcal.shape[1])[:, None]
x_np = x_np.astype('object')

##########################################################
# Note: Pick method_name = 'PCGPwM' or 'PCGPR' or 'PCSK'
##########################################################

method_name = 'PCGPR_with_flow'
is_train = True
if closure==False:
    emu_path = 'VAH_' + method_name + '.pkl' 
else:
    emu_path = 'VAH_' + method_name + '_closure_' + '.pkl'
    np.save(f'closure_params_{method_name}', closure_params)
        
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
                          args={'epsilon': 0.04,
                                'prior': prior_dict})
    elif method_name == 'PCGPR_with_flow':
        emu_tr = emulator(x=x_np,
                      theta=thetacal,
                      f=fcal.T,
                      method='PCGPR',
                      args={'epsilon': 0.1,
                            'prior': prior_dict,
                            'standardpcinfo': standardpcinfo})
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


if closure==False:
    cal_path = 'VAH_' + method_name + '_calibrator_PTMC' + '.pkl' 
else:
    cal_path = 'VAH_' + method_name + '_calibrator_PTMC' + '_closure_' +'.pkl'


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

