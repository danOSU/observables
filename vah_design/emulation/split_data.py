#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 23:30:06 2022

@author: ozgesurer
"""
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from load import simulation

def generate_split_data(**kwargs):
# Load simulation data
    simulation_files = ['mean_for_300_sliced_200_events_design',
                        'mean_for_90_sliced_test_design_800_events_design',
                        'mean_for_90_add_batch0_800_events_design',
                        'mean_for_90_add_batch1_800_events_design',
                        'mean_for_75_batch0_design_1600_events_design']
    
    df_list = [simulation(file) for file in simulation_files]


    
    # Map the design names to proper form
    model_param_dsgn = ['$N$[$2.76$TeV]',
                        '$p$',
                        '$w$ [fm]',
                        '$d_{\\mathrm{min}}$ [fm]',
                        '$\\sigma_k$',
                        '$T_{\\mathrm{sw}}$ [GeV]',
                        '$T_{\\eta,\\mathrm{kink}}$ [GeV]',
                        '$(\\eta/s)_{\\mathrm{kink}}$',
                        '$a_{\\eta,\\mathrm{low}}$ [GeV${}^{-1}$]',
                        '$a_{\\eta,\\mathrm{high}}$ [GeV${}^{-1}$]',
                        '$(\\zeta/s)_{\\max}$',
                        '$T_{\\zeta,c}$ [GeV]',
                        '$w_{\\zeta}$ [GeV]',
                        '$\\lambda_{\\zeta}$',
                        '$R$']
    
    # Check the failure rates for each simulation file
    N = 5
    sns.set_context('paper', font_scale=0.8)
    fig, ax = plt.subplots(1,N, figsize=(20,5))
    for i in range(0, N):
        sns.histplot(df_list[i].events, x='nevents', binwidth=10 , ax=ax[i])
        ax[i].set_title(label=i)
    #plt.show()
    #plt.show()
    
    # Allowed error rate for events per design as a percentage
    error_rate = 5
    
    # Split of simulation data into trainig and testing
    train = [0, 2, 3, 4]
    test = [1]
    
    f_train, f_er_train, theta_train = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    f_test, f_er_test, theta_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    for i in range(0, N):
        df = df_list[i].combine(error_rate)
        if i in train:
            f_train = pd.concat([f_train, df.iloc[:,17:140]], axis=0)
            f_er_train = pd.concat([f_er_train, df.iloc[:,140:]], axis=0)
            theta_train = pd.concat((theta_train, df.iloc[:,2:17]), axis=0)
        elif i in test:
            f_test = pd.concat([f_test, df.iloc[:,17:140]], axis=0)
            f_er_test = pd.concat([f_er_test, df.iloc[:,140:]], axis=0)
            theta_test = pd.concat((theta_test, df.iloc[:,2:17]), axis=0)
    



    # Print the shape of the data arrays
    [print(dat.shape) for dat in [f_train, f_er_train, theta_train, f_test, f_er_test, theta_test]]
    
    # Load experimental data
    experiment = pd.read_csv(filepath_or_buffer="../HIC_experimental_data/PbPb2760_experiment", index_col=0)
    
    # Gather what type of experimental data do we have.
    exp_label = []
    for i in experiment.columns:
        words = i.split('[')
        exp_label.append(words[0] + '_['+words[1])
    
    # Only keep simulation data that we have corresponding experimental data
    sd_exp_label = ['sd_' + e for e in exp_label]
    f_train = f_train[exp_label]
    f_test = f_test[exp_label]
    f_er_train = f_er_train[sd_exp_label]
    f_er_test = f_er_test[sd_exp_label]
    
    ### Drop and keep observables depending on kwargs arguments
    if 'drop_list' in kwargs.keys():
        drop_list = kwargs['drop_list']
        drop_sd = ['sd_'+obs for obs in drop_list]
        f_train, f_er_train = f_train.drop(labels=drop_list, axis=1), f_er_train.drop(labels=drop_sd, axis=1)
        f_test, f_er_test = f_test.drop(labels=drop_list, axis=1), f_er_test.drop(labels=drop_sd, axis=1)
    elif 'keep_list' in kwargs.keys():
        keep_list = kwargs['keep_list']
        keep_sd = ['sd_'+obs for obs in keep_list]
        f_train, f_er_train = f_train[keep_list], f_er_train[keep_sd]
        f_test, f_er_test = f_test[keep_list], f_er_test[keep_sd]

    return f_train.values, f_test.values, theta_train.values, theta_test.values, f_er_train.values, f_er_test.values