#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 21:20:08 2022

@author: ozgesurer
"""
import numpy as np
from sklearn import metrics
import uncertainty_toolbox as uct
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

# 8 bins
ALICE_cent_bins = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]])

obs_cent_list = {
'Pb-Pb-2760': {
    'dNch_deta' : ALICE_cent_bins,
    'dET_deta' : np.array([[0, 2.5], [2.5, 5], [5, 7.5], [7.5, 10],
                           [10, 12.5], [12.5, 15], [15, 17.5], [17.5, 20],
                           [20, 22.5], [22.5, 25], [25, 27.5], [27.5, 30],
                           [30, 32.5], [32.5, 35], [35, 37.5], [37.5, 40],
                           [40, 45], [45, 50], [50, 55], [55, 60],
                           [60, 65], [65, 70]]), # 22 bins
    'dN_dy_pion'   : ALICE_cent_bins,
    'dN_dy_kaon'   : ALICE_cent_bins,
    'dN_dy_proton' : ALICE_cent_bins,
    'dN_dy_Lambda' : np.array([[0,5],[5,10],[10,20],[20,40],[40,60]]), # 5 bins
    'dN_dy_Omega'  : np.array([[0,10],[10,20],[20,40],[40,60]]), # 4 bins
    'dN_dy_Xi'     : np.array([[0,10],[10,20],[20,40],[40,60]]), # 4 bins
    'mean_pT_pion'   : ALICE_cent_bins,
    'mean_pT_kaon'   : ALICE_cent_bins,
    'mean_pT_proton' : ALICE_cent_bins,
    'pT_fluct' : np.array([[0,5],[5,10],[10,15],[15,20], [20,25],[25,30],[30,35],[35,40], [40,45],[45,50],[50,55],[55,60]]), #12 bins
    'v22' : ALICE_cent_bins,
    'v32' : np.array([[0,5],[5,10],[10,20],[20,30], [30,40],[40,50]]), # 6 bins
    'v42' : np.array([[0,5],[5,10],[10,20],[20,30], [30,40],[40,50]]), # 6 bins
    }
}

obs_groups = {'yields' : ['dN_dy_pion', 'dN_dy_kaon', 'dN_dy_proton', 'dNch_deta', 'dET_deta'],
              'mean_pT' : ['mean_pT_pion', 'mean_pT_kaon','mean_pT_proton', ],
              'flows' : ['v22', 'v32', 'v42'],
              'fluct' : ['pT_fluct']}

obs_group_labels = {'yields' : r'$dN_\mathrm{id}/dy_p$, $dN_\mathrm{ch}/d\eta$, $dE_T/d\eta$ [GeV]',
                    'mean_pT' : r'$ \langle p_T \rangle_\mathrm{id}$' + ' [GeV]',
                    'fluct' : r'$\delta p_{T,\mathrm{ch}} / \langle p_T \rangle_\mathrm{ch}$',
                    'flows' : r'$v^{(\mathrm{ch})}_k\{2\} $'}

colors = ['b', 'g', 'r', 'c', 'm', 'tan', 'gray']

obs_tex_labels = {'dNch_deta' : r'$dN_\mathrm{ch}/d\eta$',
                  'dN_dy_pion' : r'$dN_{\pi}/dy_p$',
                  'dN_dy_kaon' : r'$dN_{K}/dy_p$',
                  'dN_dy_proton' : r'$dN_{p}/dy_p$',
                  'dET_deta' : r'$dE_{T}/d\eta$',

                  'mean_pT_proton' : r'$\langle p_T \rangle_p$',
                  'mean_pT_kaon' : r'$\langle p_T \rangle_K$',
                  'mean_pT_pion' : r'$\langle p_T \rangle_\pi$',

                  'pT_fluct' : None,
                  'v22' : r'$v^{(\mathrm{ch})}_2\{2\}$',
                  'v32' : r'$v^{(\mathrm{ch})}_3\{2\}$',
                  'v42' : r'$v^{(\mathrm{ch})}_4\{2\}$'}

def plot_UQ(f, fhat, sigmahat, method='PCGP'):

    index={}
    st_index=0
    for obs_group in  obs_groups.keys():
        for obs in obs_groups[obs_group]:
            n_centrality = len(obs_cent_list['Pb-Pb-2760'][obs])
            index[obs]=[st_index,st_index+n_centrality]
            st_index = st_index + n_centrality
    
    
    r = metrics.r2_score(fhat.flatten(), f.flatten())
    print('R2 test(sklearn) = ',r)
    
    
    sns.set_context('paper', font_scale=0.8)
    for obs in index.keys():
        st = index[obs][0]
        ed = index[obs][1]
        nrw = int(np.ceil((ed-st)/4))
        fig, axs = plt.subplots(nrows=nrw, ncols=4, figsize=(10, nrw*4), sharex=True, sharey=True)
        for iii,ax in enumerate(axs.flatten()):
            if iii>=ed-st:
                continue;
            ii = st + iii
            mse = sklearn.metrics.mean_squared_error(f[:, ii], fhat[:, ii])
            r = sklearn.metrics.r2_score(f[:, ii], fhat[:, ii])
            uct.plot_calibration(fhat[:, ii], sigmahat[:, ii], f[:, ii], ax=ax)
            cen_st = obs_cent_list['Pb-Pb-2760'][obs][iii]
            ax.set_title(f'{cen_st}  R2: {r:.2f}')
        fig.suptitle(obs_tex_labels[obs])
        plt.tight_layout()
        plt.savefig(f'{method}/{obs}.png', dpi=200)
    
    sns.set_context('paper', font_scale=0.8)
    for obs in index.keys():
        st = index[obs][0]
        ed = index[obs][1]
        nrw = int(np.ceil((ed-st)/4))
        fig, axs = plt.subplots(nrows=nrw, ncols=4, figsize=(10, nrw*4), sharex=False, sharey=False)
        for iii,ax in enumerate(axs.flatten()):
            if iii>=ed-st:
                continue;
            ii=st+iii
            mse = sklearn.metrics.mean_squared_error(f[:, ii], fhat[:, ii])
            ax.errorbar(x=f[:,ii], y=fhat[:,ii], yerr=sigmahat[:, ii], fmt='x')
            min_value = min([ax.get_xlim()[0], ax.get_ylim()[0]])
            max_value = min([ax.get_xlim()[1], ax.get_ylim()[1]])
            ax.plot([min_value, max_value], [min_value, max_value])
            ax.set_xlabel('Simulation')
            ax.set_ylabel('Emulation')
            cen_st = obs_cent_list['Pb-Pb-2760'][obs][iii]
            ax.set_title(f'{cen_st}  R2: {r:.2f}')
        fig.suptitle(obs_tex_labels[obs])
        plt.tight_layout()
        plt.savefig(f'{method}/{obs}_emu_sim.png', dpi=200)
        plt.show()


def plot_R2(fhat, f):
    rsq = []
    for i in range(fhat.shape[0]):
        sse = np.sum((fhat[i, :] - f[i, :])**2)
        sst = np.sum((f[i, :] - np.mean(f[i, :]))**2)
        rsq.append(1 - sse/sst)

    plt.scatter(np.arange(fhat.shape[0]), rsq)
    plt.xlabel('observables')
    plt.ylabel(r'test $r^2$')
    plt.show()