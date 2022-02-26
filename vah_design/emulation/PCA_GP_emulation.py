import sys, os
import pickle
import math

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process import kernels as krnl

import uncertainty_toolbox as uct

import time
#Libraries for Parameter estimation

import scipy.stats as st
from scipy import optimize
import ptemcee
from scipy.linalg import inv, lapack
from multiprocessing import Pool
from multiprocessing import cpu_count

#Library to load simulation data
from load import simulation

# Load simulation data

simulation_files = ['mean_for_300_sliced_200_events_design','mean_for_90_sliced_test_design_800_events_design','mean_for_90_add_batch0_800_events_design','mean_for_90_add_batch1_800_events_design','mean_for_75_batch0_design_1600_events_design']
  # Get all simulation data in the files into a single array
 # 0 = 'mean_for_300_sliced_200_events_design'
 # 1 = 'mean_for_90_sliced_test_design_800_events_design'
 # 2 = 'mean_for_90_add_batch0_800_events_design'
 # 3 = 'mean_for_90_add_batch1_800_events_design'
 # 4 = 'mean_for_75_batch0_design_1600_events_design']
df_list = [simulation(file) for file in simulation_files]

######(TODO)#########
# Code bits to beautify and map all the names to the standard form.

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
# Map observables to the proper form

# Map file names to a proper form
####################

# Check the failure rates for each simulation file
N = 5
sns.set_context('paper', font_scale=0.8)
fig, ax = plt.subplots(1,N, figsize=(20,5))
for i in range(0,N):
    sns.histplot(df_list[i].events,x='nevents', binwidth=10 , ax=ax[i])
    ax[i].set_title(label=i)
plt.show()

# Filter and combine all simulation files to x_train, y_train, x_test, y_test
# Allowed error rate for events per design as a percentage
error_rate = 5

# Split of simulation data into trainig and testing
train = [0,2,3,4]
test = [1]

y_train = pd.DataFrame()
y_er_train = pd.DataFrame()
x_train = pd.DataFrame()

y_test = pd.DataFrame()
y_er_test = pd.DataFrame()
x_test = pd.DataFrame()
for i in range(0,N):

    df = df_list[i].combine(error_rate)
    if i in train:
        y_train = pd.concat([y_train,df.iloc[:,17:140]], axis=0)
        y_er_train = pd.concat([y_er_train,df.iloc[:,140:]], axis=0)
        x_train = pd.concat((x_train, df.iloc[:,2:17]), axis=0)
    elif i in test:
        y_test = pd.concat([y_test,df.iloc[:,17:140]], axis=0)
        y_er_test = pd.concat([y_er_test,df.iloc[:,140:]], axis=0)
        x_test = pd.concat((x_test, df.iloc[:,2:17]), axis=0)

print(y_train.shape)
print(y_er_train.shape)
print(x_train.shape)
print('#############')
print(y_test.shape)
print(y_er_test.shape)
print(x_test.shape)


# Load experimental data
experiment=pd.read_csv(filepath_or_buffer="../HIC_experimental_data/PbPb2760_experiment",index_col=0)
# Gather what type of experimental data do we have.
exp_label=[]
for i in experiment.columns:
    words=i.split('[')
    exp_label.append(words[0]+'_['+words[1])

# Only keep simulation data that we have corresponding experimental data
y_train = y_train[exp_label]
y_test = y_test[exp_label]

print(y_train.shape)
print(y_test.shape)


############# PCA ######################

#Scaling the data to be zero mean and unit variance for each observables
SS  =  StandardScaler(copy=True)
#Singular Value Decomposition
u, s, vh = np.linalg.svd(SS.fit_transform(y_train), full_matrices=True)
print(f'shape of u {u.shape} shape of s {s.shape} shape of vh {vh.shape}')

# How many PCs to keep?
npc = 110 # number of PCs
# print the explained raito of variance
# https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(7,4))
#importance = pca_analysis.explained_variance_
importance = np.square(s/math.sqrt(u.shape[0]-1))
cumulateive_importance = np.cumsum(importance)/np.sum(importance)

[c_id for c_id, c in enumerate(cumulateive_importance) if c > 0.99]

idx = np.arange(1,1+len(importance))
ax1.bar(idx,importance)
ax1.set_xlabel("PC index")
ax1.set_ylabel("Variance")
ax2.bar(idx,cumulateive_importance)
ax2.set_xlabel(r"The first $n$ PC")
ax2.set_ylabel("Fraction of total variance")
plt.tight_layout(True)
plt.show()


#whiten and project data to principal component axis (only keeping first 10 PCs)
pc_tf_data=u[:,0:npc] * math.sqrt(u.shape[0]-1)
print(f'Shape of PC transformed data {pc_tf_data.shape}')
#Scale Transformation from PC space to original data space
inverse_tf_matrix= np.diag(s[0:npc]) @ vh[0:npc,:] * SS.scale_.reshape(1,-1)/ math.sqrt(u.shape[0]-1)


########### Buidling Emulators #############

model_param = df_list[0].design.columns
prior_min = [10, -0.7, 0.5, 0, 0.3, 0.135, 0.13, 0.01, -2, -1, 0.01, 0.12, 0.025, -0.8, 0.3]
prior_max = [30, 0.7, 1.5, 1.7, 2, 0.165, 0.3, 0.2, 1, 2, 0.25, 0.3, 0.15, 0.8, 1]




# If false, uses pre-trained emulators.
# If true, retrain emulators.
EMU = 'vah_pcgp'
train_emulators = False
import time
input_dim=len(model_param)
ptp = np.array(prior_max) - np.array(prior_min)
bound=zip(prior_min,prior_max)
if (os.path.exists(EMU)) and (train_emulators==False):
    print('Saved emulators exists and overide is prohibited')
    with open(EMU,"rb") as f:
        Emulators=pickle.load(f)
else:
    Emulators=[]
    for i in range(0,npc):
        start_time = time.time()
        kernel=1*krnl.RBF(length_scale=ptp,length_scale_bounds=np.outer(ptp, (1e-3, 1e3)))+ krnl.WhiteKernel(noise_level=.1, noise_level_bounds=(1e-3, 1e1))
        GPR=gpr(kernel=kernel,n_restarts_optimizer=50,alpha=0.0000000001)
        GPR.fit(x_train,pc_tf_data[:,i].reshape(-1,1))
        print(GPR.kernel_)
        print(f'GPR score is {GPR.score(x_train,pc_tf_data[:,i])} \n')
        #print(f'GPR log_marginal likelihood {GPR.log_marginal_likelihood()} \n')
        print("--- %s seconds ---" % (time.time() - start_time))
        Emulators.append(GPR)

if (train_emulators==True) or not(os.path.exists(EMU)):
    with open(EMU,"wb") as f:
        pickle.dump(Emulators,f)



def predict_observables(model_parameters):
    """Predicts the observables for any model parameter value using the trained emulators.

    Parameters
    ----------
    Theta_input : Model parameter values. Should be an 1D array of 15 model parametrs.

    Return
    ----------
    Mean value and full error covaraiance matrix of the prediction is returened. """

    mean=[]
    variance=[]
    theta=np.array(model_parameters).flatten()

    if len(theta)!=15:
        raise TypeError('The input model_parameters array does not have the right dimensions')
    else:
        theta=np.array(theta).reshape(1,15)
        for i in range(0,npc):
            mn,std=Emulators[i].predict(theta,return_std=True)
            mean.append(mn)
            variance.append(std**2)
    mean=np.array(mean).reshape(1,-1)
    inverse_transformed_mean=mean @ inverse_tf_matrix + np.array(SS.mean_).reshape(1,-1)
    variance_matrix=np.diag(np.array(variance).flatten())
    A_p=inverse_tf_matrix
    inverse_transformed_variance=np.einsum('ik,kl,lj-> ij', A_p.T, variance_matrix, A_p, optimize=False)
    return inverse_transformed_mean, inverse_transformed_variance


#make predictions for validation daa from trained emulators
prediction_val = []
prediction_sig_val = []
for row in x_test.values:
    prediction,pred_cov = predict_observables(row)
    prediction_sig_val.append(np.sqrt(np.diagonal(pred_cov)))
    prediction_val.append(prediction)
prediction_val = np.array(prediction_val).reshape(-1,y_test.shape[1])
prediction_sig_val = np.array(prediction_sig_val).reshape(-1,y_test.shape[1])


########## Plotting ################

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



index={}
st_index=0
for obs_group in  obs_groups.keys():
    for obs in obs_groups[obs_group]:
        #print(obs)
        n_centrality= len(obs_cent_list['Pb-Pb-2760'][obs])
        #print(n_centrality)
        index[obs]=[st_index,st_index+n_centrality]
        st_index = st_index+n_centrality
print(index)
from sklearn import metrics
y_test.shape
prediction_val.shape
r = metrics.r2_score(prediction_val.flatten(), y_test.values.flatten())
print('R2 test(sklearn) = ',r)


sns.set_context('paper',font_scale=0.8)
for obs in index.keys():
    st = index[obs][0]
    ed = index[obs][1]
    nrw = int(np.ceil((ed-st)/4))
    fig, axs = plt.subplots(nrows=nrw,ncols= 4, figsize=(10,nrw*4),sharex=True, sharey=True)
    for iii,ax in enumerate(axs.flatten()):
        if iii>=ed-st:
            continue;
        ii=st+iii
        mse = sklearn.metrics.mean_squared_error(y_test.values[:,ii]\
                                       ,prediction_val[:,ii])
        r = sklearn.metrics.r2_score(y_test.values[:,ii]\
                                       ,prediction_val[:,ii])
        #print(r)
        uct.plot_calibration(prediction_val[:,ii], prediction_sig_val[:,ii], y_test.values[:,ii], ax=ax)
        #print()
        cen_st = obs_cent_list['Pb-Pb-2760'][obs][iii]
        #print(cen_st)
        ax.set_title(f'{cen_st}  R2: {r:.2f}')
    fig.suptitle(obs_tex_labels[obs])
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'pcgp/{obs}.png', dpi=200)

sns.set_context('paper',font_scale=0.8)
for obs in index.keys():
    st = index[obs][0]
    ed = index[obs][1]
    nrw = int(np.ceil((ed-st)/4))
    fig, axs = plt.subplots(nrows=nrw,ncols= 4, figsize=(10,nrw*4),sharex=False, sharey=False)
    for iii,ax in enumerate(axs.flatten()):
        if iii>=ed-st:
            continue;
        ii=st+iii
        mse = sklearn.metrics.mean_squared_error(y_test.values[:,ii]\
                                       ,prediction_val[:,ii])
        r = sklearn.metrics.r2_score(y_test.values[:,ii]\
                                       ,prediction_val[:,ii])
            #print(r)
        ax.errorbar(x=y_test.values[:,ii],y=prediction_val[:,ii],yerr=prediction_sig_val[:,ii],fmt='x')
        min_value = min([ax.get_xlim()[0],ax.get_ylim()[0]])
        max_value = min([ax.get_xlim()[1],ax.get_ylim()[1]])
        ax.plot([min_value,max_value],[min_value,max_value])
        #ax.text(x=0.49, y= 0.025, s=f'R2 score = {r:.3f}', fontdict={'fontsize':'20'})
        ax.set_xlabel('Simulation')
        ax.set_ylabel('Emulation')
            #print()
        cen_st = obs_cent_list['Pb-Pb-2760'][obs][iii]
            #print(cen_st)
        ax.set_title(f'{cen_st}  R2: {r:.2f}')
    fig.suptitle(obs_tex_labels[obs])
    plt.tight_layout()
        #plt.show()
    plt.savefig(f'pcgp/{obs}_emu_sim.png', dpi=200)
