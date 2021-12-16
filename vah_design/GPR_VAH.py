import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surmise.emulation import emulator
from surmise.calibration import calibrator
import scipy.stats as sps
from scipy import stats

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

df_mean = pd.read_csv('mean_for_200_sliced_200_events_design', index_col=0)
df_sd = pd.read_csv('sd_for_200_sliced_200_events_design', index_col=0)

df_mean_test = pd.read_csv("mean_for_50_sliced_200_events_test_design", index_col=0)
df_sd_test = pd.read_csv("sd_for_50_sliced_200_events_test_design", index_col=0)

df_mean.shape
df_sd.shape

design = pd.read_csv('sliced_VAH_090321.txt', delimiter = ' ')
design.head()
design.shape

design_validation = pd.read_csv('sliced_VAH_090321_test.txt', delimiter = ' ')

colnames = design.columns

#drop tau_initial parameter for now because we keep it fixed
design = design.drop(labels='tau_initial', axis=1)
design.shape

design_validation = design_validation.drop(labels='tau_initial', axis=1)
colnames = colnames[0:-1]

# Read the experimental data
exp_data = pd.read_csv('PbPb2760_experiment', index_col=0)
y_mean = exp_data.to_numpy()[0, ]
y_sd = exp_data.to_numpy()[1, ]

# Get the initial 200 parameter values
theta = design.head(200)
theta.head()

theta_validation = design_validation.iloc[0:50]
theta_validation.shape

plt.scatter(theta.values[:,0], df_mean.values[:,0])
plt.show()

fig, axis = plt.subplots(3, 5, figsize=(10, 10))
theta.hist(ax=axis)
plt.show()

colname_exp = exp_data.columns
#colname_sim = df_mean.columns
#colname_theta = theta.columns

# Gather what type of experimental data do we have. 
exp_label = []
x = [] 
j = 0
x_id = []
for i in exp_data.columns:
    words = i.split('[')
    exp_label.append(words[0]+'_['+words[1])
    if words[0] in x:
        j += 1
    else:
        j = 0
    x_id.append(j)
    x.append(words[0])


# Only keep simulation data that we have corresponding experimental data
df_mean = df_mean[exp_label]
df_sd = df_sd[exp_label]

df_mean_test = df_mean_test[exp_label]
df_sd_test = df_sd_test[exp_label]

df_mean.head()

selected_observables = exp_label[0:-32]

x_np = np.column_stack((x[0:-32], x_id[0:-32])) 
x_np = x_np.astype('object')
#x_np[:, 1] = x_np[:, 1].astype(int)
y_mean = y_mean[0:-32]
y_sd = y_sd[0:-32]

print(f'Last item on the selected observable is {selected_observables[-1]}')

df_mean = df_mean[selected_observables]
df_sd = df_sd[selected_observables]

df_mean_test = df_mean_test[selected_observables]
df_sd_test = df_sd_test[selected_observables]

print(f'Shape of the constrained simulation output {df_mean.shape}')

# Remove bad designs

drop_index = np.array([19, 23, 31, 32, 71, 91, 92, 98, 129, 131, 146, 162, 171, 174, 184, 190, 194, 195, 198])
drop_index_vl = np.array([29, 35, ])
theta = theta.drop(index=drop_index)
theta.head()

theta_validation = theta_validation.drop(index=drop_index_vl)
theta_validation.head()

df_mean = df_mean.drop(index=drop_index)
df_sd = df_sd.drop(index=drop_index)

df_mean_test = df_mean_test.drop(index=drop_index_vl)
df_sd_test = df_sd_test.drop(index=drop_index_vl)

df_mean.shape
theta.shape
theta.head()


# Remove nas
theta_np = theta.to_numpy()
f_np_orig = df_mean.to_numpy()

theta_test = theta_validation.to_numpy()
f_test_orig = df_mean_test.to_numpy()
#theta_np = theta_np[-which_nas, :]
#f_np = f_np[-which_nas, :]
f_np_orig = np.transpose(f_np_orig)
f_test_orig = np.transpose(f_test_orig)

# Observe simulation outputs in comparison to real data
fig, axis = plt.subplots(4, 2, figsize=(15, 15))
j = 0
k = 0
uniquex = np.unique(x_np[:, 0])
for u in uniquex:
    whereu = u == x_np[:, 0]
    for i in range(f_np_orig.shape[1]):
        axis[j, k].plot(x_np[whereu, 1].astype(int), f_np_orig[whereu, i], zorder=1, color='grey')
    axis[j, k].scatter(x_np[whereu, 1].astype(int), y_mean[whereu], zorder=2, color='red')
    axis[j, k].set_ylabel(u)
    if j == 3:
        j = 0
        k += 1
    else:
        j += 1
plt.show()
f_np_orig = np.transpose(f_np_orig)

f_mean = np.mean(f_np_orig, axis=0)
fstd = f_np_orig - f_mean

# Singular Value Decomposition
U, S, V = np.linalg.svd(fstd, full_matrices=True)
plt.plot(np.cumsum(S)/np.sum(S))
plt.show()

print(U.shape)
print(S.shape)
print(V.shape)

eigenpc = np.linalg.eig(np.cov(np.transpose(fstd)))

from sklearn.decomposition import PCA
pca = PCA(n_components=78)
pca.fit(fstd)

print('SVD')
print(V[0:5, 0:5])
print('PCA')
print(pca.components_[0:5, 0:5])
print('eigenv')
print(eigenpc[1][0:5, 0:5])

ids = np.cumsum(S)/np.sum(S) <= 0.9


#### Block 9 #### Please refer to this number in your questions
#whiten and project data to principal component axis (only keeping first 10 PCs)

#V = eigenpc[1]

#aaa = V @ np.transpose(V)

#Vtr = V[:, ids]

#ftr = fstd @ Vtr
#ftr = fstd @ V
#ftr_1 = ftr[:, ids]

#fstd_check = (ftr @ np.transpose(V))
#fstdchck = fstdchck[:, ids]
 
prior_min = [10, -0.7, 0.5, 0, 0.3, 0.135, 0.13, 0.01, -2, -1, 0.01, 0.12, 0.025, -0.8, 0.3]
prior_max = [30, 0.7, 1.5, 1.7, 2, 0.165, 0.3, 0.2, 1, 2, 0.25, 0.3, 0.15, 0.8, 1]
prior_df = pd.DataFrame(data=np.vstack((prior_min,prior_max)), index = ['min','max'])

design_max = prior_df.loc['max'].values
design_min = prior_df.loc['min'].values

ptp = design_max - design_min
bound=zip(design_min, design_max)

npc = sum(ids)
Emulators = []

pc_tf_data = U[:, 0:npc] * np.sqrt(U.shape[0] - 1)
inverse_tf_matrix = np.diag(S[0:npc]) @ V[0:npc,:]/ np.sqrt(U.shape[0]-1)
#ch = pc_tf_data @ inverse_tf_matrix

from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process import kernels as krnl

for i in range(0, npc):
    kernel = 1*krnl.RBF(length_scale=ptp, 
                        length_scale_bounds=np.outer(ptp, (1e-3, 1e3))) + krnl.WhiteKernel(noise_level=.1, noise_level_bounds=(1e-3, 1e3))
    GPR = gpr(kernel=kernel, n_restarts_optimizer=100, alpha=0.0000000001)
    GPR.fit(theta, pc_tf_data[:, i].reshape(-1, 1))
    print(GPR.kernel_)
    print(f'GPR score is {GPR.score(theta, pc_tf_data[:,i])} \n')
    Emulators.append(GPR)
    
#theta=np.array(theta).reshape(1,15)
mean = []
variance = []
for i in range(0, npc):
    mn, std = Emulators[i].predict(theta, return_std=True)
    mean.append(mn)
    variance.append(std**2)
mean = np.hstack(mean)

mean_tr = mean @ inverse_tf_matrix + f_mean
    
#mean = np.array(mean) #.reshape(1,-1)
#inverse_transformed_mean=mean @ inverse_tf_matrix + np.array(SS.mean_).reshape(1,-1)
#variance_matrix=np.diag(np.array(variance).flatten())
#A_p=inverse_tf_matrix
#inverse_transformed_variance=np.einsum('ik,kl,lj-> ij', A_p.T, variance_matrix, A_p, optimize=False)
#return inverse_transformed_mean, inverse_transformed_variance