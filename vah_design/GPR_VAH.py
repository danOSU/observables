import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surmise.emulation import emulator
from surmise.calibration import calibrator
import scipy.stats as sps
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process import kernels as krnl

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
#f_np_orig = np.transpose(f_np_orig)
#f_test_orig = np.transpose(f_test_orig)

# Observe simulation outputs in comparison to real data
#fig, axis = plt.subplots(4, 2, figsize=(15, 15))
#j = 0
#k = 0
#uniquex = np.unique(x_np[:, 0])
#for u in uniquex:
#    whereu = u == x_np[:, 0]
#    for i in range(f_np_orig.shape[1]):
#        axis[j, k].plot(x_np[whereu, 1].astype(int), f_np_orig[whereu, i], zorder=1, color='grey')
#    axis[j, k].scatter(x_np[whereu, 1].astype(int), y_mean[whereu], zorder=2, color='red')
#    axis[j, k].set_ylabel(u)
#    if j == 3:
#        j = 0
#        k += 1
#    else:
#        j += 1
#plt.show()
#f_np_orig = np.transpose(f_np_orig)

#f_mean = np.mean(f_np_orig, axis=0)
#fstd = f_np_orig - f_mean

# Scaling the data to be zero mean and unit variance for each observables
SS = StandardScaler(copy=True)
# Singular Value Decomposition
U, S, V = np.linalg.svd(SS.fit_transform(f_np_orig), full_matrices=True)
print(f'shape of U {U.shape} shape of S {S.shape} shape of V {V.shape}')

# Singular Value Decomposition
#U, S, V = np.linalg.svd(fstd, full_matrices=True)

npc = 10
# print the explained raito of variance
# https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(7,4))
importance = np.square(S[:npc]/np.sqrt(U.shape[0]-1))
cumulateive_importance = np.cumsum(importance)/np.sum(importance)
idx = np.arange(1, 1 + len(importance))
ax1.bar(idx, importance)
ax1.set_xlabel("PC index")
ax1.set_ylabel("Variance")
ax2.bar(idx,cumulateive_importance)
ax2.set_xlabel(r"The first $n$ PC")
ax2.set_ylabel("Fraction of total variance")
plt.tight_layout(True)

 
prior_min = [10, -0.7, 0.5, 0, 0.3, 0.135, 0.13, 0.01, -2, -1, 0.01, 0.12, 0.025, -0.8, 0.3]
prior_max = [30, 0.7, 1.5, 1.7, 2, 0.165, 0.3, 0.2, 1, 2, 0.25, 0.3, 0.15, 0.8, 1]
prior_df = pd.DataFrame(data=np.vstack((prior_min, prior_max)), index = ['min','max'])

design_max = prior_df.loc['max'].values
design_min = prior_df.loc['min'].values

ptp = design_max - design_min
bound=zip(design_min, design_max)

#npc = sum(ids)
Emulators = []

pc_tf_data = U[:, 0:npc] * np.sqrt(U.shape[0] - 1)
print(f'Shape of PC transformed data {pc_tf_data.shape}')
inverse_tf_matrix = np.diag(S[0:npc]) @ V[0:npc,:] * SS.scale_.reshape(1,-1) / np.sqrt(U.shape[0]-1)



for i in range(0, npc):
    kernel = 1*krnl.RBF(length_scale=ptp, 
                        length_scale_bounds=np.outer(ptp, (1e-3, 1e3))) + krnl.WhiteKernel(noise_level=.1, noise_level_bounds=(1e-3, 1e3))
    GPR = gpr(kernel=kernel, n_restarts_optimizer=100, alpha=0.0000000001)
    GPR.fit(theta, pc_tf_data[:, i].reshape(-1, 1))
    print(GPR.kernel_)
    print(f'GPR score is {GPR.score(theta, pc_tf_data[:,i])} \n')
    Emulators.append(GPR)
 
def predict_GPR(param):
    mean = []
    variance = []
    theta = np.array(param).flatten()
    theta = np.array(theta).reshape(1,15)
    for i in range(0,npc):
        mn, std = Emulators[i].predict(theta,return_std=True)
        mean.append(mn)
        variance.append(std**2)
    mean = np.array(mean).reshape(1,-1)
    inverse_transformed_mean = mean @ inverse_tf_matrix + np.array(SS.mean_).reshape(1,-1)
    variance_matrix = np.diag(np.array(variance).flatten())
    A_p = inverse_tf_matrix
    inverse_transformed_variance = np.einsum('ik,kl,lj-> ij', A_p.T, variance_matrix, A_p, optimize=False)
    return inverse_transformed_mean, inverse_transformed_variance
    

prediction_val = []
prediction_sig_val = []
for row in theta_validation.values:
    prediction, pred_cov = predict_GPR(row)
    prediction_sig_val.append(np.sqrt(np.diagonal(pred_cov)))
    prediction_val.append(prediction)
prediction_val = np.array(prediction_val).reshape(-1,len(selected_observables))
prediction_sig_val = np.array(prediction_sig_val).reshape(-1,len(selected_observables))


# Check error
pred_test_var = np.transpose(prediction_sig_val)
pred_test_mean = np.transpose(prediction_val)
f_test = np.transpose(f_test_orig)
errors_test = (pred_test_mean - f_test).flatten()
sst = np.sum((f_test.flatten() - np.mean(f_test.flatten()))**2)
print('MSE test=', np.mean(errors_test**2))
print('rsq test=', 1 - np.sum(errors_test**2)/sst)

rsq = []
for i in range(pred_test_mean.shape[0]):
    sse = np.sum((pred_test_mean[i, :] - f_test[i, :])**2)
    sst = np.sum((f_test[i, :] - np.mean(f_test[i, :]))**2)
    rsq.append(1 - sse/sst)

plt.scatter(np.arange(pred_test_mean.shape[0]), rsq)
plt.xlabel('observables')
plt.ylabel('test rsq')
plt.show()

# Observe test prediction
fig = plt.figure()
plt.scatter(f_test, pred_test_mean, alpha=0.5)
plt.plot(range(0, 3000), range(0, 3000), color='red')
plt.xlabel('Simulator outcome (test)')
plt.ylabel('Emulator prediction (test)')
plt.show()

uniquex = np.unique(x_np[:, 0])
fig, axis = plt.subplots(4, 2, figsize=(15, 15))
i, j = 0, 0
for o in uniquex:
    idx = o == x_np[:, 0]
    axis[i, j].scatter(f_test[idx, :], pred_test_mean[idx, :], alpha=0.5)
    if np.max(pred_test_mean[idx, :]) > np.max(f_test[idx, :]):
        xlu = np.ceil(np.max(pred_test_mean[idx, :]))
    else:
        xlu = np.ceil(np.max(f_test[idx, :]))
    if np.min(pred_test_mean[idx, :]) > np.min(f_test[idx, :]):
        xll = np.floor(np.min(f_test[idx, :]))
    else:
        xll = np.floor(np.min(pred_test_mean[idx, :]))
    axis[i, j].plot(range(int(xll), int(xlu)+1), range(int(xll), int(xlu)+1), color='red')

    sse = np.sum((pred_test_mean[idx, :] - f_test[idx, :])**2)
    sst = np.sum((f_test[idx, :] - np.mean(f_test[idx, :]))**2)
    #rsq.append(1 - sse/sst)
    axis[i, j].set_title('r2:'+ str(np.round(1 - sse/sst, 2)))
    print(1 - sse/sst)
    i += 1
    if i > 3:
        i = 0
        j = 1
        
# Check error distribution
mu = 0
variance = 1
sigma = np.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

fig, axis = plt.subplots(4, 2, figsize=(15, 15))
i, j = 0, 0
for o in uniquex:
    idx = o == x_np[:, 0]
    e = ((pred_test_mean[idx, :] - f_test[idx, :])/np.sqrt(pred_test_var[idx, :])).flatten()
    axis[i, j].hist(e, bins=25, density=True)
    axis[i, j].plot(x, stats.norm.pdf(x, mu, sigma), color='red')
    axis[i, j].set_title(o)
    i += 1
    if i > 3:
        i = 0
        j = 1

# Check relative error
fig, axis = plt.subplots(4, 2, figsize=(15, 15))
i, j = 0, 0
for o in uniquex:
    idx = o == x_np[:, 0]
    e = ((pred_test_mean[idx, :] - f_test[idx, :])/f_test[idx, :]).flatten()
    axis[i, j].hist(e, bins=25, density=True)
    axis[i, j].set_title(o)
    i += 1
    if i > 3:
        i = 0
        j = 1