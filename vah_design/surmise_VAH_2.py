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
f_np = df_mean.to_numpy()

theta_test = theta_validation.to_numpy()
f_test = df_mean_test.to_numpy()
#theta_np = theta_np[-which_nas, :]
#f_np = f_np[-which_nas, :]
f_np = np.transpose(f_np)
f_test = np.transpose(f_test)

# Observe simulation outputs in comparison to real data
fig, axis = plt.subplots(4, 2, figsize=(15, 15))
j = 0
k = 0
uniquex = np.unique(x_np[:, 0])
for u in uniquex:
    whereu = u == x_np[:, 0]
    for i in range(f_np.shape[1]):
        axis[j, k].plot(x_np[whereu, 1].astype(int), f_np[whereu, i], zorder=1, color='grey')
    axis[j, k].scatter(x_np[whereu, 1].astype(int), y_mean[whereu], zorder=2, color='red')
    axis[j, k].set_ylabel(u)
    if j == 3:
        j = 0
        k += 1
    else:
        j += 1

fig, axis = plt.subplots(4, 2, figsize=(15, 15))
fig2, axis2 = plt.subplots(4, 2, figsize=(15, 15))
fig3, axis3 = plt.subplots(4, 2, figsize=(15, 15))
fig4, axis4 = plt.subplots(4, 2, figsize=(15, 15))
i, j = 0, 0

# Check error distribution
mu = 0
variance = 1
sigma = np.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)


f_np = np.log10(f_np + 1)
f_test = np.log10(f_test + 1)
# Build an emulator 
for o in uniquex:
    idx = o == x_np[:, 0]
    
    emu_tr = emulator(x=x_np[idx, :], 
                       theta=theta_np, 
                       f=f_np[idx, :], 
                       method='PCGPwM',
                       args={'epsilon': 0.01}) 

    pred_test = emu_tr.predict(x=x_np[idx, :], theta=theta_test)
    pred_test_mean = pred_test.mean()
    pred_test_var = pred_test.var()

    # Check error
    errors_test = (pred_test_mean - f_test[idx, :]).flatten()
    sst = np.sum((f_test[idx, :].flatten() - np.mean(f_test[idx, :].flatten()))**2)
    #print('MSE test=', np.mean(errors_test**2))
    #print('rsq test=', 1 - np.sum(errors_test**2)/sst)
    
    # Check error
    ft = f_test[idx, :]
    for k in range(pred_test_mean.shape[0]):
        errors_test = (pred_test_mean[k, :] - ft[k, :]).flatten()
        sst = np.sum((ft[k, :].flatten() - np.mean(ft[k, :].flatten()))**2)
        print('MSE test=', np.mean(errors_test**2))
        print('rsq test=', 1 - np.sum(errors_test**2)/sst)    

    axis[i, j].scatter(f_test[idx, :], pred_test_mean, alpha=0.5)
    if np.max(pred_test_mean) > np.max(f_test[idx, :]):
        xlu = np.ceil(np.max(pred_test_mean))
    else:
        xlu = np.ceil(np.max(f_test[idx, :]))
    if np.min(pred_test_mean) > np.min(f_test[idx, :]):
        xll = np.floor(np.min(f_test[idx, :]))
    else:
        xll = np.floor(np.min(pred_test_mean))
    axis[i, j].plot(range(int(xll), int(xlu)+1), range(int(xll), int(xlu)+1), color='red')

    e = (pred_test_mean - f_test[idx, :]).flatten()
    axis2[i, j].hist(e, bins=25)

    e = ((pred_test_mean - f_test[idx, :])/np.sqrt(pred_test_var)).flatten()
    axis3[i, j].hist(e, bins=25, density=True)
    axis3[i, j].plot(x, stats.norm.pdf(x, mu, sigma), color='red')
    axis3[i, j].set_title(o)

    e = ((pred_test_mean - f_test[idx, :])/f_test[idx, :]).flatten()
    axis4[i, j].hist(e, bins=25, density=True)
    axis4[i, j].set_title(o)
    
    i += 1
    if i > 3:
        i = 0
        j = 1


