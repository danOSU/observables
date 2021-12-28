import numpy as np
import pandas as pd
from smt.sampling_methods import LHS
import seaborn as sns
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surmise.emulation import emulator
from surmise.calibration import calibrator
import scipy.stats as sps
from scipy import stats

# Additional design points are generated based on Eq. 5 of
# ``Deterministic Sampling of Expensive Posteriors Using Minimum Energy Designs``

df_mean = pd.read_csv('mean_for_300_sliced_200_events_design', index_col=0)
df_sd = pd.read_csv('sd_for_300_sliced_200_events_design', index_col=0)

df_mean_add = pd.read_csv('mean_for_90_add_batch0_800_events_design', index_col=0)
df_sd_add = pd.read_csv('sd_for_90_add_batch0_800__events_design', index_col=0)

df_mean_test = pd.read_csv("mean_for_50_sliced_200_events_test_design", index_col=0)
df_sd_test = pd.read_csv("sd_for_50_sliced_200_events_test_design", index_col=0)

print(df_mean.shape)
print(df_sd.shape)

design = pd.read_csv('sliced_VAH_090321.txt', delimiter = ' ')
design.head()
print(design.shape)

design_add = pd.read_csv('add_design_122421.txt', delimiter = ' ')
design_add.head()

design_validation = pd.read_csv('sliced_VAH_090321_test.txt', delimiter = ' ')
design_validation.shape
colnames = design.columns

#drop tau_initial parameter for now because we keep it fixed
design = design.drop(labels='tau_initial', axis=1)
print(design.shape)

design_validation = design_validation.drop(labels='tau_initial', axis=1)
colnames = colnames[0:-1]

# Read the experimental data
exp_data = pd.read_csv('PbPb2760_experiment', index_col=0)
y_mean = exp_data.to_numpy()[0, ]
y_sd = exp_data.to_numpy()[1, ]

# Get the initial 200 parameter values
theta = design.head(300)
theta.head()

theta_validation = design_validation.iloc[0:50]
theta_validation.shape

# plt.scatter(theta.values[:,0], df_mean.values[:,0])
# plt.show()

# fig, axis = plt.subplots(3, 5, figsize=(10, 10))
# theta.hist(ax=axis)
# plt.show()

colname_exp = exp_data.columns

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

df_mean_add = df_mean_add[exp_label]
df_sd_add = df_sd_add[exp_label]

df_mean_test = df_mean_test[exp_label]
df_sd_test = df_sd_test[exp_label]

# df_mean.head()

selected_observables = exp_label[0:-32]

x_np = np.column_stack((x[0:-32], x_id[0:-32]))
x_np = x_np.astype('object')
y_mean = y_mean[0:-32]
y_sd = y_sd[0:-32]

print(f'Last item on the selected observable is {selected_observables[-1]}')

df_mean = df_mean[selected_observables]
df_sd = df_sd[selected_observables]

df_mean_add = df_mean_add[selected_observables]
df_sd_add = df_sd_add[selected_observables]

df_mean_test = df_mean_test[selected_observables]
df_sd_test = df_sd_test[selected_observables]

print(f'Shape of the constrained simulation output {df_mean.shape}')

# Remove bad designs

drop_index = np.array([19, 23, 31, 32, 71, 91, 92, 98, 129, 131, 146, 162, 171, 174, 184, 190, 194, 195, 198, 245, 248, 266, 283, 286, 291, 299])
drop_index_vl = np.array([29, 35, ])
theta = theta.drop(index=drop_index)
theta.head()

theta_validation = theta_validation.drop(index=drop_index_vl)
theta_validation.head()

df_mean = df_mean.drop(index=drop_index)
df_sd = df_sd.drop(index=drop_index)

df_mean_test = df_mean_test.drop(index=drop_index_vl)
df_sd_test = df_sd_test.drop(index=drop_index_vl)

# Additional
drop_index_add = np.array([10, 17, 27, 35, 49, 58])
theta_add = design_add.drop(index=drop_index_add)
df_mean_add = df_mean_add.drop(index=drop_index_add)
df_sd_add = df_sd_add.drop(index=drop_index_add)

frames = [theta, theta_add]
theta = pd.concat(frames)

frames = [df_mean, df_mean_add]
df_mean = pd.concat(frames)

# Remove nas
theta_np = theta.to_numpy()
f_np = df_mean.to_numpy()

theta_test = theta_validation.to_numpy()
f_test = df_mean_test.to_numpy()
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

plt.show()

# f_test = np.log10(f_test + 1)
# f_np = np.log10(f_np + 1)

# Build an emulator
emu_tr = emulator(x=x_np,
                   theta=theta_np,
                   f=f_np,
                   method='PCGPwM',
                   args={'epsilon': 0.01})

pred_test = emu_tr.predict(x=x_np, theta=theta_test)
pred_test_mean = pred_test.mean()
pred_test_var = pred_test.var()


# Check error
errors_test = (pred_test_mean - f_test).flatten()
sst = np.sum((f_test.flatten() - np.mean(f_test.flatten()))**2)
print('MSE test=', np.mean(errors_test**2))
print('rsq test=', 1 - np.sum(errors_test**2)/sst)

rsq = []
for i in range(pred_test_mean.shape[0]):
    sse = np.sum((pred_test_mean[i, :] - f_test[i, :])**2)
    sst = np.sum((f_test[i, :] - np.mean(f_test[i, :]))**2)
    rsq.append(1 - sse/sst)
    print(selected_observables[i])

plt.scatter(np.arange(pred_test_mean.shape[0]), rsq)
plt.xlabel('observables')
plt.ylabel('test rsq')
plt.show()

# Observe test prediction
fig = plt.figure()
plt.scatter(f_test, pred_test_mean, alpha=0.5)
plt.plot(range(0, 5), range(0, 5), color='red')
plt.xlabel('Simulator outcome (test)')
plt.ylabel('Emulator prediction (test)')
plt.show()

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
plt.show()

fig, axis = plt.subplots(4, 2, figsize=(15, 15))
i, j = 0, 0
for o in uniquex:
    idx = o == x_np[:, 0]
    e = (pred_test_mean[idx, :] - f_test[idx, :]).flatten()
    axis[i, j].hist(e, bins=25)
    i += 1
    if i > 3:
        i = 0
        j = 1
plt.show()
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
plt.show()
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
plt.show()
# Check training
pred_tr = emu_tr.predict(x=x_np, theta=theta_np)
pred_tr_mean = pred_tr.mean()

# Check error
errors_tr = (pred_tr_mean - f_np).flatten()
sst_tr = np.sum((f_np.flatten() - np.mean(f_np.flatten()))**2)
print('MSE train=', np.mean(errors_tr**2))
print('rsq train=', 1 - np.sum(errors_tr**2)/sst_tr)

# Observe test prediction
fig = plt.figure()
plt.scatter(f_np, pred_tr_mean, alpha=0.5)
plt.plot(range(0, 3000), range(0, 3000), color='red')
plt.xlabel('Simulator outcome (train)')
plt.ylabel('Emulator prediction (train)')
plt.show()

# Check error distribution
mu = 0
variance = 1
sigma = np.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

pred_test_var = pred_test.var()
fig, axs = plt.subplots(1, 2, tight_layout=True)
axs[0].hist((pred_test_mean-f_test).flatten())
# This should look like a standard normal
axs[1].hist(((pred_test_mean-f_test)/np.sqrt(pred_test_var)).flatten(), density=True)
axs[1].plot(x, stats.norm.pdf(x, mu, sigma), color='red')
axs[1].set_title(r'${(\hat{\mu}_{test} - \mu_{test})}/{\hat{\sigma}_{test}}$')
plt.show()

# code to create a new design
# define limits
xlimits = np.array([[20, 30],
                    [-1, 0.2],
                    [0.5, 1.5],
                    [0, 1.7],
                    [0.3, 2],
                    [0.12, 0.165],
                    [0.13, 0.3],
                    [0.01, 0.2],
                    [-2, 1],
                    [-1, 2],
                    [0.01, 0.25],
                    [0.12, 0.3],
                    [0.025, 0.15],
                    [-0.8, 0.8],
                    [0.3, 1]])

# obtain sampling object
sampling = LHS(xlimits=xlimits)
num = 2000
x = sampling(num)
print(x.shape)

# convert data into data frame
df = pd.DataFrame(x, columns = ['Pb_Pb',
                                'Mean',
                                'Width',
                                'Dist',
                                'Flactutation',
                                'Temp',
                                'Kink',
                                'eta_s',
                                'Slope_low',
                                'Slope_high',
                                'Max',
                                'Temp_peak',
                                'Width_peak',
                                'Asym_peak',
                                'R'])

obsvar = y_sd

def loglik(obsvar, emu, theta, y, x):
    # Obtain emulator results
    emupredict = emu.predict(x, theta)
    emumean = emupredict.mean()
    try:
        emucov = emupredict.covx()
        is_cov = True
    except Exception:
        emucov = emupredict.var()
        is_cov = False

    p = emumean.shape[1]
    n = emumean.shape[0]
    y = y.reshape((n, 1))

    loglikelihood = np.zeros((p, 1))

    for k in range(0, p):
        m0 = emumean[:, k].reshape((n, 1))

        # Compute the covariance matrix
        if is_cov is True:
            s0 = emucov[:, k, :].reshape((n, n))
            CovMat = s0 + np.diag(np.squeeze(obsvar))
        else:
            s0 = emucov[:, k].reshape((n, 1))
            CovMat = np.diag(np.squeeze(s0)) + np.diag(np.squeeze(obsvar))

        # Get the decomposition of covariance matrix
        CovMatEigS, CovMatEigW = np.linalg.eigh(CovMat)

        # Calculate residuals
        resid = m0 - y

        CovMatEigInv = CovMatEigW @ np.diag(1/CovMatEigS) @ CovMatEigW.T
        loglikelihood[k] = float(-0.5 * resid.T @ CovMatEigInv @ resid -
                                 0.5 * np.sum(np.log(CovMatEigS)))

    return loglikelihood


import random


theta = np.array(df)
loglikelihood = loglik(obsvar, emu_tr, theta, y_mean, x_np)
#loglikelihood_tr = (loglikelihood - np.min(loglikelihood))/(np.max(loglikelihood) - np.min(loglikelihood))
maxid = np.argmax(loglikelihood)

theta_sc = xlimits[:,1] - xlimits[:,0]

continuing = True
theta_curr = theta[maxid]

iterator = 0
in_id = []
in_id.append(maxid)

out_id = list(np.arange(0, num))
out_id.remove(maxid)

n_select = 89
p = theta.shape[1]

def inner(loglikelihood, theta, theta_cand_id, in_id):

    best_metric = np.inf
    for i in in_id:
        dist = np.sqrt(np.sum(((theta[theta_cand_id, :] - theta[i, :]) / theta_sc)**2))
        ll_cand = 1/(2*p)*(loglikelihood[theta_cand_id])
        ll_i = 1/(2*p)*(loglikelihood[i])
        inner_metric = ll_cand+ll_i+np.log(dist)
        if inner_metric < best_metric:
            best_metric = inner_metric


    return best_metric



while continuing:

    iterator += 1
    best_obj = -np.inf
    best_id = -5
    for o_id in out_id:
        cand_value = inner(loglikelihood, theta, o_id, in_id)

        if cand_value > best_obj:
            best_obj = cand_value
            best_id = o_id

    in_id.append(best_id)
    out_id.remove(best_id)

    if iterator >= n_select:
        continuing = False




plt.hist(loglikelihood[in_id])
plt.show()

plt.hist(loglikelihood[out_id])
plt.show()

theta_in = pd.DataFrame(theta[in_id, :])
theta_out = pd.DataFrame(theta[out_id, :])
theta_in['data'] = 'in'
#theta_out['data'] = 'out'
frames = [theta_in]
frames = pd.concat(frames)
sns.pairplot(frames, hue='data', diag_kind="hist")
plt.show()

theta_in = pd.DataFrame(np.round(theta[in_id, :], 4), columns = ['Pb_Pb',
                                                                 'Mean',
                                                                 'Width',
                                                                 'Dist',
                                                                 'Flactutation',
                                                                 'Temp',
                                                                 'Kink',
                                                                 'eta_s',
                                                                 'Slope_low',
                                                                 'Slope_high',
                                                                 'Max',
                                                                 'Temp_peak',
                                                                 'Width_peak',
                                                                 'Asym_peak',
                                                                 'R'])
theta_in.to_csv(r'add_design_122721.txt', header=True, index=None, sep=' ', mode='a')
