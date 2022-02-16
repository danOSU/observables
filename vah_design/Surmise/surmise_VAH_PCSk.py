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

# Initial data set of 300 points with 200 events
df_mean = pd.read_csv('../simulation_data/mean_for_300_sliced_200_events_design', index_col=0)
df_sd = pd.read_csv('../simulation_data/sd_for_300_sliced_200_events_design', index_col=0)
design = pd.read_csv('../design_data/sliced_VAH_090321.txt', delimiter = ' ')
design = design.drop(labels='tau_initial', axis=1)
colnames = design.columns
theta = design.head(300)
drop_index = np.array([19, 23, 31, 32, 71, 91, 92, 98, 129, 131, 146, 162, 171, 174, 184, 190, 194, 195, 198])
theta = theta.drop(index=drop_index)
df_mean = df_mean.drop(index=drop_index)
df_sd = df_sd.drop(index=drop_index)

# A batch of 90 points with 800 events 
df_mean_b0 = pd.read_csv('../simulation_data/mean_for_90_add_batch0_800_events_design', index_col=0)
df_sd_b0 = pd.read_csv('../simulation_data/sd_for_90_add_batch0_800_events_design', index_col=0)
design_b0 = pd.read_csv('../design_data/add_design_122421.txt', delimiter = ' ')
drop_index_b0 = np.array([10, 17, 27, 35, 49, 58])
design_b0 = design_b0.drop(index=drop_index_b0)
df_mean_b0 = df_mean_b0.drop(index=drop_index_b0)
df_sd_b0 = df_sd_b0.drop(index=drop_index_b0)

# A batch of 90 points with 800 events 
df_mean_b1 = pd.read_csv('../simulation_data/mean_for_90_add_batch1_800_events_design', index_col=0)
df_sd_b1 = pd.read_csv('../simulation_data/sd_for_90_add_batch1_800_events_design', index_col=0)
design_b1 = pd.read_csv('../design_data/add_design_122721.txt', delimiter = ' ')
drop_index_b1 = np.array([0, 3, 6, 16, 18, 20, 26, 33, 37, 41, 48])
design_b1 = design_b1.drop(index=drop_index_b1)
df_mean_b1 = df_mean_b1.drop(index=drop_index_b1)
df_sd_b1 = df_sd_b1.drop(index=drop_index_b1)

# A batch of 90 points with 800 events 
df_mean_b2 = pd.read_csv("../simulation_data/mean_for_90_sliced_test_design_800_events_design", index_col=0)
df_sd_b2 = pd.read_csv("../simulation_data/sd_for_90_sliced_test_design_800_events_design", index_col=0)
design_b2 = pd.read_csv('../design_data/sliced_VAH_090321_test.txt', delimiter = ' ')
design_b2 = design_b2.drop(labels='tau_initial', axis=1)
design_b2 = design_b2.iloc[0:90]
drop_index_vl = np.array([29, 35, 76, 84, 87])
design_b2 = design_b2.drop(index=drop_index_vl)
df_mean_b2 = df_mean_b2.drop(index=drop_index_vl)
df_sd_b2 = df_sd_b2.drop(index=drop_index_vl)

# A batch of 75 points with 1600 events 
df_mean_b3 = pd.read_csv("../simulation_data/mean_for_75_batch0_design_1600_events_design", index_col=0)
df_sd_b3 = pd.read_csv("../simulation_data/sd_for_75_batch0_design_1600_events_design", index_col=0)
design_b3 = pd.read_csv('../design_data/add_design_020522.txt', delimiter = ' ')
#design_b3 = design_b3.drop(labels='tau_initial', axis=1)
#design_b2 = design_b2.iloc[0:90]
#drop_index_vl = np.array([29, 35, 76, 84, 87])
#design_b2 = design_b2.drop(index=drop_index_vl)
#df_mean_b2 = df_mean_b2.drop(index=drop_index_vl)

# Read the experimental data
exp_data = pd.read_csv('../PbPb2760_experiment', index_col=0)
y_mean = exp_data.to_numpy()[0, ]
y_sd = exp_data.to_numpy()[1, ]
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
df_mean_b0 = df_mean_b0[exp_label]
df_mean_b1 = df_mean_b1[exp_label]
df_mean_b2 = df_mean_b2[exp_label]
df_mean_b3 = df_mean_b3[exp_label]

df_sd = df_sd[exp_label]
df_sd_b0 = df_sd_b0[exp_label]
df_sd_b1 = df_sd_b1[exp_label]
df_sd_b2 = df_sd_b2[exp_label]
df_sd_b3 = df_sd_b3[exp_label]

selected_observables = exp_label[0:-32]

x_np = np.column_stack((x[0:-32], x_id[0:-32]))
x_np = x_np.astype('object')
y_mean = y_mean[0:-32]
y_sd = y_sd[0:-32]

#print(f'Last item on the selected observable is {selected_observables[-1]}')

df_mean = df_mean[selected_observables]
df_mean_b0 = df_mean_b0[selected_observables]
df_mean_b1 = df_mean_b1[selected_observables]
df_mean_b2 = df_mean_b2[selected_observables]
df_mean_b3 = df_mean_b3[selected_observables]

print(df_mean.shape)
print(df_mean_b0.shape)
print(df_mean_b1.shape)
print(df_mean_b2.shape)
print(df_mean_b3.shape)

df_sd = df_sd[selected_observables]
df_sd_b0 = df_sd_b0[selected_observables]
df_sd_b1 = df_sd_b1[selected_observables]
df_sd_b2 = df_sd_b2[selected_observables]
df_sd_b3 = df_sd_b3[selected_observables]

print('Total obs.:', df_mean.shape[0] + df_mean_b0.shape[0] + df_mean_b1.shape[0] + df_mean_b2.shape[0]+ df_mean_b3.shape[0])

print(theta.shape)
print(design_b0.shape)
print(design_b1.shape)
print(design_b2.shape)
print(design_b3.shape)

#frames800 = [df_mean, df_mean_b0, df_mean_b1, df_mean_b2]
#feval800 = pd.concat(frames800)
#print(feval800.shape)

#sdframes800 = [df_sd, df_sd_b0, df_sd_b1, df_sd_b2]
#sdeval800 = pd.concat(sdframes800)
#print(sdeval800.shape)

#frames_design800 = [theta, design_b0, design_b1, design_b2]
#design800 = pd.concat(frames_design800)
#print(design800.shape)

# Split test using 800
#msk = np.random.rand(len(feval800)) < 0.8
#df_mean_train1 = feval800[msk]
#df_sd_train1 = feval800[msk]
#df_mean_test = feval800[~msk]

#design_mean_train1 = design800[msk]
#theta_validation = design800[~msk]

feval800 = pd.concat([df_mean_b0, df_mean_b1, df_mean_b2])
sd800 = pd.concat([df_sd_b0, df_sd_b1, df_sd_b2])
design800 = pd.concat([design_b0, design_b1, design_b2])

# Split test using 800
msk = ((np.random.rand(len(feval800)) < 0.6)  +
       (np.arange(len(feval800)) > (design_b0.shape[0] + design_b1.shape[0])))
df_mean_train800 = feval800[msk]
df_sd_train800 = sd800[msk]
design_train800 = design800[msk]

df_mean_test = feval800[~msk]
theta_validation = design800[~msk]

feval = pd.concat([df_mean, df_mean_train800])
sdfeval = pd.concat([df_sd, df_sd_train800])
theta = pd.concat([theta, design_train800])

#frames = [df_mean, df_mean_train1]
#feval = pd.concat(frames)

#frames = [theta, design_mean_train1]
#theta = pd.concat(frames)

# Final training data
#theta = theta.to_numpy()
#feval = feval.to_numpy()

#theta = design_mean_train1.to_numpy()
#feval = df_mean_train1.to_numpy()
#sdfeval = df_sd_train1.to_numpy()

print('tr. shape (feval):', feval.shape)
print('tr. shape (theta):', theta.shape)
print('tr. shape (sfd):', sdfeval.shape)

# Final test data
theta_test = theta_validation.to_numpy()
feval_test = df_mean_test.to_numpy()

print('test shape (feval):', feval_test.shape)
print('test shape (feval):', theta_test.shape)

#assert feval.shape[0] + feval_test.shape[0] == df_mean.shape[0] + df_mean_b0.shape[0] + df_mean_b1.shape[0] + df_mean_b2.shape[0] + df_mean_b3.shape[0]


feval = np.transpose(feval)
sdfeval = np.transpose(sdfeval)
feval_test = np.transpose(feval_test)

# Observe simulation outputs in comparison to real data

#feval_test = np.log10(feval_test + 0.1)
#feval = np.log10(feval + 0.1)
# Build an emulator
emu_tr = emulator(x=x_np,
                   theta=theta,
                   f=feval,
                   method='PCSK',
                   args={'epsilonPC': 0.01,
                         'simsd': sdfeval,
                         'hypregmean': -16})

pred_test = emu_tr.predict(x=x_np, theta=theta_test)
pred_test_mean = pred_test.mean()
pred_test_var = pred_test.var()


# Check error
errors_test = (pred_test_mean - feval_test).flatten()
sst = np.sum((feval_test.flatten() - np.mean(feval_test.flatten()))**2)
print('MSE test=', np.mean(errors_test**2))
print('rsq test=', 1 - np.sum(errors_test**2)/sst)

rsq = []
for i in range(pred_test_mean.shape[0]):
    sse = np.sum((pred_test_mean[i, :] - feval_test[i, :])**2)
    sst = np.sum((feval_test[i, :] - np.mean(feval_test[i, :]))**2)
    rsq.append(1 - sse/sst)
    print(selected_observables[i])

plt.scatter(np.arange(pred_test_mean.shape[0]), rsq)
plt.xlabel('observables')
plt.ylabel('test rsq')
plt.show()

# Observe test prediction
fig = plt.figure()
plt.scatter(feval_test, pred_test_mean, alpha=0.5)
plt.plot(range(0, 5), range(0, 5), color='red')
plt.xlabel('Simulator outcome (test)')
plt.ylabel('Emulator prediction (test)')
plt.show()

fig, axis = plt.subplots(4, 2, figsize=(15, 15))
i, j = 0, 0
for o in uniquex:
    idx = o == x_np[:, 0]
    axis[i, j].scatter(feval_test[idx, :], pred_test_mean[idx, :], alpha=0.5)
    if np.max(pred_test_mean[idx, :]) > np.max(feval_test[idx, :]):
        xlu = np.ceil(np.max(pred_test_mean[idx, :]))
    else:
        xlu = np.ceil(np.max(feval_test[idx, :]))
    if np.min(pred_test_mean[idx, :]) > np.min(feval_test[idx, :]):
        xll = np.floor(np.min(feval_test[idx, :]))
    else:
        xll = np.floor(np.min(pred_test_mean[idx, :]))
    axis[i, j].plot(range(int(xll), int(xlu)+1), range(int(xll), int(xlu)+1), color='red')

    sse = np.sum((pred_test_mean[idx, :] - feval_test[idx, :])**2)
    sst = np.sum((feval_test[idx, :] - np.mean(feval_test[idx, :]))**2)
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
    e = (pred_test_mean[idx, :] - feval_test[idx, :]).flatten()
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
    e = ((pred_test_mean[idx, :] - feval_test[idx, :])/np.sqrt(pred_test_var[idx, :])).flatten()
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
    e = ((pred_test_mean[idx, :] - feval_test[idx, :])/feval_test[idx, :]).flatten()
    axis[i, j].hist(e, bins=25, density=True)
    axis[i, j].set_title(o)
    i += 1
    if i > 3:
        i = 0
        j = 1
plt.show()

# Check training
pred_tr = emu_tr.predict(x=x_np, theta=theta)
pred_tr_mean = pred_tr.mean()

# Check error
errors_tr = (pred_tr_mean - feval).flatten()
sst_tr = np.sum((feval.flatten() - np.mean(feval.flatten()))**2)
print('MSE train=', np.mean(errors_tr**2))
print('rsq train=', 1 - np.sum(errors_tr**2)/sst_tr)

# Observe test prediction
fig = plt.figure()
plt.scatter(feval, pred_tr_mean, alpha=0.5)
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
axs[0].hist((pred_test_mean-feval_test).flatten())
# This should look like a standard normal
axs[1].hist(((pred_test_mean-feval_test)/np.sqrt(pred_test_var)).flatten(), density=True)
axs[1].plot(x, stats.norm.pdf(x, mu, sigma), color='red')
axs[1].set_title(r'${(\hat{\mu}_{test} - \mu_{test})}/{\hat{\sigma}_{test}}$')
plt.show()
