import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surmise.emulation import emulator
from surmise.calibration import calibrator
import scipy.stats as sps
from scipy import stats


# Initial data set of 300 points with 200 events
df_mean = pd.read_csv('mean_for_300_sliced_200_events_design', index_col=0)
df_sd = pd.read_csv('sd_for_300_sliced_200_events_design', index_col=0)
design = pd.read_csv('sliced_VAH_090321.txt', delimiter = ' ')
design = design.drop(labels='tau_initial', axis=1)
colnames = design.columns
theta = design.head(300)
drop_index = np.array([19, 23, 31, 32, 71, 91, 92, 98, 129, 131, 146, 162, 171, 174, 184, 190, 194, 195, 198])
theta200 = theta.drop(index=drop_index)
df_mean200 = df_mean.drop(index=drop_index)
df_sd200 = df_sd.drop(index=drop_index)

# A batch of 90 points with 800 events 
df_mean_b0 = pd.read_csv('mean_for_90_add_batch0_800_events_design', index_col=0)
df_sd_b0 = pd.read_csv('sd_for_90_add_batch0_800_events_design', index_col=0)
design_b0 = pd.read_csv('add_design_122421.txt', delimiter = ' ')
drop_index_b0 = np.array([10, 17, 27, 35, 49, 58])
design800_b0 = design_b0.drop(index=drop_index_b0)
df_mean800_b0 = df_mean_b0.drop(index=drop_index_b0)
df_sd800_b0 = df_sd_b0.drop(index=drop_index_b0)

# A batch of 90 points with 800 events 
df_mean_b1 = pd.read_csv('mean_for_90_add_batch1_800_events_design', index_col=0)
df_sd_b1 = pd.read_csv('sd_for_90_add_batch1_800_events_design', index_col=0)
design_b1 = pd.read_csv('add_design_122721.txt', delimiter = ' ')
drop_index_b1 = np.array([0, 3, 6, 16, 18, 20, 26, 33, 37, 41, 48])
design800_b1 = design_b1.drop(index=drop_index_b1)
df_mean800_b1 = df_mean_b1.drop(index=drop_index_b1)
df_sd800_b1 = df_sd_b1.drop(index=drop_index_b1)

# A batch of 90 points with 800 events 
df_mean_b2 = pd.read_csv("mean_for_90_sliced_test_design_800_events_design", index_col=0)
df_sd_b2 = pd.read_csv("sd_for_90_sliced_test_design_800_events_design", index_col=0)
design_b2 = pd.read_csv('sliced_VAH_090321_test.txt', delimiter = ' ')
design_b2 = design_b2.drop(labels='tau_initial', axis=1)
design_b2 = design_b2.iloc[0:90]
drop_index_vl = np.array([29, 35, 76, 84, 87])
design800_b2 = design_b2.drop(index=drop_index_vl)
df_mean800_b2 = df_mean_b2.drop(index=drop_index_vl)
df_sd800_b2 = df_sd_b2.drop(index=drop_index_vl)

# Read the experimental data
exp_data = pd.read_csv('PbPb2760_experiment', index_col=0)
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
df_mean200 = df_mean200[exp_label]
df_mean800_b0 = df_mean800_b0[exp_label]
df_mean800_b1 = df_mean800_b1[exp_label]
df_mean800_b2 = df_mean800_b2[exp_label]

selected_observables = exp_label[0:-32]

x_np = np.column_stack((x[0:-32], x_id[0:-32]))
x_np = x_np.astype('object')
y_mean = y_mean[0:-32]
y_sd = y_sd[0:-32]

#print(f'Last item on the selected observable is {selected_observables[-1]}')

df_mean200 = df_mean200[selected_observables]
df_mean800_b0 = df_mean800_b0[selected_observables]
df_mean800_b1 = df_mean800_b1[selected_observables]
df_mean800_b2 = df_mean800_b2[selected_observables]

df_sd200 = df_sd200[selected_observables]
df_sd800_b0 = df_sd800_b0[selected_observables]
df_sd800_b1 = df_sd800_b1[selected_observables]
df_sd800_b2 = df_sd800_b2[selected_observables]

print('Total obs.:', df_mean200.shape[0] + df_mean800_b0.shape[0] + df_mean800_b1.shape[0] + df_mean800_b2.shape[0])


feval800 = pd.concat([df_mean800_b0, df_mean800_b1, df_mean800_b2])
sd800 = pd.concat([df_sd800_b0, df_sd800_b1, df_sd800_b2])
design800 = pd.concat([design800_b0, design800_b1, design800_b2])

# Split test using 800
msk = ((np.random.rand(len(feval800)) < 0.6)  +
       (np.arange(len(feval800)) > (df_mean800_b0.shape[0] + df_mean800_b1.shape[0])))
df_mean_train800 = feval800[msk]
df_sd_train800 = sd800[msk]
design_train800 = design800[msk]

df_mean_test = feval800[~msk]
theta_validation = design800[~msk]

feval = pd.concat([df_mean200, df_mean_train800])
sdeval = pd.concat([df_sd200, df_sd_train800])
theta = pd.concat([theta200, design_train800])

# Final training/test data
theta = theta.to_numpy()
feval = feval.to_numpy()
sdeval = sdeval.to_numpy()
theta_test = theta_validation.to_numpy()
feval_test = df_mean_test.to_numpy()

assert feval.shape[0] + feval_test.shape[0] == df_mean200.shape[0] + df_mean800_b0.shape[0] + df_mean800_b1.shape[0] + df_mean800_b2.shape[0]

feval = np.transpose(feval)
sdeval = np.transpose(sdeval)
feval_test = np.transpose(feval_test)


# Build an emulator
emu_tr = emulator(x=x_np,
                   theta=theta,
                   f=feval,
                   method='PCGPwM',
                   args={'epsilonPC':  0.1})
pred_test = emu_tr.predict(x=x_np, theta=theta_test)
pred_test_mean = pred_test.mean()
pred_test_var = pred_test.var()

sse = np.sum((pred_test_mean - feval_test) ** 2,1)
sst = np.sum((feval_test.T - np.mean(feval_test,1)) ** 2,0)
rsqwM = 1-sse/sst


# Build another emulator
emu_tr = emulator(x=x_np,
                   theta=theta,
                   f=feval,
                   method='PCSK',
                   args={'epsilonPC': 0.1, #this does not mean full errors
                         'simsd': sdeval})
pred_test = emu_tr.predict(x=x_np, theta=theta_test)
pred_test_mean = pred_test.mean()
pred_test_var = pred_test.var()

sse = np.sum((pred_test_mean - feval_test) ** 2,1)
sst = np.sum((feval_test.T - np.mean(feval_test,1)) ** 2,0)
rsqsk = 1-sse/sst

plt.plot(rsqwM, rsqwM, 'k-', lw=2)
plt.plot(rsqsk, rsqsk, 'k-', lw=2)
plt.scatter(rsqwM,rsqsk)
plt.xlabel('PCGPwM rsq')
plt.ylabel('PCSK rsq')
plt.show()

# say we are choosing these points.

# events = 2400 guess on r2
fv = pred_test.mean().var(1)
fpv = pred_test.var().mean(1)
vbar = np.square(sdeval).mean(1)

#
r2guess = fv/(vbar+fv+fpv)
plt.scatter(rsqsk,r2guess)
plt.plot(r2guess, r2guess, 'k-', lw=2)
plt.plot(rsqsk, rsqsk, 'k-', lw=2)
plt.ylabel('PCSK rsq')
plt.xlabel('predicted rsq')
plt.show()
#this shows this is a decent approximation, but we are overestimating the error.

