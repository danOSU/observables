import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surmise.emulation import emulator
from surmise.calibration import calibrator
import scipy.stats as sps
from scipy import stats


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

df_sd = df_sd[selected_observables]
df_sd_b0 = df_sd_b0[selected_observables]
df_sd_b1 = df_sd_b1[selected_observables]
df_sd_b2 = df_sd_b2[selected_observables]
df_sd_b3 = df_sd_b3[selected_observables]

print('Total obs.:', df_mean.shape[0] + df_mean_b0.shape[0] + df_mean_b1.shape[0] + df_mean_b2.shape[0] + df_mean_b3.shape[0])


feval_add = pd.concat([df_mean_b0, df_mean_b1, df_mean_b2, df_mean_b3])
sd_add = pd.concat([df_sd_b0, df_sd_b1, df_sd_b2, df_sd_b3])
design_add = pd.concat([design_b0, design_b1, design_b2, design_b3])

# Split test using 800
msk = ((np.random.rand(len(feval_add)) < 0.6)  +
       (np.arange(len(feval_add)) > (df_mean_b0.shape[0] + df_mean_b1.shape[0])))

feval_add_train = feval_add[msk]
sd_add_train = sd_add[msk]
design_add_train = design_add[msk]

df_mean_test = feval_add[~msk]
theta_validation = design_add[~msk]

feval = pd.concat([df_mean, feval_add_train])
sdeval = pd.concat([df_sd, sd_add_train])
theta = pd.concat([theta, design_add_train])

# Final training/test data
theta = theta.to_numpy()
feval = feval.to_numpy()
sdeval = sdeval.to_numpy()
theta_test = theta_validation.to_numpy()
feval_test = df_mean_test.to_numpy()

#assert feval.shape[0] + feval_test.shape[0] == df_mean200.shape[0] + df_mean800_b0.shape[0] + df_mean800_b1.shape[0] + df_mean800_b2.shape[0]

feval = np.transpose(feval)
sdeval = np.transpose(sdeval)
feval_test = np.transpose(feval_test)



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