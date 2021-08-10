import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surmise.emulation import emulator
from surmise.calibration import calibrator

df_mean = pd.read_csv('mean_for_ozge_150design')
df_sd = pd.read_csv('sd_for_ozge_150design')

df_mean.shape
df_sd.shape

design = pd.read_csv('design_20210627.txt', delimiter = ' ')

design.head()

design.shape

# Read the experimental data
exp_data = pd.read_csv('PbPb2760_experiment')
y_mean = exp_data.to_numpy()[0, ][1:]
y_sd = exp_data.to_numpy()[1, ][1:]

# Get the initial 150 parameter values
theta = design.head(150)

fig, axis = plt.subplots(4, 4, figsize=(10, 10))
theta.hist(ax=axis)
plt.show()

colname_exp = exp_data.columns
colname_sim = df_mean.columns

# Remove nas
which_nas = df_mean.isnull().any(axis=1)
theta_np = theta.to_numpy()
f_np = df_mean.to_numpy()

theta_np = theta_np[-which_nas, :]
f_np = f_np[-which_nas, :]

# Remove points where we do not have experimental data
f_np = np.transpose(f_np[:, 1:])
x = pd.read_excel('xlabels.xlsx', sheet_name='Sheet1')
x_np = x.to_numpy()
x_np = np.delete(x_np, [54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66], axis=0)
f_np = np.delete(f_np, [54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66], axis=0)

# Observe simulation outputs in comparison to real data
fig, axis = plt.subplots(4, 3, figsize=(15, 15))
j = 0
k = 0
uniquex = np.unique(x_np[:, 0])
for u in uniquex:
    whereu = u == x_np[:, 0]
    for i in range(f_np.shape[1]):
        axis[j, k].plot(x_np[whereu, 1], f_np[whereu, i],zorder=1, color='grey')
    axis[j, k].scatter(x_np[whereu, 1], y_mean[whereu],zorder=2, color='red')
    axis[j, k].set_ylabel(u)
    if j == 3:
        j = 0
        k += 1
    else:
        j += 1

# Split data into test and training
theta_test = theta_np[0:37, :] #theta_np#theta_np[0:37, :]
theta_tr = theta_np[37:, :] #theta_np#theta_np[37:, :]
f_test = f_np[:, 0:37] #f_np#f_np[:, 0:37]
f_tr = f_np[:, 37:]#f_np#f_np[:, 37:]

# Build an emulator 
emu_tr = emulator(x=x_np, 
                   theta=theta_tr, 
                   f=f_tr, 
                   method='PCGPwM') 

pred_test = emu_tr.predict(x=x_np, theta=theta_test)
pred_test_mean = pred_test.mean()

# Check error
print('SSE=', np.sqrt(np.sum((pred_test_mean - f_test)**2)))

