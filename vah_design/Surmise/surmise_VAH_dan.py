import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surmise.emulation import emulator
from surmise.calibration import calibrator
import scipy.stats as sps
from scipy import stats
import seaborn as sns

import numpy as np
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
# Gather what type of experimental data do we have.
exp_label = []
x = []
j = 0
x_id = []
for i in experiment.columns:
    words = i.split('[')
    exp_label.append(words[0]+'_['+words[1])
    if words[0] in x:
        j += 1
    else:
        j = 0
    x_id.append(j)
    x.append(words[0])
# Only keep simulation data that we have corresponding experimental data
y_train = y_train[exp_label]
y_test = y_test[exp_label]
label_sd = ['sd_'+exp for exp in exp_label]
y_er_train = y_er_train[label_sd]

x_np = np.column_stack((x, x_id))
x_np = x_np.astype('object')

print(y_train.shape)
print(y_er_train.shape)
print(y_test.shape)
print(y_test.shape)


# Final training/test data
theta = x_train.to_numpy()
feval = y_train.to_numpy()
sdeval = y_er_train.to_numpy()
theta_test = x_test.to_numpy()
feval_test = y_test.to_numpy()

#assert feval.shape[0] + feval_test.shape[0] == df_mean200.shape[0] + df_mean800_b0.shape[0] + df_mean800_b1.shape[0] + df_mean800_b2.shape[0]

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

fig, ax = plt.subplots(figsize=(50,5))
ax.scatter(y_train.keys(),rsqwM)
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
#plt.plot(rsqsk, rsqsk, 'k-', lw=2)
plt.scatter(rsqwM)
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
plt.xlabel('PCSK rsq')
plt.ylabel('predicted rsq')
plt.show()
#this shows this is a decent approximation, but we are overestimating the error.
