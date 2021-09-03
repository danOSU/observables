import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surmise.emulation import emulator
from surmise.calibration import calibrator
import scipy.stats as sps

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

df_mean = pd.read_csv('mean_for_150_design')
df_sd = pd.read_csv('sd_for_150_design')

df_mean.shape
df_sd.shape

design = pd.read_csv('design_20210829.txt', delimiter = ' ')

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
colname_theta = theta.columns[0:15]
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
theta_np = theta_np[:, 0:15]
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

# Observe test prediction
fig = plt.figure()
plt.scatter(f_test, pred_test_mean, alpha=0.5)
plt.xlabel('Simulator outcome (test)')
plt.ylabel('Emulator prediction (test)')
plt.show()

pred_test_var = pred_test.var()
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist((pred_test_mean-f_test).flatten())
# This should look like a standard normal
axs[1].hist(((pred_test_mean-f_test)/np.sqrt(pred_test_var)).flatten())
plt.show()

# Run calibrator for each column of the figure

# Define prior for parameters

class prior_VAH:
    def lpdf(theta):
        return (sps.uniform.logpdf(theta[:, 0], 10, 20) +  # Pb_Pb
                              sps.uniform.logpdf(theta[:, 1], -0.7, 1.4) + # Mean
                              sps.uniform.logpdf(theta[:, 2], 0.5, 1) + # Width
                              sps.uniform.logpdf(theta[:, 3], 0, 1.7) + # Dist
                              sps.uniform.logpdf(theta[:, 4], 0.3, 1.7) + # Flactuation
                              sps.uniform.logpdf(theta[:, 5], 0.135, 0.03) + # Temp
                              sps.uniform.logpdf(theta[:, 6], 0.13, 0.27) + # Kink       
                              sps.uniform.logpdf(theta[:, 7], 0.01, 0.19) + # eta_s  
                              sps.uniform.logpdf(theta[:, 8], -2, 3) + # slope_low  
                              sps.uniform.logpdf(theta[:, 9], -1, 3) + # slope_high 
                              sps.uniform.logpdf(theta[:, 10], 0.01, 0.24) + # max 
                              sps.uniform.logpdf(theta[:, 11], 0.12, 0.18) + # Temp_peak  
                              sps.uniform.logpdf(theta[:, 12], 0.025, 0.125) + # Width_peak      
                              sps.uniform.logpdf(theta[:, 13], -0.8, 1.6) + # Asym_peak
                              sps.uniform.logpdf(theta[:, 14], 0.3, 0.7)).reshape((len(theta), 1))   # tau_initial                                    
                                                             

    def rnd(n):
        return np.vstack((sps.uniform.rvs(10, 20, size=n), # 0 
                          sps.uniform.rvs(-0.7, 1.4, size=n), # 1 
                          sps.uniform.rvs(0.5, 1, size=n), # 2 
                          sps.uniform.rvs(0, 1.7, size=n), # 3
                          sps.uniform.rvs(0.3, 1.7, size=n), # 4 
                          sps.uniform.rvs(0.135, 0.03, size=n), # 5 
                          sps.uniform.rvs(0.13, 0.27, size=n), # 6 
                          sps.uniform.rvs(0.01, 0.19, size=n), # 7 
                          sps.uniform.rvs(-2, 3, size=n), # 8 
                          sps.uniform.rvs(-1, 3, size=n), # 9 
                          sps.uniform.rvs(0.01, 0.24, size=n), # 10 
                          sps.uniform.rvs(0.12, 0.18, size=n), # 11 
                          sps.uniform.rvs(0.025, 0.125, size=n), # 12 
                          sps.uniform.rvs(-0.8, 1.6, size=n), # 13 
                          sps.uniform.rvs(0.3, 0.7, size=n))).T  
    
# dET_deta, dN_dy_kaon, dN_dy_pion, dN_dy_proton
# Left column
# Mid-column
# Right column
u1 = ['dET_deta', 'dN_dy_kaon', 'dN_dy_pion', 'dN_dy_proton']
#u1 = ['dNch_deta', 'mean_pT_kaon', 'mean_pT_pion', 'mean_pT_proton']
#u1 = ['pT_fluct', 'v22', 'v32', 'v42']

xcal_all = []
ycal_all = []
y_calsd = []
ycal_sd_all = []
for u in u1:
    whereu = u == x_np[:, 0]
    x_cal = x_np[whereu, :]
    xcal_all.extend(x_cal)
    y_cal = y_mean[whereu]
    y_calsd = y_sd[whereu]
    ycal_all.extend(y_cal)
    ycal_sd_all.extend(y_calsd)
    
xcal_all = np.array(xcal_all)
ycal_all = np.array(ycal_all)
ycal_sd_all = np.array(ycal_sd_all)
#obsvar = np.maximum(0.1, 0.2*ycal_all)
obsvar = np.maximum(0.00001, ycal_sd_all)

#breakpoint()
cal_1 = calibrator(emu=emu_tr,
                   y=ycal_all,
                   x=xcal_all,
                   thetaprior=prior_VAH, 
                   method='directbayeswoodbury',
                   args={'sampler': 'PTLMC'},
                   yvar=0.1*obsvar)

theta_rnd = cal_1.theta.rnd(1000)

df = pd.DataFrame(theta_rnd, columns = colname_theta)
import seaborn as sns
fig, axs = plt.subplots(5, 3, figsize=(16, 16))
theta_prior = pd.DataFrame(prior_VAH.rnd(1000), columns = colname_theta)
theta_prior.hist(ax=axs)
df.hist(ax=axs, bins=25)


sns.pairplot(df)
plt.show()

post = cal_1.predict(xcal_all)
rndm_m = post.rnd(s = 1000)

fig, axis = plt.subplots(1, 4, figsize=(20, 5))
for u_id, u in enumerate(u1):
    whereu = u == xcal_all[:, 0]
    xp = xcal_all[whereu, :]
    yp = ycal_all[whereu]
    rnd_obs = rndm_m[:, whereu]
    
    for i in range(1000):
        axis[u_id].plot(xp[:, 1], rnd_obs[i, :], color='grey', zorder=1)
    axis[u_id].scatter(xp[:, 1], yp, color='red', zorder=2)
    axis[u_id].set_ylabel(u)
plt.show()
