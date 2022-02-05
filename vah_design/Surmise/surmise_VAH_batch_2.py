import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surmise.emulation import emulator
from surmise.calibration import calibrator
import scipy.stats as sps
from scipy import stats
import random

df_mean_b0 = pd.read_csv('mean_for_90_add_batch0_800_events_design', index_col=0)
design_b0 = pd.read_csv('add_design_122421.txt', delimiter = ' ')
drop_index_b0 = np.array([10, 17, 27, 35, 49, 58])
design_b0 = design_b0.drop(index=drop_index_b0)
df_mean_b0 = df_mean_b0.drop(index=drop_index_b0)
print(df_mean_b0.shape)

df_mean_b1 = pd.read_csv('mean_for_90_add_batch1_800_events_design', index_col=0)
design_b1 = pd.read_csv('add_design_122721.txt', delimiter = ' ')
drop_index_b1 = np.array([0, 3, 6, 16, 18, 20, 26, 33, 37, 41, 48])
design_b1 = design_b1.drop(index=drop_index_b1)
df_mean_b1 = df_mean_b1.drop(index=drop_index_b1)
print(df_mean_b1.shape)

frames = [df_mean_b0, df_mean_b1]
feval = pd.concat(frames)

frames = [design_b0, design_b1]
theta = pd.concat(frames)

print(feval.shape)
print(theta.shape)

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
feval = feval[exp_label]

selected_observables = exp_label[0:-32]
x_np = np.column_stack((x[0:-32], x_id[0:-32]))
x_np = x_np.astype('object')
y_mean = y_mean[0:-32]
y_sd = y_sd[0:-32]

#print(f'Last item on the selected observable is {selected_observables[-1]}')
feval = feval[selected_observables]
print(feval.shape)

# Remove nas
theta = theta.to_numpy()
feval = feval.to_numpy()

feval = np.transpose(feval)

# Observe simulation outputs in comparison to real data
fig, axis = plt.subplots(4, 2, figsize=(15, 15))
j = 0
k = 0
uniquex = np.unique(x_np[:, 0])
for u in uniquex:
    whereu = u == x_np[:, 0]
    for i in range(feval.shape[1]):
        axis[j, k].plot(x_np[whereu, 1].astype(int), feval[whereu, i], zorder=1, color='grey')
    axis[j, k].scatter(x_np[whereu, 1].astype(int), y_mean[whereu], zorder=2, color='red')
    axis[j, k].set_ylabel(u)
    if j == 3:
        j = 0
        k += 1
    else:
        j += 1
plt.show()

n = feval.shape[1]
n_test = 40
n_train = n - n_test

array = np.arange(0, n)
random.shuffle(array)
tr_id = array[0:n_train]
test_id = array[n_train:n]
theta_tr = theta[tr_id, :]
theta_test = theta[test_id, :]

feval_tr = feval[:, tr_id]
feval_test = feval[:, test_id]

feval_tr = np.log(feval_tr + 0.1)
feval_test = np.log(feval_test + 0.1)
# Build an emulator
emu_tr = emulator(x=x_np,
                  theta=theta_tr,
                  f=feval_tr,
                  method='PCGPwM',
                  args={'epsilon': 0.01})

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
pred_tr = emu_tr.predict(x=x_np, theta=theta_tr)
pred_tr_mean = pred_tr.mean()

# Check error
errors_tr = (pred_tr_mean - feval_tr).flatten()
sst_tr = np.sum((feval_tr.flatten() - np.mean(feval_tr.flatten()))**2)
print('MSE train=', np.mean(errors_tr**2))
print('rsq train=', 1 - np.sum(errors_tr**2)/sst_tr)

# Observe test prediction
fig = plt.figure()
plt.scatter(feval_tr, pred_tr_mean, alpha=0.5)
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
                              sps.uniform.logpdf(theta[:, 14], 0.3, 0.7)).reshape((len(theta), 1)) # R


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
#u1 = ['dET_deta', 'dN_dy_kaon', 'dN_dy_pion', 'dN_dy_proton',
#      'dNch_deta', 'mean_pT_kaon', 'mean_pT_pion', 'mean_pT_proton',
#      'pT_fluct', 'v22', 'v32', 'v42']
#u1 = ['dET_deta', 'dN_dy_kaon', 'dN_dy_pion', 'dN_dy_proton']
#u1 = ['dNch_deta', 'mean_pT_kaon', 'mean_pT_pion', 'mean_pT_proton']
#u1 = ['pT_fluct', 'v22', 'v32', 'v42']

#xcal_all = []
#ycal_all = []
#y_calsd = []
#ycal_sd_all = []
#for u in u1:
#    whereu = u == x_np[:, 0]
#    x_cal = x_np[whereu, :]
#    xcal_all.extend(x_cal)
#    y_cal = y_mean[whereu]
#    y_calsd = y_sd[whereu]
#    ycal_all.extend(y_cal)
#    ycal_sd_all.extend(y_calsd)

#xcal_all = np.array(xcal_all)
#ycal_all = np.array(ycal_all)
#ycal_sd_all = np.array(ycal_sd_all)
#obsvar = np.maximum(0.1, 0.2*ycal_all)
obsvar = np.maximum(0.00001, 0.2*y_mean)

#breakpoint()
#cal_1 = calibrator(emu=emu_tr,
#                   y=y_mean,
#                   x=x_np,
#                   thetaprior=prior_VAH,
#                   method='directbayes',
#                   args={'sampler': 'PTLMC'},
#                   yvar=obsvar)

#theta_rnd = cal_1.theta.rnd(1000)

#df = pd.DataFrame(theta_rnd, columns=colnames)
#import seaborn as sns
#fig, axs = plt.subplots(5, 3, figsize=(16, 16))
#theta_prior = pd.DataFrame(prior_VAH.rnd(1000), columns=colnames)
#theta_prior.hist(ax=axs)
#df.hist(ax=axs, bins=25)

#dfpost = pd.DataFrame(theta_rnd, columns = colnames)
#dfprior = pd.DataFrame(theta_prior, columns = colnames)
#df = pd.concat([dfprior, dfpost])
#pr = ['prior' for i in range(1000)]
#ps = ['posterior' for i in range(1000)]
#pr.extend(ps)
#df['distribution'] = pr
#map_parameters = [2.5, 2.5, .65, 5]
#sns.set_context('poster',font_scale=1)
#sns.set(style="white")
#def corrfunc(x, y, **kws):
#    r, _ = stats.pearsonr(x, y)
#    ax = plt.gca()
#    ax.annotate("r = {:.2f}".format(r),
#                xy=(.1, .9), xycoords=ax.transAxes)

#g = sns.PairGrid(df, palette=["blue", "red"], corner=True, diag_sharey=False, hue='distribution')
#g.map_diag(sns.kdeplot, shade=True)
#g.map_lower(sns.kdeplot, fill=True)
#g.map_lower(corrfunc)
#cn = ['rc', 'vr0', 'a', 'vso']
#for n,i in enumerate(map_parameters):
#    ax=g.axes[n][n]
 #   ax.axvline(x=map_parameters[n], ls='--')
#g.add_legend()
#g.savefig('surmise_vah_posterior',dpi=100)
#seaborn.reset_defaults()

#sns.pairplot(df)
#plt.show()

#post = cal_1.predict(xcal_all)
#rndm_m = post.rnd(s = 1000)

#fig, axis = plt.subplots(4, 3, figsize=(15, 15))
#for u_id, u in enumerate(u1):
#m = 0

#for k in range(3):
#    for j in range(4):
#        u = uniquex[m]
#        whereu = u == xcal_all[:, 0]
#        xp = xcal_all[whereu, :]
#        yp = ycal_all[whereu]
#        rnd_obs = rndm_m[:, whereu]

#        for i in range(500):
#            axis[j, k].plot(xp[:, 1], rnd_obs[i, :], color='grey', zorder=1)
#        axis[j, k].scatter(xp[:, 1], yp, color='red', zorder=2)
#        axis[j, k].set_ylabel(u)
#        m += 1
#plt.show()
