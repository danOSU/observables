import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from surmise.emulation import emulator
from surmise.calibration import calibrator
import scipy.stats as sps
from scipy import stats



#import pyximport
#pyximport.install(setup_args={"include_dirs":np.get_include()},
#                  reload_support=True)


n_train = 200

df_mean = pd.read_csv('mean_for_200_sliced_200_events_design', index_col=0)

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
df_mean.head()
selected_observables = exp_label[0:-32]
df_mean = df_mean[selected_observables]

# Remove bad designs
drop_index = np.array([19, 23, 31, 32, 71, 91, 92, 98, 129, 131, 146, 162, 171, 174, 184, 190, 194, 195, 198])
drop_index_vl = np.array([29, 35, ])
df_mean = df_mean.drop(index=drop_index)
df_mean.shape

# Remove nas
f_np = df_mean.to_numpy()
f_np = np.transpose(f_np)

def __standardizef(fitinfo, offset=None, scale=None):
    "Standardizes f by creating offset, scale and fs."
    # Extracting from input dictionary
    f = fitinfo['f']

    if (offset is not None) and (scale is not None):
        if offset.shape[0] == f.shape[1] and scale.shape[0] == f.shape[1]:
            if np.any(np.nanmean(np.abs(f-offset)/scale, 1) > 4):
                offset = None
                scale = None
        else:
            offset = None
            scale = None
    if offset is None or scale is None:
        offset = np.zeros(f.shape[1])
        scale = np.zeros(f.shape[1])
        for k in range(0, f.shape[1]):
            offset[k] = np.nanmean(f[:, k])
            scale[k] = np.nanstd(f[:, k])
            if scale[k] == 0:
                scale[k] = 0.0001

    # Initializing values
    fs = np.zeros(f.shape)
    fs = (f - offset) / scale

    # Assigning new values to the dictionary
    fitinfo['offset'] = offset
    fitinfo['scale'] = scale
    fitinfo['fs'] = fs
    return


def __PCs(fitinfo):
    "Apply PCA to reduce the dimension of f"
    # Extracting from input dictionary
    f = fitinfo['f']
    fs = fitinfo['fs']
    epsilon = fitinfo['epsilon']
    pct = None
    pcw = None

    U, S, _ = np.linalg.svd(fs.T, full_matrices=False)
    Sp = S ** 2 - epsilon
    pct = U[:, Sp > 0]
    pcw = np.sqrt(Sp[Sp > 0])
    pcstdvar = np.zeros((f.shape[0], pct.shape[1]))

    fitinfo['pcw'] = pcw
    fitinfo['pcto'] = 1*pct
    fitinfo['pct'] = pct * pcw / np.sqrt(pct.shape[0])
    fitinfo['pcti'] = pct * (np.sqrt(pct.shape[0]) / pcw)
    fitinfo['pc'] = fs @ fitinfo['pcti']
    fitinfo['extravar'] = np.mean((fs - fitinfo['pc'] @
                                   fitinfo['pct'].T) ** 2, 0) *\
        (fitinfo['scale'] ** 2)
    fitinfo['pcstdvar'] = 10*pcstdvar
    return

fitinfo = {}
fitinfo['epsilon'] = 0.1
fitinfo['f'] = f_np.T
__standardizef(fitinfo)
__PCs(fitinfo)

for i in range(5):
    plt.plot(np.arange(0, 78), fitinfo['pct'][:, i], label=str(i))
plt.xlabel(r'all observables')
plt.ylabel(r'PCA weights')
plt.legend()
plt.show()
#plt.plot(cs_x[0:50], fitinfo['pct'][:, 1])
#plt.plot(cs_x[0:50], fitinfo['pct'][:, 2])
