import streamlit as st
import numpy as np
import time
import os
import subprocess
import matplotlib
import altair as alt
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from configurations import *
#### Block 1 #### Please refer to this number in your questions


import math

from sklearn.decomposition import PCA
from numpy.linalg import inv
import sklearn, matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process import kernels as krnl

#import emce

import time
#### Block 2 #### Please refer to this number in your questions

name="JETSCAPE_bayes"
#Saved emulator name
EMU='PbPb2760_emulators_scikit.dat'
# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "Data/"

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)
#from configurations import *
#from emulator import Trained_Emulators, _Covariance
#from bayes_exp import Y_exp_data
#from bayes_plot import obs_tex_labels_2

# https://gist.github.com/beniwohli/765262
greek_alphabet_inv = { u'\u0391': 'Alpha', u'\u0392': 'Beta', u'\u0393': 'Gamma', u'\u0394': 'Delta', u'\u0395': 'Epsilon', u'\u0396': 'Zeta', u'\u0397': 'Eta', u'\u0398': 'Theta', u'\u0399': 'Iota', u'\u039A': 'Kappa', u'\u039B': 'Lamda', u'\u039C': 'Mu', u'\u039D': 'Nu', u'\u039E': 'Xi', u'\u039F': 'Omicron', u'\u03A0': 'Pi', u'\u03A1': 'Rho', u'\u03A3': 'Sigma', u'\u03A4': 'Tau', u'\u03A5': 'Upsilon', u'\u03A6': 'Phi', u'\u03A7': 'Chi', u'\u03A8': 'Psi', u'\u03A9': 'Omega', u'\u03B1': 'alpha', u'\u03B2': 'beta', u'\u03B3': 'gamma', u'\u03B4': 'delta', u'\u03B5': 'epsilon', u'\u03B6': 'zeta', u'\u03B7': 'eta', u'\u03B8': 'theta', u'\u03B9': 'iota', u'\u03BA': 'kappa', u'\u03BB': 'lamda', u'\u03BC': 'mu', u'\u03BD': 'nu', u'\u03BE': 'xi', u'\u03BF': 'omicron', u'\u03C0': 'pi', u'\u03C1': 'rho', u'\u03C3': 'sigma', u'\u03C4': 'tau', u'\u03C5': 'upsilon', u'\u03C6': 'phi', u'\u03C7': 'chi', u'\u03C8': 'psi', u'\u03C9': 'omega', }
greek_alphabet = {v: k for k, v in greek_alphabet_inv.items()}

zeta_over_s_str=greek_alphabet['zeta']+'/s(T)'
eta_over_s_str=greek_alphabet['eta']+'/s(T)'

v2_str='v'u'\u2082''{2}'
v3_str='v'u'\u2083''{2}'
v4_str='v'u'\u2084''{2}'


short_names = {
                'norm' : r'Energy Normalization', #0
                'trento_p' : r'TRENTo Reduced Thickness', #1
                'sigma_k' : r'Multiplicity Fluctuation', #2
                'nucleon_width' : r'Nucleon width [fm]', #3
                'dmin3' : r'Min. Distance btw. nucleons cubed [fm^3]', #4
                'Tswitch' : 'Particlization temperature [GeV]', #16
                'eta_over_s_T_kink_in_GeV' : r'Temperature of shear kink [GeV]', #7
                'eta_over_s_low_T_slope_in_GeV' : r'Low-temp. shear slope [GeV^-1]', #8
                'eta_over_s_high_T_slope_in_GeV' : r'High-temp shear slope [GeV^-1]', #9
                'eta_over_s_at_kink' : r'Shear viscosity at kink', #10
                'zeta_over_s_max' : r'Bulk viscosity max.', #11
                'zeta_over_s_T_peak_in_GeV' : r'Temperature of max. bulk viscosity [GeV]', #12
                'zeta_over_s_width_in_GeV' : r'Width of bulk viscosity [GeV]', #13
                'zeta_over_s_lambda_asymm' : r'Skewness of bulk viscosity', #14
                'initial_pressure_ratio' : r'R',
}


system_observables = {
                    'Pb-Pb-2760' : ['dNch_deta', 'dET_deta', 'dN_dy_pion', 'dN_dy_kaon' ,'dN_dy_proton', 'mean_pT_pion','mean_pT_kaon', 'mean_pT_proton', 'pT_fluct', 'v22', 'v32', 'v42'],
                    #'Au-Au-200' : ['dN_dy_pion', 'dN_dy_kaon', 'mean_pT_pion', 'mean_pT_kaon', 'v22', 'v32']
                    }

obs_lims = {'dNch_deta': 2000. , 'dET_deta' : 2000. , 'dN_dy_pion' : 2000., 'dN_dy_kaon' : 500., 'dN_dy_proton' : 100., 'mean_pT_pion' : 1., 'mean_pT_kaon' : 1., 'mean_pT_proton' : 2., 'pT_fluct' : .05, 'v22' : .2, 'v32' : .05, 'v42' :.03 }

obs_word_labels = {
                    'dNch_deta' : r'Charged multiplicity',
                    'dN_dy_pion' : r'Pion dN/dy',
                    'dN_dy_kaon' : r'Kaon dN/dy',
                    'dN_dy_proton' : r'Proton dN/dy',
                    'dN_dy_Lambda' : r'Lambda dN/dy',
                    'dN_dy_Omega' : r'Omega dN/dy',
                    'dN_dy_Xi' : r'Xi dN/dy',
                    'dET_deta' : r'Transverse energy [GeV]',
                    'mean_pT_pion' : r'Pion mean pT [GeV]',
                    'mean_pT_kaon' : r'Kaon mean pT [GeV]',
                    'mean_pT_proton' : r'Proton mean pT [GeV]',
                    'pT_fluct' : r'Mean pT fluctuations',
                    'v22' : v2_str,
                    'v32' : v3_str,
                    'v42' : v4_str,
}

system = 'Pb-Pb-2760'

#@st.cache(persist=True)
def load_design(system):
    #load the design
    design = pd.read_csv('design_20210627.txt', delimiter = ' ')
    labels = design.columns
    design_range = pd.read_csv(filepath_or_buffer="\Data\priorVAH.csv", index_col=0)
    design_max = design_range.values[1,:]
    design_min = design_range.values[0,:]
    return design, labels, design_max, design_min


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_emu(system, idf):
    #load the emulator
    with open('Data/PbPb2760_emulators_scikit.dat',"rb") as f:
        emu=pickle.load(f)
    return emu

@st.cache(persist=True)
def load_obs(system):
    observables = system_observables[system]
    nobs = len(observables)
    experiment=pd.read_csv("Data/PbPb2760_experiment",index_col=0)

    Yexp = experiment.values
    return observables, nobs, Yexp


design = pd.read_csv('design_20210829.txt', delimiter = ' ')
design = design.iloc[0:150]
simulation = pd.read_csv(filepath_or_buffer=data_path("mean_for_150_design"), index_col=0)
simulation_sd = pd.read_csv(filepath_or_buffer=data_path("sd_for_150_design"), index_col=0)

drop_index=np.unique(np.argwhere(np.isnan(simulation).values)[:,0])
design=design.drop(index=drop_index)
simulation=simulation.drop(index=drop_index)
simulation_sd=simulation_sd.drop(index=drop_index)
experiment=pd.read_csv(filepath_or_buffer="PbPb2760_experiment",index_col=0)

exp_label=[]
for i in experiment.columns:
    words=i.split('[')
    exp_label.append(words[0]+'_['+words[1])


simulation=simulation[exp_label]
simulation_sd=simulation_sd[exp_label]

#### Block 6 #### Please refer to this number in your questions

X = design.values
Y = simulation.values

print( "X.shape : "+ str(X.shape) )
print( "Y.shape : "+ str(Y.shape) )

#Model parameter names in Latex compatble form
model_param_dsgn = ['$N$[$2.76$TeV]',
 '$p$',
 '$\\sigma_k$',
 '$w$ [fm]',
 '$d_{\\mathrm{min}}$ [fm]',
 '$\\tau_R$ [fm/$c$]',
 '$\\alpha$',
 '$T_{\\eta,\\mathrm{kink}}$ [GeV]',
 '$a_{\\eta,\\mathrm{low}}$ [GeV${}^{-1}$]',
 '$a_{\\eta,\\mathrm{high}}$ [GeV${}^{-1}$]',
 '$(\\eta/s)_{\\mathrm{kink}}$',
 '$(\\zeta/s)_{\\max}$',
 '$T_{\\zeta,c}$ [GeV]',
 '$w_{\\zeta}$ [GeV]',
 '$\\lambda_{\\zeta}$',
 '$b_{\\pi}$',
 '$T_{\\mathrm{sw}}$ [GeV]']

 #### Block 7 #### Please refer to this number in your questions
#Scaling the data to be zero mean and unit variance for each observables
SS  =  StandardScaler(copy=True)
#Singular Value Decomposition
u, s, vh = np.linalg.svd(SS.fit_transform(Y), full_matrices=True)
print(f'shape of u {u.shape} shape of s {s.shape} shape of vh {vh.shape}')


#### Block 9 #### Please refer to this number in your questions
#whiten and project data to principal component axis (only keeping first 10 PCs)
pc_tf_data=u[:,0:10] * math.sqrt(u.shape[0]-1)
print(f'Shape of PC transformed data {pc_tf_data.shape}')
#Scale Transformation from PC space to original data space
inverse_tf_matrix= np.diag(s[0:10]) @ vh[0:10,:] * SS.scale_.reshape(1,110)/ math.sqrt(u.shape[0]-1)

def predict_observables(model_parameters, Emulators):
    mean=[]
    variance=[]
    theta=np.array(model_parameters).flatten()

    if len(theta)!=15:
        raise TypeError('The input model_parameters array does not have the right dimensions')
    else:
        theta=np.array(theta).reshape(1,15)
        for i in range(0,10):
            mn,std=Emulators[i].predict(theta,return_std=True)
            mean.append(mn)
            variance.append(std**2)
    mean=np.array(mean).reshape(1,-1)
    inverse_transformed_mean=mean @ inverse_tf_matrix + np.array(SS.mean_).reshape(1,-1)
    variance_matrix=np.diag(np.array(variance).flatten())
    A_p=inverse_tf_matrix
    inverse_transformed_variance=np.einsum('ik,kl,lj-> ij', A_p.T, variance_matrix, A_p, optimize=False)
    return inverse_transformed_mean, inverse_transformed_variance



#@st.cache(allow_output_mutation=True, show_spinner=False)
def emu_predict(emu, params):
    start = time.time()
    Yemu_cov = 0
    #Yemu_mean = emu.predict( np.array( [params] ), return_cov=False )
    Yemu_mean, Yemu_cov = predict_observables( np.array( [params] ), emu )
    end = time.time()
    time_emu = end - start
    return Yemu_mean, Yemu_cov, time_emu

#@st.cache(show_spinner=False)
def make_plot_altair(observables, Yemu_mean, Yemu_cov, Yexp, idf):
    ii=0
    for iobs, obs in enumerate(observables):
        print(obs)
        xbins = np.array(obs_cent_list[system][obs])
        cen_i = len(xbins)
        #centrality bins
        x = (xbins[:,0]+xbins[:,1])/2.
        #emulator prediction
        i_new=ii+cen_i
        y_emu = Yemu_mean[0,ii:i_new]

        dy_emu = (np.diagonal(np.abs(Yemu_cov[ii:i_new, ii:i_new]))**.5)

    #    print(y_emu)
    #    print(dy_emu)
    #    print(x)
        df_emu = pd.DataFrame({'cent': x, 'yl':y_emu - dy_emu, "yh":y_emu + dy_emu})
        chart_emu = alt.Chart(df_emu).mark_area().encode(x='cent', y='yl', y2='yh').properties(width=150,height=150)

        #experiment
        exp_mean = Yexp[0,ii:i_new]
        exp_err = np.sqrt(Yexp[1,ii:i_new])
        print(exp_mean)
        print(exp_err)
        ii=i_new

        df_exp = pd.DataFrame({"cent": x, obs:exp_mean, obs+"_dy":exp_err, obs+"_dy_low":exp_mean-exp_err, obs+"_dy_high":exp_mean+exp_err})

        # Adjust font size for the v_n's
        normal_font_size=14
        if (obs in ['v22','v32','v42']):
            normal_font_size=18

        pre_chart_exp=alt.Chart(df_exp)

        chart_exp = pre_chart_exp.mark_circle(color='white').encode(
        x=alt.X( 'cent', axis=alt.Axis(title='Centrality (%)', titleFontSize=14), scale=alt.Scale(domain=(0, 70)) ),
        y=alt.Y(obs, axis=alt.Axis(title=obs_word_labels[obs], titleFontSize=normal_font_size), scale=alt.Scale(domain=(0, obs_lims[obs]))  )
        )

        # generate the error bars
        errorbars = pre_chart_exp.mark_errorbar().encode(
                x=alt.X('cent', axis=alt.Axis(title='')),
                y=alt.Y(obs+"_dy_low", axis=alt.Axis(title=''), scale=alt.Scale(domain=(0, obs_lims[obs]))  ),
                y2=alt.Y2(obs+"_dy_high"),
        )

        chart = alt.layer(chart_emu, chart_exp + errorbars)

        if iobs == 0:
            charts0 = chart
        if iobs in [1, 2, 3]:
            charts0 = alt.hconcat(charts0, chart)

        if iobs == 4:
            charts1 = chart
        if iobs in [5, 6, 7]:
            charts1 = alt.hconcat(charts1, chart)

        if iobs == 7:
            charts2 = chart
        if iobs in [8, 9, 10]:
            charts2 = alt.hconcat(charts2, chart)
    charts0 = st.altair_chart(charts0)
    charts1 = st.altair_chart(charts1)
    charts2 = st.altair_chart(charts2)

    #return charts0, charts1, charts2

#@st.cache(suppress_st_warning=True)
def update_plot_altair(Yemu_mean, Yemu_cov, Yexp, idf, charts0, charts1, charts2):
    ii=0
    for iobs, obs in enumerate(observables):
        xbins = np.array(obs_cent_list[system][obs])
        cen_i = len(xbins)
        #centrality bins
        x = (xbins[:,0]+xbins[:,1])/2.
        #emulator prediction
        i_new=ii+cen_i
        y_emu = Yemu_mean[0,ii:i_new]

        dy_emu = (np.diagonal(np.abs(Yemu_cov[ii:i_new, ii:i_new]))**.5)
        df_emu = pd.DataFrame({'cent': x, 'yl':y_emu - dy_emu, "yh":y_emu + dy_emu})
        ii=i_new
        charts0.add_rows(df_emu)


def make_plot_eta_zeta(params):
    T_low = 0.1
    T_high = 0.35
    T = np.linspace(T_low, T_high, 100)
    eta_s = eta_over_s(T, *params[7:11])
    zeta_s = zeta_over_s(T, *params[11:15])

    df_eta_zeta = pd.DataFrame({'T': T, 'eta':eta_s, 'zeta':zeta_s})

    chart_eta = alt.Chart(df_eta_zeta, title='Specific shear viscosity').mark_line(strokeWidth=4).encode(
    x=alt.X('T', axis=alt.Axis(title='T [GeV]', titleFontSize=14), scale=alt.Scale(domain=(T_low, T_high)) ),
    y=alt.Y('eta', axis=alt.Axis(title=eta_over_s_str, titleFontSize=14), scale=alt.Scale(domain=(0., 0.5 ))  ),
    color=alt.value("#FF0000")
    ).properties(width=150,height=150)

    chart_zeta = alt.Chart(df_eta_zeta, title='Specific bulk viscosity').mark_line(strokeWidth=4).encode(
    x=alt.X('T', axis=alt.Axis(title='T [GeV]', titleFontSize=14), scale=alt.Scale(domain=(T_low, T_high)) ),
    y=alt.Y('zeta', axis=alt.Axis(title=zeta_over_s_str, titleFontSize=14), scale=alt.Scale(domain=(0., 0.5 ))  ),
    color=alt.value("#FF0000")
    ).properties(width=150,height=150)

    #st_chart = st.altair_chart(chart)
    #charts = alt.hconcat(chart_zeta, chart_eta)
    #st.write(chart_zeta)
    #st.write(chart_eta)

def main():
    st.title('Hadronic Observable Emulator for Heavy Ion Collisions')
    st.markdown('Our [model](https://inspirehep.net/literature/1821941) for the outcome of [ultrarelativistic heavy ion collisions](https://home.cern/science/physics/heavy-ions-and-quark-gluon-plasma) include many parameters which affects final hadronic observables in non-trivial ways. You can see how each observable (blue band) depends on the parameters by varying them using the sliders in the sidebar(left). All observables are plotted as a function of centrality for Pb nuclei collisions at'r'$\sqrt{s_{NN}} = 2.76$ TeV.')
    st.markdown('The experimentally measured observables by the [ALICE collaboration](https://home.cern/science/experiments/alice) are shown as black dots.')
    st.markdown('The last row displays the temperature dependence of the specific shear and bulk viscosities (red lines), as determined by different parameters on the left sidebar.')
    st.markdown('By default, these parameters are assigned the values that fit the experimental data *best* (maximize the likelihood).')
    st.markdown(r'An important modelling ingredient is the particlization model used to convert hydrodynamic fields into individual hadrons. Three different viscous correction models can be selected by clicking the "Particlization model" button below.')

    idf_names = ['Grad', 'Chapman-Enskog R.T.A', 'Pratt-Torrieri-Bernhard']
    idf_name = st.selectbox('Particlization model',idf_names)

    # Reset button
    st.markdown('<a href="javascript:window.location.href=window.location.href">Reset</a>', unsafe_allow_html=True)


    inverted_idf_label = dict([[v,k] for k,v in idf_label.items()])
    idf = inverted_idf_label[idf_name]

    #load the design
    design, labels, design_max, design_min = load_design(system)
    #design_min = [10, -0.7, 0.5, 0, 0.3, 0.135, 0.13, 0.01, -2, -1, 0.01, 0.12, 0.025, -0.8, 0.3]
    #design_max = [30, 0.7, 1.5, 4.91, 2, 0.165, 0.3, 0.2, 1, 2, 0.25, 0.3, 0.15, 0.8, 1]
    #load the emu
    emu = load_emu(system, idf)

    #load the exp obs
    observables, nobs, Yexp = load_obs(system)

    #initialize parameters
    #params_0 = (design_min+design_max)/2
    #print(params_0)
    params = []
    print(design_min)
    print(design_max)
    #updated params
    for i_s, s_name in enumerate(short_names.keys()):
        min = design_min[i_s]
        max = design_max[i_s]
        middle = (min+max)/2
        step = (max - min)/100.
        p = st.sidebar.slider(short_names[s_name], min_value=min, max_value=max, value=round(float(middle),1)+0.001, step=step)
        params.append(p)

    #get emu prediction
    Yemu_mean, Yemu_cov, time_emu = emu_predict(emu, params)

    #redraw plots
    make_plot_altair(observables, Yemu_mean, Yemu_cov, Yexp, idf)
    make_plot_eta_zeta(params)

    st.header('How it works')
    st.markdown('A description of the physics model and parameters can be found [here](https://indico.bnl.gov/event/6998/contributions/35770/attachments/27166/42261/JS_WS_2020_SIMS_v2.pdf).')
    st.markdown('The observables above (and additional ones not shown) are combined into [principal components](https://en.wikipedia.org/wiki/Principal_component_analysis) (PC).')
    st.markdown('A [Gaussian Process](https://en.wikipedia.org/wiki/Gaussian_process) (GP) is fitted to each of the dominant principal components by running our physics model on a coarse [space-filling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling) set of points in parameter space.')
    st.markdown('The Gaussian Process is then able to interpolate between these points, while estimating its own uncertainty.')

    st.markdown('To update the widget with latest changes, click the button below, and then refresh your webpage.')
    if st.button('(Update widget)'):
        subprocess.run("git pull origin master", shell=True)


if __name__ == "__main__":
    main()
