import numpy as np
import pandas as pd

# 8 bins
ALICE_cent_bins = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]])

obs_cent_list = {
'Pb-Pb-2760': {
    'dNch_deta' : ALICE_cent_bins,
    'dET_deta' : np.array([[0, 2.5], [2.5, 5], [5, 7.5], [7.5, 10],
                           [10, 12.5], [12.5, 15], [15, 17.5], [17.5, 20],
                           [20, 22.5], [22.5, 25], [25, 27.5], [27.5, 30],
                           [30, 32.5], [32.5, 35], [35, 37.5], [37.5, 40],
                           [40, 45], [45, 50], [50, 55], [55, 60],
                           [60, 65], [65, 70]]), # 22 bins
    'dN_dy_pion'   : ALICE_cent_bins,
    'dN_dy_kaon'   : ALICE_cent_bins,
    'dN_dy_proton' : ALICE_cent_bins,
    'dN_dy_Lambda' : np.array([[0,5],[5,10],[10,20],[20,40],[40,60]]), # 5 bins
    'dN_dy_Omega'  : np.array([[0,10],[10,20],[20,40],[40,60]]), # 4 bins
    'dN_dy_Xi'     : np.array([[0,10],[10,20],[20,40],[40,60]]), # 4 bins
    'mean_pT_pion'   : ALICE_cent_bins,
    'mean_pT_kaon'   : ALICE_cent_bins,
    'mean_pT_proton' : ALICE_cent_bins,
    'pT_fluct' : np.array([[0,5],[5,10],[10,15],[15,20], [20,25],[25,30],[30,35],[35,40], [40,45],[45,50],[50,55],[55,60]]), #12 bins
    'v22' : ALICE_cent_bins,
    'v32' : np.array([[0,5],[5,10],[10,20],[20,30], [30,40],[40,50]]), # 6 bins
    'v42' : np.array([[0,5],[5,10],[10,20],[20,30], [30,40],[40,50]]), # 6 bins
    }
}

float_t = '<f8'
number_of_models_per_run = 4

sdtype = [('Pb-Pb-2760',
              [(obs, [("mean", float_t, len(cent_list)),
                      ("err", float_t, len(cent_list))]) \
                for obs, cent_list in obs_cent_list['Pb-Pb-2760'].items()],
           number_of_models_per_run)]


df_mean = {}
df_sd = {}
for i in range(0,150):
    result_data = np.fromfile(f"/home/ac.liyanage/vah_run_events/ozge_events/{i}/obs_Pb-Pb-2760.dat", dtype = sdtype)
    for obs, cent_list in obs_cent_list['Pb-Pb-2760'].items():
        for i,cen in enumerate(cent_list):
            mean=result_data["Pb-Pb-2760"][obs]["mean"][0][0][i]
            sd=result_data["Pb-Pb-2760"][obs]["err"][0][0][i]


            obs_name=obs+'_'+str(cen)
            print(f"for {obs_name} mean is {mean}")
            print(f"for {obs_name} sd is {sd}")
            if obs_name in df_mean:
                df_mean[obs_name].append(mean)
                df_sd[obs_name].append(sd)
            else:
                df_mean[obs_name] = [mean]
                df_sd[obs_name] = [sd]

df1 = pd.DataFrame(data=df_mean, index=np.arange(150))
df2 = pd.DataFrame(data=df_sd, index=np.arange(150))

df1.to_csv("mean_for_ozge_150design")
df2.to_csv("sd_for_ozge_150design")
