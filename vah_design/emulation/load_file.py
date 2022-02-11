import pandas as pd
class simulation:
    """
    A class to represent a batch of simulation.

    ...

    Attributes
    ----------
    obs : Pandas DataFrame
        Final simulation output. [n_designs x n_observables]
    obs_sd: Pandas DataFrame
        Final simulation output standard deviation. [n_designs x n_observables]
    design : Pandas DataFrame
        Design matrix. [n_designs x n_model_parameters]
    events : Pandas DataFrame
        Number of successful events per each  design. [n_designs,2]


    """
    def __init__(self, sim_path, sd_path, des_path, neve_path):
        """
        Construct simulation data object.

        Parametrs
        ---------
        sim_name : str
            path to the simulation output
        des_path : str
            path to the corresponding design file
        neve_path : str
            path to the corresponding number of events file
        """
        self.obs = pd.read_csv(sim_path, index_col=0)
        self.obs_sd = pd.read_csv(sd_path, index_col=0)
        self.design = pd.read_csv(des_path, delimiter = ' ')
        if 'tau_initial' in self.design:
            self.design = self.design.drop(['tau_initial'], axis=1)
        self.design = self.design.iloc[0:self.obs.shape[0]]
        event_dic = {'design':[],'nevents':[]}
        with open(neve_path) as f:
            for l in f:
                des_number = l.split('/')[-2]
                num_events = l.split(' ')[0]
                event_dic['design'].append(des_number)
                event_dic['nevents'].append( num_events)
        self.events = pd.DataFrame.from_dict(event_dic, dtype=float)
        # Drop last row because we did not consider the last design point when
        # gathering our results. This was by mistake.
        self.events = self.events[:-1]
