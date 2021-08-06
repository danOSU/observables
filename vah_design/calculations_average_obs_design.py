#!/usr/bin/env python3

import numpy as np
import sys, os, glob
# Input data format
from calculations_file_format_single_event import *
# Output data format
from configurations import *
from calculations_average_obs import *

if __name__ == '__main__':
    print("Computing observables for design ")
    design_raw_obs_file = str(sys.argv[1])
    design_obs_out_file = str(sys.argv[2])
    for system in system_strs:
        try:
            print("System = " + system)
            file_input = design_raw_obs_file
            file_output = design_obs_out_file
            print("Averaging events in " + file_input)
            print("##########################")
            results = []
            print("starting load_and_compute")
            results.append(load_and_compute(file_input, system, specify_idf=idf)[0])
            results = np.array(results)
            print("writing to file")
            results.tofile(file_output)
        except:
            print("No MAP events found for system " + s)
