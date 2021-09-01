#!/usr/bin/bash
for f in  ~/vah_run_events/design/[0-9]*/[0-9]*.results.dat
do
	folder=$(echo $f| cut -d '/' -f 6)
	echo $f
	python calculations_average_obs_design.py $f ~/vah_run_events/design/$folder/obs_Pb-Pb-2760.dat 
done
