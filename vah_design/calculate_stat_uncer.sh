#!/bin/bash
set -u
for i in {0..49};
do
	echo $i
	input_dir="/Users/dananjayaliyanage/git/observables/vah_design/stat_uncertainty/$i-fraction.dat"
	output_dir="/Users/dananjayaliyanage/git/observables/vah_design/stat_uncertainty/$i-obs.dat"
	echo "input file is"
	echo $input_dir
	echo "output file is"
	echo $output_dir
	python calculations_average_obs_design.py $input_dir $output_dir
done
