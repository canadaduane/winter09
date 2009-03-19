#!/bin/bash

#PBS -l nodes=32:ppn=2:ib,walltime=00:05:00
#PBS -N pquick64M
#PBS -m a
#PBS -M duane.johnson@gmail.com

PROG="/fslhome/dmj35/cs484/pquick/pquicksort"
PROGARGS="--size 62500"
OUTFILE=""

# The following line changes to the directory that you submit your job from
cd $PBS_O_WORKDIR

mpiexec $PROG $PROGARGS

exit 0

