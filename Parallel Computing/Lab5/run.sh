#!/bin/bash

#PBS -l nodes=2:ppn=4:ib,walltime=00:05:00
#PBS -q quad
#PBS -N pquick
#PBS -m a
#PBS -M duane.johnson@gmail.com

PROG="/fslhome/dmj35/cs484/pquick/pquicksort"
PROGARGS="--size $SIZE"
OUTFILE=""

# The following line changes to the directory that you submit your job from
cd $PBS_O_WORKDIR

mpiexec $PROG $PROGARGS

exit 0
