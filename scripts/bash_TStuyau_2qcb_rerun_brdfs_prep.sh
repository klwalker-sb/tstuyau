#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-04:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o TStuyau_CleanBRDFs.%N.%a.%j.out # STDOUT
#SBATCH -e TStuyau_CleanBRDFs.%N.%a.%j.err # STDERR
#SBATCH --job-name="Clean_BRDFs"
#SBATCH --array=115
###################################################################
### Modify --array above with grid cell numbers to run as array job.

GRID_ID=$SLURM_ARRAY_TASK_ID
### note: if grid cell > 999, enter last three digits in array and use
### GRID_ID=$(($SLURM_ARRAY_TASK_ID + 2000))

## Set permissions on output files
umask 002

###################################################################
### Settables

CLEANUP='brdf'
SAT_SENSORS='All'
DATES='[0]'
XLIST='rerun_brdfs.csv'

###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel"
BACKUP_DIR = "home/downspout-cel"
PROJECT="paraguay_lc"

###################################################################
### activate the virtual environment
conda activate venv.tstuyau_pipe

###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"

CONFIG_UPDATES="grids:[${GRIDS}]
clean:sat_sensors:${SAT_SENSORS}
clean:remove_items:${CLEANUP}
clean:date_range:${DATES}
clean:xlist:${XLIST}
clean:prep_rerun:True
main_path:"${MAIN_DIR}/{PROJECT}/stac/grid"
backup_path:"${BACKUP_DIR}/{PROJECT}/stac/grid"
num_workers:${SLURM_CPUS_ON_NODE}"

tuyau clean --config-updates $CONFIG_UPDATES

conda deactivate
