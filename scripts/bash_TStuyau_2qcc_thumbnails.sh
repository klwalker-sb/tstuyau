#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o Tstuyau_thumbs.%N.%a.%j.out # STDOUT
#SBATCH -e Tstuyau_thumbs.%N.%a.%j.err # STDERR
#SBATCH --job-name="thumbs"
#SBATCH --array=351

###################################################################
### Modify --array above with grid cell numbers to run as array job.

GRID_ID=$SLURM_ARRAY_TASK_ID
### note: if grid cell > 999, enter last three digits in array and use
### GRID_ID=$(($SLURM_ARRAY_TASK_ID + 2000)


#################################################################
### Settables:
IMGTYPE="LS2"
START="2022-01-01"
END="2026-01-02"
INCLUDE=''
EXCLUDE='X'
GAMMA=2
REDUCT=10

#################################################################
###  Project settings
MAIN_DIR="/home/sandbox-cel"
BACKUP_DIR= "home/downspout-cel"
PROJECT="paraguay_lc"

##################################################################

export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"

# activate the virtual environment
conda activate venv.tstuyau_pipe

###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

CONFIG_UPDATES="grids:[${GRIDS}]
plot:gamma:${GAMMA}
plot:reduct_factor:${REDUCT}
reconstruct:start:${START}
reconstruct:end:${END}
reconstruct:include:${INCLUDE}
reconstruct:exclude:${EXCLUDE}
image_type:${IMGTYPE}
main_path:${MAIN_DIR}/${PROJECT}/stac/grid
backup_path:${BACKUP_DIR}/${PROJECT}/stac/grid
num_workers:${SLURM_CPUS_ON_NODE}"

tuyau make_thumbnails --config-updates $CONFIG_UPDATES

conda deactivate

