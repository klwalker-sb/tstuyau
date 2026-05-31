#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 2 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o stacpipe_crg.%N.%a.%j.out # STDOUT
#SBATCH -e stacpipe_crg.%N.%a.%j.err # STDERR
#SBATCH --job-name="stpipe_crg"
#SBATCH --array=900-950%10
###################################################################
### Modify --array above with grid cell numbers to run as array job.

GRID_ID=$SLURM_ARRAY_TASK_ID
### note: if grid cell > 999, enter last three digits in array and use
### GRID_ID=$(($SLURM_ARRAY_TASK_ID + 2000))

## Set permissions on output files
umask 002

###################################################################
### Settables:

IMGTYPE='LS2'
NCHUNKS=512
RERUN="True"

## NEW_REF is whether to remake the reference image if it already exists. 
##    should be False if a small subset is being run relative to the full sequence
NEW_REF="True"
MAX_SHIFT=5
###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel/"
BACKUP_DIR="/home/downspout-cel"
PROJECT="paraguay_lc"

###################################################################
### activate the virtual environment
conda activate venv.tstuyau_pipe

###################################################################
###################################################################
# SHOULD NOT NEED TO MODIFY BELOW
###################################################################

#############################################
# Turn off NumPy parallelism and rely on dask
#############################################
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# This should be sufficient for OpenBlas and MKL
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"
################################################

CONFIG_UPDATES="grids:[${GRIDS}] res:${REF_RES} crs:${REF_CRS} 
image_type:$IMGTYPE
status:reset_db:${RERUN}
main_path:"${MAIN_DIR}/${PROJECT}/stac/grid"
backup_path:"${BACKUP_DIR}/${PROJECT}/stac/grid"
num_workers:${SLURM_CPUS_ON_NODE} 
io:n_chunks:${NCHUNKS}
coreg:overwrite_ref:${NEW_REF}
coreg:max_shift:${MAX_SHIFT}
"

tuyau preprocess --config-updates $CONFIG_UPDATES

conda deactivate
