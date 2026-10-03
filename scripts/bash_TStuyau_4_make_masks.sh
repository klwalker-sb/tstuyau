#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 2 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o tstuyau_mask.%N.%a.%j.out # STDOUT
#SBATCH -e tstuyau_mask.%N.%a.%j.err # STDERR
#SBATCH --job-name="masks"
#SBATCH --array=177

GRIDS="${SLURM_ARRAY_TASK_ID}"

########## Settables ##############################################
METHOD='terrain_shade'
EXCLUDE='X'
SEN='LS2'
RESETDB=True
BUFFER=3
RETRY=False

###################################################################
### Project settings
MAIN_DIR="/home/sandbox-cel/"
BK_DIR="/home/downspout-cel/"
PROJECT="paraguay_lc"
DLMETHOD='stac'
RES=10.0
###################################################################
### activate the virtual environment
conda activate venv.tstuyau_pipe
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"
###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

CONFIG_UPDATES="grids:[${GRIDS}] 
main_path:${MAIN_DIR}/${PROJECT}/stac/grid 
backup_path:${BK_DIR}/${PROJECT}/stac/grid
masking:method:$METHOD
masking:exclude:$EXCLUDE
masking:sat_sensors:$SEN
masking:buffer_px:$BUFFER
masking:retry_masks:$RETRY
status:reset_db:$RESETDB
dlMethod:$DLMETHOD
res:$RES
"

tuyau mask --config-updates $CONFIG_UPDATES

conda deactivate
