#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o tstuyau_s2cloudless.%N.%a.%j.out # STDOUT
#SBATCH -e tstuyau_s2cloudless.%N.%a.%j.err # STDERR
#SBATCH --job-name="clouds"
#SBATCH --array=1-20%2
################################################################
### Modify --array above with grid cell numbers to run as array job.

GRID_ID=$SLURM_ARRAY_TASK_ID
### note: if grid cell > 999, enter last three digits in array and use
### GRID_ID=$(($SLURM_ARRAY_TASK_ID + 2000))

## Set permissions on output files
umask 002

###################################################################
### Settables:

## DATES can be [0] for all relevant dates or range ['YYYY-M-D','YYYY-M-D']
DATES=['2020-1-1','2020-2-1'] 
RESET_DB='True'
NCHUNKS=512
MYGEEKEY="~/.tstuyau/gee_key.json"

###################################################################
### Project settings

EPSG=8858
RES=10.0
MAIN_DIR="/home/sandbox-cel/"
BK_DIR="/home/downspout-cel/"
PROJECT="paraguay_lc"
PROJECT_DIR="${MAIN_DIR}/${PROJECT}"
GRID_FILE="${PROJECT_DIR}/project_grid_${EPSG}.gpkg"
GEEIDX_DIR=${PROJECT_DIR}/raster"
###################################################################

export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"

### activate the virtual environment
conda activate venv.tstuyau_pipe

###################################################################
###################################################################
# SHOULD NOT NEED TO MODIFY BELOW
###################################################################
CONFIG_UPDATES="grids:[${GRIDS}] res:$RES
main_path:"${MAIN_DIR}/${PROJECT}/stac/grid"
backup_path:"${BK_DIR}/${PROJECT}/stac/grid" 
scratch_dir:$SCRATCH 
masking:sat_sensors:S2cp
masking:method:s2cloudless
status:reset_db:$RESET_DB
io:n_chunks:$NCHUNKS
masking:gee_index_dir:$GEEIDX_DIR
num_workers:${SLURM_CPUS_ON_NODE} 
masking:gee_key:$MYGEEKEY
masking:date_range:$DATES
io:file_format:geotiff
dlMehod:STAC
nodata:255"

tuyau mask --config-updates $CONFIG_UPDATES
