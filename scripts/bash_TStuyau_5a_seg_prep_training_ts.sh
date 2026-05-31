#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o seg_feats.%N.%a.%j.out # STDOUT
#SBATCH -e seg_feats.%N.%a.%j.err # STDERR
#SBATCH --job-name="segfeats"
#SBATCH --array=349-999%10
###################################################################
### Modify --array above with grid cell numbers to run as array job.

GRID_ID=$SLURM_ARRAY_TASK_ID
### note: if grid cell > 999, enter last three digits in array and use
#GRID_ID=$(($SLURM_ARRAY_TASK_ID + 2000))

STEP='train'  # 'train' | 'predict'
NEW_POLYS=False
## GET_LIST should be True for any runs with training cells
##    holdout cells should not be included in cell list here, or should have GET_LIST==False.
GET_LIST=True
CLIP_CHIPS=True
CHIP_SIZE=100  ## width, in num pixels

SIS=['kndvi','gcvi','wi','ndmi']
YR=2024 ## Yr should be the last year of a training period, if it spans 2 yrs
SEGDIR='/home/downspout-cel/paraguay_lc/Segmentations'
SUBDIR='cnet4VIk'
TRAINPOLYS="${SEGDIR}/00_digitizations/PyCropSeg_Polys_8858.shp"
PROJECT_VER='Py_0'
###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel/"
BACKUP_DIR="/home/downspout-cel"
PROJECT="paraguay_lc"
PROJECT_DIR="${MAIN_DIR}/${PROJECT}/stac"
SCRATCH="/home/scratch-cel"
EPSG=8858
GRID_FILE="${PROJECT_DIR}/project_grid_${EPSG}.gpkg"
RES=10.0
BUFFER=100
FIRSTMO=7
###################################################################
### activate the virtual environment
conda activate venv.tstuyau_pipe
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"
###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

CONFIG_UPDATES="grids:[${GRID_ID}]
grid_file:$GRID_FILE
buffer:$BUFFER
res:$RES
segment:step:$STEP
segment:update_polys:$NEW_POLYS
segment:seg_train_polys:$TRAINPOLYS
segment:clip_imagery:$CLIP_CHIPS
segment:train_chip_size:$CHIP_SIZE
segment:get_chip_list:$GET_LIST
segment:temp_inputs:True
calendar:first_mo:$FIRSTMO
project_ver:$PROJECT_VER
segment:spec_indices:$SIS
sample_model:train_yrs:$YRS
main_path:${MAIN_DIR}/${PROJECT}/stac/grid
backup_path:${BACKUP_DIR}/${PROJECT}/stac/grids
scratch_dir:$SCRATCH
segment:seg_dir_main:$SEGDIR
segment:seg_dir_mod:$SUBDIR
num_workers:${SLURM_CPUS_ON_NODE}"

tuyau prep_training_ts_for_segmentation --config-updates $CONFIG_UPDATES

conda deactivate
