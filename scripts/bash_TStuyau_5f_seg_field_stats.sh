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

YR=2024

SIS=['gcvi']
SIVARS=['avg-NovDec-std']
SEGDIR="home/downspout-cel/paraguay_lc/Segmentations"
MODDIR='cnet4VIk'
## note seg features are named with the last year of the time period
POLYS="${SEGDIR}/${MODDIR}/infer_polys_EO_8pt5_$((YR+1))"
POLYVARS="${SEGDIR}/${MODDIR}/feats_EO_8pt5_$((YR+1))"
UOA='pixel'

PROJECT_VER='Py_0'
MAKEBLANKS=True
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

###################################################################
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"
### activate the virtual environment
conda activate venv.tstuyau
###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

CONFIG_UPDATES="grids:[${GRID_ID}]
grid_file:$GRID_FILE
buffer:$BUFFER
res:$RES
project_ver:$PROJECT_VER
feature_model:spec_indices:$SIS
feature_model:si_vars:$SIVARS
sample_model:train_yrs:$YR
feature_model:poly_vector_path:$POLYS
feature_model:unit_of_analysis:$UOA
feature_model:poly_var_path:$POLYVARS
segment:make_blank_vars:$MAKEBLANKS
main_path:${MAIN_DIR}/${PROJECT}/stac/grid
backup_path:${BACKUP_DIR}/${PROJECT}/stac/grid
scratch_dir:$SCRATCH
num_workers:${SLURM_CPUS_ON_NODE}"

tuyau make_polygon_features --config-updates $CONFIG_UPDATES

conda deactivate
