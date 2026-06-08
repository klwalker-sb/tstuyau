#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o Tstuyau_final_filters.%N.%a.%j.out # STDOUT
#SBATCH -e Tstuyau_final_filters.%N.%a.%j.err # STDERR
#SBATCH --job-name="filt"
#SBATCH --array=978,979

###################################################################
### Modify --array above with grid cell numbers to run as array job.
#GRID_ID=$SLURM_ARRAY_TASK_ID
### note: if grid cell > 999, enter last three digits in array and use
GRID_ID=$(($SLURM_ARRAY_TASK_ID + 3000))

YR=2024
#POLYS="/home/downspout-cel/paraguay_lc/Segmentations/cnet4VIk/infer_polys_EO_8pt5_2025/EO_8pt5_merged.gpkg"
POLYS="/home/downspout-cel/paraguay_lc/Segmentations/cnet4VIk/infer_polys_EO_8pt5_2025"
POLYVARS="/home/downspout-cel/paraguay_lc/Segmentations/cnet4VIk/feats_EO_8pt5_2025"
BUF=20
NBHD=100
## If TEST=True final outputs will be saved to scratch dir. Else to 'mosaics' is backup dir
TEST=True
## COMPDIR = 'input_dir' if looking for map inputs in comp folder of main_dir
##        or 'backup' if looking for map  inputs in comp folder in backup dir
COMPDIR='input_dir'
#SING=['cel2024-majority']
PROJECT_VER='Py_0'
###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel"
BK_DIR="/home/downspout-cel"
PROJECT="paraguay_lc"
PROJECT_DIR="${MAIN_DIR}/${PROJECT}/stac"
SCRATCH="/home/scratch-cel"
MOD="base4Poly6_bal200mix4_LC36_1723_RF_2024.tif"
EPSG=8858
GRID_FILE="${PROJECT_DIR}/project_grid_${EPSG}.gpkg"
RES=10.0
BUFFER=100
###################################################################
### activate the virtual environment
conda activate venv.lucinsa38_pipe
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"
###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

CONFIG_UPDATES="grids:[${GRID_ID}]
grid_file:$GRID_FILE
project_ver:$PROJECT_VER
classify:out_yrs:$YR
classify:name:relative_${MOD}
classify:test:$TEST
classify:comp_dir:$COMPDIR
refine:buffer:$BUF
refine:post_filter:smCrop
refine:neighborhood:$NBHD 
feature_model:poly_vector_path:$POLYS
feature_model:poly_var_path:$POLYVARS
main_path:${MAIN_DIR}/${PROJECT}/stac/grid
backup_path:${BK_DIR}/${PROJECT}/stac/grids
scratch_dir:$SCRATCH
num_workers:${SLURM_CPUS_ON_NODE}"

tuyau post_aggregation_filter --config-updates $CONFIG_UPDATES

conda deactivate
