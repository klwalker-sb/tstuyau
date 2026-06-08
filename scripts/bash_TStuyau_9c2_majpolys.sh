#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o seg_maj.%N.%a.%j.out # STDOUT
#SBATCH -e seg_maj.%N.%a.%j.err # STDERR
#SBATCH --job-name="segmaj"
#SBATCH --array=978,979

###################################################################
### Modify --array above with grid cell numbers to run as array job.
#GRID_ID=$SLURM_ARRAY_TASK_ID
### note: if grid cell > 999, enter last three digits in array and use
GRID_ID=$(($SLURM_ARRAY_TASK_ID + 3000))

SIS=['']
SIVARS=['']
YR=2024
#POLYS="/home/downspout-cel/paraguay_lc/Segmentations/cnet4VIk/infer_polys_EO_8pt5_2025/EO_8pt5_merged.gpkg"
POLYS="/home/downspout-cel/paraguay_lc/Segmentations/cnet4VIk/infer_polys_EO_8pt5_2025"
POLYVARS="/home/downspout-cel/paraguay_lc/Segmentations/cnet4VIk/majclass"
UOA='pixel'
SING=['cel2024rc-majority']
#SING=['cel2024-majority']
PROJECT_VER='Py_0'
MAKEBLANKS=False
###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel"
BK_DIR="/home/downspout-cel"
PROJECT="paraguay_lc"
PROJECT_DIR="${MAIN_DIR}/${PROJECT}/stac"
SINGDICT="${BK_DIR}/${PROJECT}/ancillary_var_dict.json"
SCRATCH="/home/scratch-cel"
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
buffer:$BUFFER
res:$RES
project_ver:$PROJECT_VER
feature_model:spec_indices:$SIS
feature_model:si_vars:$SIVARS
sample_model:train_yrs:$YR
feature_model:poly_vector_path:$POLYS
feature_model:unit_of_analysis:$UOA
feature_model:poly_var_path:$POLYVARS
segment:make_blank_vars':$MAKEBLANKS
feature_model:ancillary_vars:$SING
feature_model:ancillary_var_dict:$SINGDICT
main_path:${MAIN_DIR}/${PROJECT}/stac/grid
backup_path:${BK_DIR}/${PROJECT}/stac/grids
scratch_dir:$SCRATCH
num_workers:${SLURM_CPUS_ON_NODE}"

tuyau make_polygon_features --config-updates $CONFIG_UPDATES

conda deactivate
