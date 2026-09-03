#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of tasks
#SBATCH -c 8 # number of cores-per-task
#SBATCH -t 0-24:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o Tstuyau_final_tsfilters.%N.%a.%j.out # STDOUT
#SBATCH -e Tstuyau_final_tsfilters.%N.%a.%j.err # STDERR
#SBATCH --job-name="tsfilt"
#SBATCH --array=10,5

YRS=[2017,2024]
PREFIX="CELPyTile${SLURM_ARRAY_TASK_ID}_base4Poly6_bal200mix4"
POST="filtsmh-sp"
SPFILT='smCrop'

STABLE="[shrub_for palm_for open_for dense_for wet grass_Py36]"
STABLEREG="[1 0 0 0 0 0 ]"
ILLOGICAL="[sugar-palm sugar-grass banana-wet palm_for-grass-aggressive palm_for-wetgrass grass-forest-aggressive tree_plant-med_crop rice-water rice-built forest-treeplant forest-brieflow grass-to-forest grass-to-palmforest twmix-medwet noplant-plant]"
ILLOGICALREG="[0 0 [2,3] [2,3] 0 0 0 0 0 0 0 0 [1, 4] 0 0]"
REGFILE="/home/downspout-cel/paraguay_lc/ancillary/forest_strata.tif"
## If TEST=True final outputs will be saved to scratch dir. Else to 'mosaics' is backup dir
TEST=True
SAVE_INTER=True
## If MODDIR is blank, inputs come from scratch_dir / classified
MODDIR=''
PROJECT_VER='Py_0'
SUFFIX='Py36'
###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel"
BK_DIR="/home/downspout-cel"
PROJECT="paraguay_lc"
PROJECT_DIR="${MAIN_DIR}/${PROJECT}/stac"
SCRATCH="/home/scratch-cel/${PROJECT}"
MOD="base4Poly6_bal200mix4_LC32_1723_RF_${YR}.tif"
EPSG=8858
RES=10.0
BUFFER=100
###################################################################
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK}"

### activate the virtual environment
conda activate tstuyau_pipe
###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

CONFIG_UPDATES="
buffer:$BUFFER
project_ver:$PROJECT_VER
classify:test:$TEST
refine:save_inter:$SAVE_INTER
classify:mod_dir:$MODDIR
refine:buffer:$BUF
refinemod_postscript:$POST
refine:mod_prescript:$PREFIX
refine:spatial_filter:$SPFILT
refine:stable_group:$STABLE
refine:stable_region:$STABLEREG
refine:group_suffix:$SUFFIX
refine:illogical:$ILLOGICAL
refine:illogical_regions:$ILLOGICALREG
refine:illogical_region_file:$REGFILE
classify:out_yrs:$YRS
main_path:${MAIN_DIR}/${PROJECT}/stac/grid
backup_path:${BK_DIR}/${PROJECT}/stac/grids
scratch_dir:$SCRATCH
num_workers:${SLURM_CPUS_ON_NODE}"

tuyau ts_filter --config-updates $CONFIG_UPDATES

conda deactivate
