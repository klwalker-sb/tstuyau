#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o TStuyau_VarDFmod.%N.%a.%j.out # STDOUT
#SBATCH -e TStuyau_VarDFmod.%N.%a.%j.err # STDERR
#SBATCH --job-name="VarDFmod"

###################################################################
###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel/"
BK_DIR="/home/downspout-cel/"
PROJECT="paraguay_lc"
PTFILE="${BK_DIR}/${PROJECT}/vector/pts_training/SamplePts_Dec2024.csv"
PTSFT="${BK_DIR}/${PROJECT}/vector/pts_training/features"
PTSAMPDIR="${BK_DIR}/${PROJECT}"
VARDFDIR="${BK_DIR}/${PROJECT}/classification/inputs"
MODDICT="${BK_DIR}/${PROJECT}/Feature_Models.json"
########## Settables #####################################################
FEATMOD='base4NoPoly'
## Use SUBSET=True if a larger stack with a different name contains all desired features
SUBSET=True
FULLMOD='base4Poly6'

SAMPMOD='bal200mix6'
#FOCUS='EPy'
FOCUS=None
LCMOD='all'
LUTCOL='LC32'
TRAINYRS='[2017,2023]'
#TRAINYRS=[2021]
OUTYR=2022
#MODNAME="${FEATMOD}_${SAMPMOD}_${LUTCOL}_1723"
LUT="${BK_DIR}/${PROJECT}/Class_LUT.csv"
OPTIMIZE="smCrop"

###################################################################
### activate the virtual environment
conda activate venv.tstuyau_pipe
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"
###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

CONFIG_UPDATES="grids:$CELLS
main_path:${MAIN_DIR}/${PROJECT}/stac/grid
backup_path:${BK_DIR}/${PROJECT}/stac/grid 
scratch_dir:$SCRATCH 
grid_file:$GRID
classify:ptsfeat_dir:$PTSFT
classify:vardf_dir:$VARDFDIR
sample_model:point_file:$PTFILE
schematic_model:lut:$LUT
schematic_model:lc_mod:$LCMOD
sample_model:train_yrs:$TRAINYRS
sample_model:name:$SAMPMOD
sample_model:point_samp_dir:$PTSAMPDIR
sample_model:focus_area:$FOCUS
feature_model:name:$FEATMOD
feature_model:start_yr:$OUTYR
sample_model:optimize_on:$OPTIMIZE
feature_model:subset_features:$SUBSET
feature_model:full_feature_mod:$FULLMOD
feature_model:feature_mod_dict:$MODDICT"

tuyau format_ptfeat_set --config-updates $CONFIG_UPDATES

conda deactivate
