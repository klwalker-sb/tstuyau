#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH -t 0-12:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o TStuyau_moditer.%N.%a.%j.out # STDOUT
#SBATCH -e TStuyau_moditer.%N.%a.%j.err # STDERR
#SBATCH --job-name="moditer"
###################################################################
###################################################################
### Main settables

TRAINYRS='[2021]'
CLASSYR=2021
FOCUS='EPy'  # 'All' or None if full area to be modeled
SAMPMOD='bal100'
FEATMOD='base4Poly'
LCMOD='all'
RNG_REDUCE='[0,.7]'
INC_REDUCE=0.1
RNG_MINSAMP='[100,300]'
INC_MINSAMP=100
RNG_MINBAL='[0,7]'
ITER=10
MODTYPE='RF'
RNG_TREES='[100,300]'
INC_TREES=100

NUMHOPULLS=20
HOPULLSEED=758
##############################################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel/"
BK_DIR="/home/downspout-cel/"
PROJECT="paraguay_lc"
PTFILE="${BK_DIR}/${PROJECT}/vector/pts_training/SamplePts_Dec2024.csv"
PTSFT="${BK_DIR}/${PROJECT}/vector/pts_training/features"
MODDIR="${BK_DIR}/${PROJECT}/classification/${MODTYPE}"
PTSAMPDIR="${BK_DIR}/${PROJECT}"
VARDFDIR="${BK_DIR}/${PROJECT}/classification/inputs"
SCRATCH="/home/scratch-cel/"
HODIR="${BK_DIR}/${PROJECT}/vector/pts_calval/fixedHOs"
OPTDIR="${BK_DIR}/${PROJECT}/optimization"
HOLDSEED=88
###################################################################
### activate the virtual environment
conda activate venv.tstuyau_pipe

###################################################################
###################################################################
# SHOULD NOT NEED TO MODIFY BELOW
###################################################################

CONFIG_UPDATES="main_path:${MAIN_DIR}/${PROJECT}/stac/grid
backup_path:${BK_DIR}/${PROJECT} 
scratch_dir:${SCRATCH}/rf_mods 
sample_model:name:$SAMPMOD
sample_model:train_yrs:$TRAINYRS
sample_model:fixed_ho:$FIXEDHO
sample_model:fixed_ho_dir:$HODIR
sample_model:test_thresh:$THRESH
sample_model:ran_hold_seed:$HOLDSEED
sample_model:num_subsamples:$NUMHOPULLS
sample_model:subsamp_seed:$HOPULLSEED
sample_model:point_samp_dir:$PTSAMPDIR
feature_model:name:$FEATMOD
feature_model:feature_mod_dict:$MODDICT 
feature_model:ancillary_var_dict:$SINGDICT 
schematic_mod:lc_mod:$LCMOD
schematic_mod:lut:$LUT
classify:vardf_dir:$VARDFDIR
classify:importance_method:$IMPORTANCE
classify:optimize_on:$OPTIMIZE
sample_model:overwrite_ho:False
feature_model:update_mod_dic:False
iter_models:feat_models:$FEATMODS
sample_model:focus_area:$FOCUS
iter_models:range_minsamp:$RNG_MINSAMP
iter_models:inc_minsamp:$INC_MINSAMP
iter_models:range_minbal:$RNG_MINBAL
iter_models:range_reduce:$RNG_REDUCE
iter_models:inc_reduce:$INC_REDUCE
classify:mod_type:$MODTYPE
iter_models:range_est:$RNG_TREES
iter_models:inc_est:$INC_TREES
iter_models:iterations:$ITER
iter_models:opt_dir:$OPTDIR
"

tuyau iterate_sample_model --config-updates $CONFIG_UPDATES

conda deactivate

