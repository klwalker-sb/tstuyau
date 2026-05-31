#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o TStuyau_RFmod.%N.%a.%j.out # STDOUT
#SBATCH -e TStuyau_RFmod.%N.%a.%j.err # STDERR
#SBATCH --job-name="RFmod"

###################################################################
###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel/"
BK_DIR="/home/downspout-cel/"
PROJECT="biltong"
PTFILE="${BK_DIR}/${PROJECT}/vector/pts_training/Biltong_LCsamp2024.csv"
PTSFT="${BK_DIR}/${PROJECT}/vector/pts_training/features"
PTSAMPDIR="${BK_DIR}/${PROJECT}/vector/pts_training/features"
VARDFDIR="${BK_DIR}/${PROJECT}/classification/inputs"

########## Settables #####################################################
IMGTYPE='LS2'
RES=10.0
PROCSEQ='mu.br.cga'
FEATMOD='base6svh23'
SAMPMOD='KWGEall'
LCMOD='LC29'
LUTCOL='LC29'
TRAINYRS=[2023]
OUTYR=2023
TESTTHRESH=10
MODDIR="${BK_DIR}/${PROJECT}/classification"
#TRAINYRS=[2021]
#MODNAME="${FEATMOD}_${SAMPMOD}_${LUTCOL}_1723"
LUT="${BK_DIR}/${PROJECT}/classification/LUT_Biltong.csv"
OPTIMIZE="none"
HO=False
HODIR=None
###################################################################
### activate the virtual environment
conda activate venv.tuyau

###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW

export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"
###################################################################

CONFIG_UPDATES="grids:$CELLS
main_path:${MAIN_DIR}/${PROJECT}/stac/grid
res:${RES}
image_type:${IMGTYPE}
procseq:${PROCSEQ}
backup_path:${BK_DIR}/${PROJECT}/stac/grid 
scratch_dir:$SCRATCH 
classify:mod_type:RF
classify:vardf_dir:$VARDFDIR
schematic_model:lut:$LUT
schematic_model:lc_mod:$LCMOD
sample_model:train_yrs:$TRAINYRS
sample_model:name:$SAMPMOD
sample_model:fixed_ho_dir:None
feature_model:name:$FEATMOD
feature_model:start_yr:$OUTYR
sample_model:optimize_on:$OPTIMIZE
sample_model:fixed_ho:$HO
sample_model:fixed_ho_dir:$HODIR
classify:existing_mod:None
feature_model:update_model_dict:False
sample_model:test_thresh:$TESTTHRESH
classify:mod_dir:$MODDIR
classify:n_est:400
classify:importance_method:Permutation
classify:perm_seed:88"
tuyau make_and_score_model --config-updates $CONFIG_UPDATES

conda deactivate
