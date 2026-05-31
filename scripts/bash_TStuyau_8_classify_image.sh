#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o TStuyau_RFclass.%N.%a.%j.out # STDOUT
#SBATCH -e TStuyau_RFclass.%N.%a.%j.err # STDERR
#SBATCH --job-name="RFclass"
#SBATCH --array=715-780%5
################################################################


## If a running smallish number of cells, can enter in array above and use these values as the cell input:
#CELLS="${SLURM_ARRAY_TASK_ID}"
CELLS="$(($SLURM_ARRAY_TASK_ID + 3000))"

## If running a lot of cells, can use a list. To parallelize, can split list into multiple lists in a directory:
#CELLS="/home/downspout-cel/paraguay_lc/cell_lists/Tile123Ready.csv"
#CELLS="/home/downspout-cel/paraguay_lc/cell_lists/TrainingCells/${SLURM_ARRAY_TASK_ID}.csv"
##############################################################################################
##############################################################################################
### Main Settables:

### Needed to find model and name output file: 
FEATMOD='base4Poly6'
SAMPMOD='bal200mix4'
LCMOD='pymax'
LUTCOL='LC36'
TRAINYRS=[2017,2023]
#TRAINYRS=[2021]
CLASSYR=2024
MODTYPE='RF'
OPTIMIZE='smCrop'
LUT="../classes_LUT.csv"

## OUTDIR can be 'backup', 'input_dir' or 'tmp' -- classified rasters will be in 'comp' subdirectory of OUTDIR
OUTDIR='input_dir'
OVERWRITE=True

##############################################################################################
### Project settings
MAIN_DIR="/home/sandbox-cel/"
BK_DIR="/home/downspout-cel/"
PROJECT="paraguay_lc"

CELLDIR="${MAIN_DIR}/${PROJECT}/stac/grid"
MODDIR="${BK_DIR}/${PROJECT}/classification"
SINGDICT="${BK_DIR}/${PROJECT}/ancillary_var_dict.json"
MODDICT="${BK_DIR}/${PROJECT}/Feature_Models.json"
VARDFDIR="${BK_DIR}/${PROJECT}/classification/inputs"
VARDF="${VARDFDIR}/pixdf_${MODNAME}.csv"
SEGPATH="${BK_DIR}/${PROJECT}/Segmentations"
POLYPATH="${SEG_PATH}/cnet4VIk/feats_EO_8pt5/$((CLASSYR+1))"
### Scratch dir (to save intermedite products). 
###    If blank, saves to 'comp' folder of TSDIR for each cell
#SCRATCH=""
SCRATCH="/home/scratch-cel/stackprods"
HODIR="${BK_DIR}/${PROJECT}/vector/pts_calval/fixedHOs"

#################################################################################################
### other Settables:
###   only needed if making new model directly -- not recommended

## ALTMOD is None If using default rf model named  ${FEATMOD}_${SAMPMOD}_${LUTCOL}_${TRAINYRS}_${MODTYPE}mod.joblib"
ALTMOD=None
#ALTMOD='currentMap_base4Poly6_bal200mix6_LC32_21_RFmod.joblib"
OVERWRITE=True
FIXEDHO=True
THRESH=20
RANHOLD=0
NTREES=100
IMPORTANCE=None
HOLDSEED=88

## Only need to update below if recreating new model without existing variable dataframe -- really not recommended
##################################
STARTMO=6
SIs=None
SIVARS=None
SING=None
POLYVARS="[]"
PHENOSIS=None
PHENOVARS=None
COMBOBANDS="[]"

###################################################################
### activate the virtual environmen
#conda activate venv.tuyau
conda activate venv.lucinsa38_pipe

###################################################################
###################################################################
# SHOULD NOT NEED TO MODIFY BELOW
###################################################################

#############################################
# Turn off NumPy parallelism and rely on dask
#############################################
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# This should be sufficient for OpenBlas and MKL
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"
################################################

CONFIG_UPDATES="grids:[${CELLS}]
main_path:${MAIN_DIR}/${PROJECT}/stac/grid
backup_path:${BK_DIR}/${PROJECT}/stac/grids
scratch_dir:$SCRATCH
classify:comp_dir:$OUTDIR 
classify:overwrite_image:$OVERWRITE
classify:out_yrs:$CLASSYR
feature_model:name:$FEATMOD 
sample_model:name:$SAMPMOD
sample_model:train_yrs:$TRAINYRS
schematic_model:lc_mod:$LCMOD
schematic_model:lut:$LUT
classify:mod_type:$MODTYPE
classify:mod_dir:$MODDIR
sample_model:fixed_ho:$FIXEDHO
sample_model:fixed_ho_dir:$HODIR
sample_model:test_thresh:$THRESH
sample_model:ran_hold_seed:$HOLDSEED
feature_model:feature_mod_dict:$MODDICT 
feature_model:singelton_var_dict:$SINGDICT 
classify:vardf_dir:$VARDFDIR
classify:n_est:$NTREES
classify:importance_method:$IMPORTANCE
classify:existing_mod:$ALTMOD
classify:optimize_on:$OPTIMIZE
feature_model:spec_indices:$SIs 
feature_model:si_vars:$SIVARS 
feature_model:spec_indices_pheno:$PHENOSIS 
feature_model:pheno_vars:$PHENOVARS
calendar:first_mo:$STARTMO 
feature_model:ancillary_vars:$SING 
feature_model:poly_vars:$POLYVARS 
feature_model:poly_var_path:$POLYPATH
feature_model:combo_bands:$COMBOBANDS" 

tuyau classify_timestep --config-updates $CONFIG_UPDATES

conda deactivate
