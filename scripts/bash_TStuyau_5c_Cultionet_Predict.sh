#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o cnet_predict.%N.%a.%j.out # STDOUT
#SBATCH -e cnet_predict.%N.%a.%j.err # STDERR
#SBATCH --job-name="cnetpred"
#SBATCH --array=[900-999]%1
###################################################################
### Modify --array above with grid cell numbers to run as array job.
GRID_ID=$SLURM_ARRAY_TASK_ID
### note: if grid cell > 999, enter last three digits in array and use
#GRID_ID=$(($SLURM_ARRAY_TASK_ID + 3000))

YR=2024   ## Yr should be the last year of a training period if it spans 2 yrs
SIS="[gcvi,kndvi,nbr,ndmi]"
SEGDIR='/home/downspout-cel/paraguay_lc/Segmentations'
VERSION='cnet4VIk'
PROJECT_VER='Py_0'

###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel/"
BACKUP_DIR="/home/downspout-cel"
PROJECT="paraguay_lc"
SCRATCH="/home/scratch-cel"
EPSG=8858
GRID_FILE="${PROJECT_DIR}/project_grid_${EPSG}.gpkg"
RES=10.0
BUFFER=100
FIRSTMO=7

###################################################################
## Cultionet settings

END_YR=${YR}
CPU_GPU="gpu"
MMDD='07-01'
PRED_PREFIX="PY_"
RES=10
REGION=00${GRID_ID} ## change REGION to save only smaller chip, will look in time_series_vars dir but the images are already copied  
VERSION_DIR="/home/scratch-cel/cnet/${VERSION}"
## NEED TO FIRST MODIFY CULTIONET CONFIG FILE AND MAKE SURE IT IS IN THE SAME FOLDER THE MODEL WILL BE BUIlT IN (Version_dir)
CONFIG="${VERSION_DIR}/config_cultionet.yml"
OUT_FILE="${VERSION_DIR}/composites_probas/pred_${PRED_PREFIX}${GRID_ID}.tif"

## getting the first spectral index from list for the reference raster:
SISm="${SIS:1}"
SIS1="${SISm%%,*}"
REF_IMG="${VERSION_DIR}/time_series_vars/${REGION}/brdf_ts/ms/${SIS1}/${END_YR}001.tif"



###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"

conda activate venv.tuyau

CONFIG_UPDATES="grids:[${GRID_ID}]
segment:step:predict
grid_file:$GRID_FILE
buffer:$BUFFER
res:$RES
segment:update_polys:False
segment:temp_inputs:True
calendar:first_mo:$FIRSTMO
project_ver:$PROJECT_VER
segment:spec_indices:$SIS
sample_model:train_yrs:$YR
main_path:${MAIN_DIR}/${PROJECT}/stac/grid
backup_path:${BACKUP_DIR}/${PROJECT}/stac/grids
scratch_dir:$SCRATCH
segment:seg_dir_main:$SEGDIR
segment:seg_dir_mod:$VERSION
num_workers:${SLURM_CPUS_ON_NODE}"

## first run this to check ts folders and copy ts data into temp dir
tuyau prep_training_ts_for_segmentation --config-updates $CONFIG_UPDATES

## now run cultionet (with gpu) to predict the cell. Note this requires a different environment
cd ~
conda deactivate
conda activate .cultionet38

cd $VERSION_DIR/

cultionet create-predict -p . -y $END_YR -w 100 --padding 110 --ts-path $REGION --res $RES --append-ts y --image-date-format %Y%j -n 4 --config-file $CONFIG -sd $MMDD -ed $MMDD

cultionet predict -p . -y $END_YR -o $OUT_FILE -d data/predict/processed/ --region $REGION --ref-image $REF_IMG -g $GRID_ID -w 100 --padding 101 --device $CPU_GPU --batch-size 4 -sd $MMDD -ed $MMDD --config-file $CONFIG


conda deactivate



######################### EU settings

## GRID_DIR="/home/downspout-cel/paraguay_lc/Segmentations/AI4Boundaires/grid"
## VERSION_DIR="/home/downspout-cel/paraguay_lc/Segmentations/AI4Boundaires"
## GEOREF_IMG="${VERSION_DIR}/time_series_vars/${REGION}/brdf_ts/ms/evi2/2020001.tif"
## PRED_YR=2020 
## ED="01-01"
## SD="01-01"  


## PY cli: cultionet predict -p . -y 2021 -o /home/downspout-cel/paraguay_lc/Segmentations/composites_probas/ctrl_003949.tif -d data/predict/processed/ --region 003949 --ref-image /home/downspout-cel/paraguay_lc/Segmentations/time_series_vars/003949/brdf_ts/ms/evi2/2021001.tif -g 3949 -w 100 --padding 101 --config-file /home/downspout-cel/paraguay_lc/Segmentations/config_cultionet.yml  --device cpu --precision 32 --batch-size 8 -sd 07-01 -ed 07-01 
## cultionet predict -p . -y $PRED_YR -o $OUT_NAME -d data/predict/processed/ --region $REGION --ref-image $GEOREF_IMG -g $GRID_ID -w 100 --padding 101 --config-file $CONFIG_FILE  --device cpu --precision 32 --batch-size 8 -sd $SD -ed $ED $EXTRA_ARGS_pred
## EU
## cultionet predict -p . -y $PRED_YR -o $OUT_NAME -d data/predict/processed/ --region $REGION --ref-image $GEOREF_IMG -g $GRID_ID -w 100 --padding 101 --config-file $CONFIG_FILE  --device cpu --precision 32 --batch-size 8 -sd $SD -ed $ED $EXTRA_ARGS_pred

## 3) predict-transfer on grid 
## PY cli: cultionet predict-transfer -p . -y 2021 -o /home/downspout-cel/paraguay_lc/Segmentations/composites_probas/tsfr_003949.tif -d data/predict/processed/ -sd 07-01 -ed 07-01 --region 003949 --ref-image /home/downspout-cel/paraguay_lc/Segmentations/time_series_vars/003949/brdf_ts/ms/evi2/2021001.tif -g 3949 -w 100 --padding 101 --config-file /home/downspout-cel/paraguay_lc/Segmentations/config_cultionet.yml --device cpu --precision 32 --batch-size 8 --load-batch-workers 2 
##cultionet predict-transfer -p . -y $PRED_YR -o $OUT_NAME -d data/predict/processed/ -sd $SD -ed $ED --region $REGION --ref-image $GEOREF_IMG -g $GRID_ID -w 100 --padding 101 --config-file $CONFIG_FILE --device cpu --precision 32 --batch-size 8 --load-batch-workers 2 $EXTRA_ARGS_pred 

 
## test EU -> PY: use regular .cultionet38 environment to predict (not predict-transfer) w/ EU's last.ckpt 