#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o CnetTrain.%N.%a.%j.out # STDOUT
#SBATCH -e CnetTrain.%N.%a.%j.err # STDERR
#SBATCH --job-name="cnettr"
###################################################################


## STEP = 1 to create pytorch training dataset
## STEP = 2 to train ResUnet using pytorch training data from step 1
STEP=1

#STEP 1 PARAMS:
## NEED TO FIRST MODIFY CULTIONET CONFIG FILE AND MAKE SURE IT IS IN THE SAME FOLDER THE MODEL WILL BE BUIlT IN (Version_dir)
CONFIG="${VERSION_DIR}/config_cultionet.yml"
## VERSION_DIR is directory where model will be built. Creates a lot of temp files, so best to build in scratch directory and transfer final model
VERSION_DIR='/home/scratch-cel/cnet'
## TS_DIR is directory where TS images are located (front padded so each folder has 6 digits) 
##   -- but chips have been copied into version_dir in step 1, so don't think this is being used here.
TS_DIR='/home/downspout-cel/paraguay_lc/stac/grids'
## GS is grid size, in pixels
GS=100
## RES is resolution in m
RES=10.0
## MonthDay (with or without dash) to start & end ##PY: "07-01" | ##AI4B EU: "01-01" 
MMDD='07-01'
CROPCOL='class'
MAXPOS=1

#STEP 2 PARAMS:
VAL_FRAC=0.2
SEED=100
BATCH=8
NUM_EPOCHS=30   ###TRAIN-TRANSFER TESTING w/ LEARNING RATE: 100 | 150 | 200
LEARNING_RATE=0.01    ###TRAIN-TRANSFER TESTING w/ 0.001 | 0.0001 | 0.00001 
CPU_GPU='gpu'  ## "gpu" | "cpu"


###################################################################
### activate the virtual environment
conda activate cultionet38
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"
umask 002
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW

if [ "$STEP" -eq 1 ]; then
    cd $VERSION_DIR/
    cultionet create --project-path . -gs $GS $GS --res $RES --destination train --start-date $MMDD --end-date $MMDD --config-file $CONFIG --crop-column $CROPCOL --max-crop-class $MAXPOS $EXTRA_ARGS_create

if [ "$STEP" -eq 2 ]; then
    cultionet train -p . --val-frac $VAL_FRAC --random-seed $SEED --batch-size $BATCH_SIZE --epochs $NUM_EPOCHS -lr $LEARNING_RATE --start-date $MMDD --end-date $MMDD --device $CPU_GPU

    #cultionet train -p . --val-frac $VAL_FRAC --random-seed $SEED --batch-size $BATCH_SIZE --epochs $NUM_EPOCHS --expected-dim 13 --expected-height 100 --expected-width 100 --delete-mismatches -lr $LEARNING_RATE --start-date $MMDD --end-date $MMDD --device $CPU_GPU
    
fi

## params for alternative models and EU transfer training
#TS_DIR="/home/downspout-cel/paraguay_lc/Segmentations/AI4Boundaires/"

## EU cli: 
#cultionet train -p . --val-frac 0.2 --random-seed 130 --batch-size 8 --epochs 50 -lr 0.001 -sd 01-01 -ed 01-01
#EU transfer testin: cultionet train --project-path . --val-frac 0.2 --epochs $NUM_EPOCHS --accumulate-grad-batches 1 --model-type ResELUNetPsi --activation-type SiLU --res-block-type res --attention-weights spatial_channel --filters 32 --device gpu --processes 2 --load-batch-workers 2 --batch-size 8 --precision 16 --deep-sup-dist --deep-sup-edge --deep-sup-mask --lr-scheduler OneCycleLR
#cultionet train-transfer -p . --val-frac 0.2 --batch-size 8 --epochs 150 -lr 0.00001 -sd 07-01 -ed 07-01 --finetune
#train-transfer ResUnet using pytorch training data from 2) 

## PY cli: 
#cultionet train-transfer -p . --val-frac 0.2 --random-seed 130 --batch-size 8 --epochs 50 -lr 0.00001 -sd 07-01 -ed 07-01
## orig tsfr: 
#patience:50 epochs:150 lr:0.0001: cultionet train-transfer -p . --val-frac $VAL_FRAC --random-seed $SEED --batch-size $BATCH_SIZE -sd $SD -ed $ED $EXTRA_ARGS_train --patience 50 --epochs 150 -lr 0.0001
#cultionet train-transfer -p . --val-frac $VAL_FRAC --random-seed $SEED --batch-size $BATCH_SIZE -sd $SD -ed $ED $EXTRA_ARGS_train --patience 50 --epochs 150 -lr 0.0001

conda deactivate
