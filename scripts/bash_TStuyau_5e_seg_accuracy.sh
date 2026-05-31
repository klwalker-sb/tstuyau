#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o seg_acc.%N.%a.%j.out # STDOUT
#SBATCH -e seg_acc.%N.%a.%j.err # STDERR
#SBATCH --job-name="segacc"
###################################################################

POLYS="/home/downspout-cel/paraguay_lc/Segmentations/00_digitizations/PyCropSeg_Polys_8858.shp"
ACC_ID="cnet_training_regions_holdout.txt"
POS_CLASSES='[1]'
TARGET='crop'
VECMETHOD='EO' # 'EO" | "threshold" | 'water'
EOTHRESH=8.5 #needed if VECMETHOD = 'EO'
BOUNDTHRESH=0.4 # needed if VECMETHOD = 'threshold' or 'water'
EXTTHRESH=0.6 # needed if VECMETHOD = 'threshold' or 'water'
SEED=15 # needed if VECMETHOD = 'water'

ENFYR= 24
MODDIR="cnet4VIk"
PROJECT_VER='Py_0'
###################################################################
### Project settings

SEGDIRMAIN="home/downspout-cel/paraguay_lc/Segmentations"
MAIN_DIR="/home/sandbox-cel/"
BACKUP_DIR="/home/downspout-cel"
PROJECT="paraguay_lc"
PROJECT_DIR="${MAIN_DIR}/${PROJECT}/stac"
SCRATCH="/home/scratch-cel"

###################################################################
### activate the virtual environment
conda activate venv.tstuyau_pipe
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"
###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

CONFIG_UPDATES="
segment:seg_dir_main:$SEGDIRMAIN
segment:seg_dir_mod:$MODDIR
segment:seg_train-polys:$POLYS
segment:acc_id_file:$ACC_ID
segment:instance_method:$VECMETHOD
segment:seg_dir_main:$SEGDIRMAIN
segment:seg_dir_mod:$MODDIR
segment:prefix:$PRE
vectorize:eo_thresh:$EOTHRESH
vectorize:bound_thresh:$BOUNDTHRESH
vectorize:ext_thresh:$EXTTHRESH
vectorize:seed_size:$SEED
project_ver:$PROJECT_VER
segment:target:$TARGET
segment:pos_classes:$POS_CLASSES
scratch_dir:$SCRATCH
num_workers:${SLURM_CPUS_ON_NODE}"

tuyau segmentation_accuracy --config-updates $CONFIG_UPDATES

conda deactivate
