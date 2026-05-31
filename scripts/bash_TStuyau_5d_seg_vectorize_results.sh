#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o seg_vectorize.%N.%a.%j.out # STDOUT
#SBATCH -e seg_vectorize.%N.%a.%j.err # STDERR
#SBATCH --job-name="vectorize"
###################################################################

MMU=900.0
VECMETHOD='EO' # 'EO" | "threshold" | 'water'
EOTHRESH=8.5 #needed if VECMETHOD = 'EO'
BOUNDTHRESH=0.4 # needed if VECMETHOD = 'threshold' or 'water'
EXTTHRESH=0.6 # needed if VECMETHOD = 'threshold' or 'water'
SEED=15 # needed if VECMETHOD = 'water'

ENDYR=2024
SEGDIR="home/downspout-cel/paraguay_lc/Segmentations"
MODDIR="cnet4VIk"
## note seg features are named based on the last year of the period
POLYS="${SEGDIR}/${MODDIR}/infer_polys_EO_8pt5_$((ENDYR+1))"
POLYVARS="${SEGDIR}/${MODDIR}/feats_EO_8pt5_$((ENDYR+1))"
TEMPIN=True

OVERWRITE_MERGE=True
CLEAN_TEMP=True
###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel/"
BACKUP_DIR="/home/downspout-cel"
PROJECT="paraguay_lc"
PROJECT_DIR="${MAIN_DIR}/${PROJECT}/stac"
SCRATCH="/home/scratch-cel"
EPSG=8858
GRID_FILE="${PROJECT_DIR}/project_grid_${EPSG}.gpkg"
RES=10.0
BUFFER=100
PROJECT_VER='Py_0'
PRE='Py'

###################################################################
### activate the virtual environment
conda activate venv.tstuyau_pipe
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"
###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

CONFIG_UPDATES="
buffer:$BUFFER
res:$RES
project_ver:$PROJECT_VER
sample_model:train_yrs:$ENDYR
vectorize:single_band:True
segment:instance_method:$VECMETHOD
vectorize:mmu:$MMU
segment:seg_dir_main:$SEGDIR
segment:seg_dir_mod:$MODDIR
segment:prefix:$PRE
feature_model:poly_vector_path:$POLYS
feature_model:unit_of_analysis:$UOA
feature_model:poly_var_path:$POLYVARS
segment:temp_inputs:$TEMPIN
segment:clean_temp_data:$CLEAN_TEMP
vectorize:eo_thresh:$EOTHRESH
vectorize:bound_thresh:$BOUNDTHRESH
vectorize:ext_thresh:$EXTTHRESH
vectorize:seed_size:$SEED
vectorize:overwrite_merged:$OVERWRITE_MERGE
main_path:${MAIN_DIR}/${PROJECT}/stac/grid
backup_path:${BACKUP_DIR}/${PROJECT}/stac/grid
scratch_dir:$SCRATCH
num_workers:${SLURM_CPUS_ON_NODE}"

tuyau vectorize_seg_results --config-updates $CONFIG_UPDATES

conda deactivate
