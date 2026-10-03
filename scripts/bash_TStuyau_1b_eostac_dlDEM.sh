#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH -t 0-20:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o stac_bl_DEM_bl.%N.%a.%j.out # STDOUT
#SBATCH -e stac_bl_DEM.%N.%a.%j.err # STDERR
#SBATCH --job-name="DEMdl"
#SBATCH --array=#88,115,116,143,148,173
##############################################

GRID_ID=$SLURM_ARRAY_TASK_ID
### note: if grid cell > 999, enter last three digits in array and use
### GRID_ID=$(($SLURM_ARRAY_TASK_ID + 2000))

# Set permissions on output files
umask 002

##Settables:
FILETYPE='.tif'
SENSORS='DEM'
L7STOPYR=2017
BUFFER=100
START_DATE='2000-01-01'
END_DATE='2030-01-01'

### Project settings

MAIN_DIR="/home/sandbox-cel/"
PROJECT="paraguay_lc"
PROJECT_HOME="${MAIN_DIR}/${PROJECT}/stac"
LANDSAT_DIR="${PROJECT_DIR}/grid/000${GRID_ID}/landsat"
SENTINEL_DIR="${PROJECT_DIR}/grid/000${GRID_ID}/sentinel2"
OUT_DIR="${PROJECT_DIR}/grid/000${GRID_ID}"
GRID_FILE="${PROJECT_DIR}/project_grid_${EPSG}.gpkg"
EPSG=8858

################################################
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"
### activate the virtual environment
conda activate venv.tstuyau_dl

eostac download --start-date $START_DATE --end-date $END_DATE --bounds $GRID_FILE --bounds-query UNQ==$GRID_ID --out-path $OUT_DIR --epsg $EPSG --bounds-buffer $BUFFER --sensors $SENSORS --l7-stop_year $L7STOPYR --max-items -1 -w 4 -t 2

conda deactivate
