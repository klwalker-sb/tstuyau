#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o TStuyau_clean_rm.%N.%a.%j.out # STDOUT
#SBATCH -e TStuyau_clean_rm.%N.%a.%j.err # STDERR
#SBATCH --job-name="clean_rm"
#SBATCH --array=951
###################################################################
### Modify --array above with grid cell numbers to run as array job.

GRID_ID=$SLURM_ARRAY_TASK_ID
### note: if grid cell > 999, enter last three digits in array and use
### GRID_ID=$(($SLURM_ARRAY_TASK_ID + 2000))

## Set permissions on output files
umask 002

###################################################################
### DELETEREC='False' by default. 
###      setting to true will delete the record from the database so download/brdf
###      will be reprocessed in the future. 
### CAUTION: If cleaning out downloads, this will also delete the brdf file
###          even if brdf is not in CLEANUP

DELETEREC='True'

###################################################################
### Directories to clean: downloads,brdf,nocoreg,processing_db
###   format: 
###      list: comma, no quotes -- e.g. CLEANUP=downloads,nocoreg
###      single: brackets with quotes -- e.g. CLEANUP=['downloads']  

CLEANUP=downloads,brdf
#CLEANUP=['brdf']

###################################################################
### Sensor filter:
###   SAT_SENSORS=S2A,S2B,LT05,LE07,LC08,LC09  
###   can also do S2 for all sentinel only, L for all Landsat only, or 'All' for all
###   format:
###       list: comma, no quotes -- e.g. SAT_SENSORS=LT05,LE07
###       single: quotes (no brackets) -- e.g. SAT_SENSORS='LE07' 

SAT_SENSORS='L'

###################################################################
### Date filter:
###   format: '[YYYYMMDD, YYYYMMDD]' 
###   if a single image is to be cleaned, can use '[YYYYMMDD]'
###   if all files are to be cleaned (all dates), use '[0]'

DATES='[20190101,20220101]' 
#DATES='[0]'

###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel/"
BACKUP_DIR="/home/downspout-cel"
PROJECT="paraguay_lc"

###################################################################
### activate the virtual environment

export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"

conda activate venv.tstuyau_pipe

###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

CONFIG_UPDATES="grids:[${GRIDS}] res:${REF_RES} crs:${REF_CRS} 
dlMehod:STAC
clean:sat_sensors:${SAT_SENSORS}
clean:remove_items:${CLEANUP}
clean:delete_record:${DELETEREC}
clean:date_range:${DATES}
main_path:"${MAIN_DIR}/${PROJECT}/stac/grid"
backup_path:"${BACKUP_DIR}/${PROJECT}/stac/grid"
num_workers:${SLURM_CPUS_ON_NODE}"

tuyau clean --config-updates $CONFIG_UPDATES

conda deactivate
