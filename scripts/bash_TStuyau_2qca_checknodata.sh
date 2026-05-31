#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-02:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o TStuyau_nodata.%N.%a.%j.out # STDOUT
#SBATCH -e TStuyau_nodata.%N.%a.%j.err # STDERR
#SBATCH --job-name="qc_nodata"
#SBATCH --array=115
###################################################################
### Modify --array above with grid cell numbers to run as array job.

GRID_ID=$SLURM_ARRAY_TASK_ID
### note: if grid cell > 999, enter last three digits in array and use
### GRID_ID=$(($SLURM_ARRAY_TASK_ID + 2000))

###################################################################
#Settables:
ID_METHOD='pixel_check'
MOVE_METHOD='flag_dbX'
#filesize=400000  #use if ID_METHOD == filesize
BELOW=10

###################################################################
### Project settings
MAIN_DIR='/home/sandbox-cel'
PROJECT='paraguay_dl' 

####################################################################
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"

# activate the virtual environment
conda activate venv.tstuyau_dl

#####################################################################
# SHOULD NOT NEED TO MODIFY BELOW
#####################################################################

CONFIG_UPDATES="grids:[${GRIDS}] move_no_data:id_method:$ID_METHOD 
main_path:"${MAIN_DIR}/${PROJECT}/stac/grid" 
move_no_data:exclude_below:$BELOW move_no_data:move_method:$MOVE_METHOD"

tuyau move_nodata --config-updates $CONFIG_UPDATES

conda deactivate
