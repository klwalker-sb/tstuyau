#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 2 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o stac_bl_applymasks.%N.%a.%j.out # STDOUT
#SBATCH -e stac_bl_applymasks.%N.%a.%j.err # STDERR
#SBATCH --job-name="masks_bl"
#SBATCH --array=233
#143,148,153,174,176,202,204,207,208,230
#P1
#115,116,143,148,173,174,176,202,203,204,207,208,230-236,258,260-264,271-272,289,290,299,300,311,312,316,320-322,323,327,340-342,346-352,369,370,374,377-380,398,399,406,407,460,461,470,471,488,498-500,525-528,552-556,5    80-583,608-611,636-638,641,666%2
#88


GRIDS="${SLURM_ARRAY_TASK_ID}"

MASKS='cloud_shadowcast'

EXCLUDE='X'
SEN='LS2'
RES=10.0
RESETDB=True
BUFFER=3
RETRY=True
IMGDIR='dl'
PROJECT='biltong'
MAIN_DIR='/home/sandbox-cel'
BK_DIR='/home/downspout-cel'
GRID_FILE="/home/sandbox-cel/biltong/vector/biltong_grid_utm35S.gpkg"
#INDIR='${MAIN_DIR}/${PROJECT}/stac/grid/'
OUTDIR='tmp'
###############################
# DO NOT MODIFY BELOW THIS LINE
###############################

# activate the virtual environment
conda activate venv.tuyau

CONFIG_UPDATES="grids:[${GRIDS}] 
project_name:$PROJECT
main_path:${MAIN_DIR}/${PROJECT}/stac/grid 
backup_path:${BK_DIR}/${PROJECT}/stac/grid
scratch_dir:/home/scratch-cel/${PROJECT}
masking:img_dir:$IMGDIR
masking:imgout_dir:$OUTDIR
masking:method:$MASKS
masking:exclude:$EXCLUDE
masking:sat_sensors:$SEN
masking:buffer_px:$BUFFER
masking:retry_masks:$RETRY
status:reset_db:$RESETDB
dlMethod:stac
res:$RES
"

tuyau apply_masks --config-updates $CONFIG_UPDATES

conda deactivate
