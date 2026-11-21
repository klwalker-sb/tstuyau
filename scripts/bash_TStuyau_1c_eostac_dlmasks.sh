#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH -t 0-20:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o stac_bl_dlclouds_bl.%N.%a.%j.out # STDOUT
#SBATCH -e stac_bl_dlclouds.%N.%a.%j.err # STDERR
#SBATCH --job-name="clouddl"
#SBATCH --array=115,116,143,148,173,174,176,202,203,204,207,208,230-236,258,260-264,271-272,289,290,299,300,311,312,316,320-322,323,327,340-342,346-352,369,370,374,377-380,398,399,406,407,460,461,470,471,488,498-500,525-528,552-556,580-583,608-611,636-638,641,666%2
#88

#P1 cells:
#88,115,116,143,148,173,174,176,202,203,204,207,208,230-236,258,260-264,271-272,289,290,299,300,311,312,316,320-322,323,327,340-342,346-352,369,370,374,377-380,398,399,406,407,460,461,470,471,488,498-500,525-528,552-556,580-583,608-611,636-638,641,666%2
##############################################
### As an array job:
GRID_ID=$SLURM_ARRAY_TASK_ID
echo cell_id = $GRID_ID >&2
# Set permissions on output files
umask 002

##Settables:
PROJECT_HOME="/home/sandbox-cel/biltong/stac"
LANDSAT_DIR="${PROJECT_HOME}/grid/000${GRID_ID}/landsat"
SENTINEL_DIR="${PROJECT_HOME}/grid/000${GRID_ID}/sentinel2"
FILETYPE='.tif'
OUT_DIR="${PROJECT_HOME}/grid/000${GRID_ID}"
SENSORS='S2'
L7STOPYR=2017
GRID_FILE="/home/sandbox-cel/biltong/vector/biltong_grid_utm35S.gpkg"
EPSG=32735
BUFFER=100
BUFFER_EX=11000
MASK_SFX='_maskbig'
MASKDIR='clouds/native'
RES=10.0
MASK_RES=0.0
PROJ='biltong_1'
START_YEAR=2016
END_YEAR=2026

################################################
### activate the virtual environment

conda activate venv.lucinla38_dlv2

YEAR=$START_YEAR
while [ "$YEAR" -lt "$END_YEAR" ]; do
    EYEAR=$(($YEAR+1))

    START_DATE="${YEAR}-01-01"
    END_DATE="${EYEAR}-01-01"
    
    echo -e \\ Working on ${YEAR} to ${EYEAR}>&2
    TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
    echo $TIMESTAMP >&2

    eostac download --get-masks --not-masked --resolution $RES --mask-resolution $MASK_RES --mask-dir $MASKDIR --mask-suffix $MASK_SFX --buffer-extra $BUFFER_EX --start-date $START_DATE --end-date $END_DATE --proj-ver $PROJ --bounds $GRID_FILE --bounds-query UNQ==$GRID_ID --out-path $OUT_DIR --epsg $EPSG --bounds-buffer $BUFFER --sensors $SENSORS --l7-stop_year $L7STOPYR --max-items -1 -w 4 -t 2

    YEAR=$EYEAR    
done        
    
       

conda deactivate
