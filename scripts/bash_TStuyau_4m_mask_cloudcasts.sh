#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 2 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o stac_bl_clouds.%N.%a.%j.out # STDOUT
#SBATCH -e stac_bl_clouds.%N.%a.%j.err # STDERR
#SBATCH --job-name="clouds_bl"
#SBATCH --array=115,116,143,148,173,174,176,202,203,204,207,208,230-236,258,260-264,271-272,289,290,299,300,311,312,316,320-322,323,327,340-342,346-352,369,370,374,377-380,398,399,406,407,460,461,470,471,488,498-500,525-528,552-556,580-583,608-611,636-638,641,666%10
#88

#87,144,175,205,206,288,292,293,319,324,408,427,472,318,375,376,291,317,343,344,345,371,372,373,313,314,315,201,229,328,524,145,146,147,149,171,172,489,432,433,404,405,89,117,88,115,116,143,148,173,174,176,202,203,204,207,208,230,231,232,233,234,235,236,258,260,261,262,263,264,271,272,289,290,299,300,311,312,316,320,321,322,323,327,340,341,342,346,347,348,349,350,351,352,369,370,374,377,378,379,380,398,399,406,407,460,461,470,471,488,498,499,500,525,526,527,528,552,553,554,555,556,580,581,582,583,608,609,610,611,636,637,638,641,666,404,405,432,433,489,171,172,145,147,149,524,328,201,229,313-315%20

GRIDS="${SLURM_ARRAY_TASK_ID}"

SEN='LS2'
RES=10.0
RESETDB=True
RETRY=True
IMG_DIR='dl'
PROJECT='biltong'
MAIN_DIR='/home/sandbox-cel'
BK_DIR='/home/downspout-cel'
GRID_FILE="/home/sandbox-cel/biltong/vector/biltong_grid_utm35S.gpkg"

CLOUDBUF=True
MINCLOUD=1
MINSHAD=2
GROWBAND='blue'
GROWMIN=1000
###############################
# DO NOT MODIFY BELOW THIS LINE
###############################

# activate the virtual environment
conda activate venv.tuyau

CONFIG_UPDATES="grids:[${GRIDS}] 
project_name:$PROJECT
main_path:${MAIN_DIR}/${PROJECT}/stac/grid 
backup_path:${BK_DIR}/${PROJECT}/stac/grid
masking:method:cloud_shadowcast
masking:img_dir:$IMG_DIR
masking:exclude:$EXCLUDE
masking:sat_sensors:$SEN
masking:retry_masks:$RETRY
masking:use_dilated:$CLOUDBUF
masking:min_cloud_conf:$MINCLOUD
masking:min_shadow_conf:$MINSHAD
masking:grow_band:$GROWBAND
masking:grow_min:$GROWMIN
status:reset_db:$RESETDB
dlMethod:stac
res:$RES
"

tuyau mask --config-updates $CONFIG_UPDATES

conda deactivate
