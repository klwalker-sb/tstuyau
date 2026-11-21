#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 2 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o stac_bl_shadeagg.%N.%a.%j.out # STDOUT
#SBATCH -e stac_bl_shadeagg.%N.%a.%j.err # STDERR
#SBATCH --job-name="sgghade_bl"
##SBATCH --array=322
#115,116,147,175,203,231,258,260-264,271,272,315,321,343,349,371,377,398,399,404-408,427,433,461,500,528,552-556,638%10
#398,399,406,407,460,461,470,471,488,498,499,500,525,526,527,528,552,553,554,555,556,580,581,582,583,608,609,610,611,636,637,638,641,666,404,405,432,433,489,489,432,433,404,405%5
#87,144,175,205,206,288,292,293,319,324,408,427,472,318,375,376,291,317,343,344,345,371,372,373,313,314,315,201,229,328,524,145,146,147,149,171,172,489,432,433,404,405,89,117,88,115,116,143,148,173,174,176,202,203,204,207,208,230,231,232,233,234,235,236,258,260,261,262,263,264,271,272,289,290,299,300,311,312,316,320,321,322,323,327,340,341,342,346,347,348,349,350,351,352,369,370,374,377,378,379,380,398,399,406,407,460,461,470,471,488,498,499,500,525,526,527,528,552,553,554,555,556,580,581,582,583,608,609,610,611,636,637,638,641,666,404,405,432,433,489,171,172,145,147,149,524,328,201,229,313-315%20

#GRIDS="${SLURM_ARRAY_TASK_ID}"
GRIDS="/home/sandbox-cel/biltong/vector/cells_P1P2.csv"
START='2022-10-01'
END='2023-09-01'

SEN='LS2'
EXCLUDE='X'
RES=10.0
RESETDB=True
BUFFER=3
RETRY=True

PROJECT='biltong'
MAIN_DIR='/home/sandbox-cel'
BK_DIR='/home/downspout-cel'
GRID_FILE="/home/sandbox-cel/biltong/vector/biltong_grid_utm35S.gpkg"
###############################
# DO NOT MODIFY BELOW THIS LINE
###############################

# activate the virtual environment
conda activate venv.tuyau

CONFIG_UPDATES="grids:${GRIDS} 
project_name:$PROJECT
main_path:${MAIN_DIR}/${PROJECT}/stac/grid 
backup_path:${BK_DIR}/${PROJECT}/stac/grid
masking:method:terrain_shade
masking:aggregate:True
reconstruct:start:$START
reconstruct:end:$END
masking:exclude:$EXCLUDE
masking:sat_sensors:$SEN
masking:buffer_px:$BUFFER
masking:retry_masks:$RETRY
status:reset_db:$RESETDB
dlMethod:stac
res:$RES
"

tuyau mask --config-updates $CONFIG_UPDATES

conda deactivate
