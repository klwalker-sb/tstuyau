#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH -t 0-20:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o TStuyau_dl.%N.%a.%j.out # STDOUT
#SBATCH -e TStuyau_dl.%N.%a.%j.err # STDERR
#SBATCH --job-name="TStuyau_dl"
#SBATCH --array=1-50%2
###################################################################
### Modify --array above with grid cell numbers to run as array job.

GRID_ID=$SLURM_ARRAY_TASK_ID
### note: if grid cell > 999, enter last three digits in array and use
### GRID_ID=$(($SLURM_ARRAY_TASK_ID + 2000))

## Set permissions on output files
umask 002

###################################################################
### Settables:

FILETYPE='.tif'
L7STOPYR=2017
BUFFER=100
START_YEAR=2019
END_YEAR=2020	

###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel/"
PROJECT="paraguay_lc"
EPSG=8858

PROJECT_DIR="${MAIN_DIR}/${PROJECT}/stac"
LANDSAT_DIR="${PROJECT_DIR}/grid/000${GRID_ID}/landsat"
SENTINEL_DIR="${PROJECT_DIR}/grid/000${GRID_ID}/sentinel2"
OUT_DIR="${PROJECT_DIR}/grid/000${GRID_ID}"
GRID_FILE="${PROJECT_DIR}/project_grid_${EPSG}.gpkg"

###################################################################
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"
### activate the virtual environment
conda activate venv.tstuyau_dl

###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

echo cell_id = $GRID_ID >&2

## if directories are not empty, run script to check for corrupt files
if [ -n "$LANDSAT_DIR" ]
then	
  eostac check --out-path $LANDSAT_DIR --file-type $FILETYPE
fi

if [ -n "$SENTINEL_DIR" ]
then
  eostac check --out-path $SENTINEL_DIR --file-type $FILETYPE
fi

TIMESTAMP0=`date "+%Y-%m-%d %H:%M:%S"`

YEAR=$START_YEAR
while [ $YEAR -ne $END_YEAR ]
do
	for m in {1..11}
	do
		CURRENT_MONTH=$(printf "%02d" $m)
		NEXT_ITER=$(($m+1))
		NEXT_MONTH=$(printf "%02d" $NEXT_ITER)
		START_DATE="${YEAR}-${CURRENT_MONTH}-01"
                
                if [[ $m -eq 11 ]]
                then
                  END_DATE="${YEAR}-${NEXT_MONTH}-31"
                else
		  END_DATE="${YEAR}-${NEXT_MONTH}-01"
                fi

		echo -e \\ Working on ${START_DATE} to ${END_DATE}>&2	
		TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
		echo $TIMESTAMP >&2

		eostac download --start-date $START_DATE --end-date $END_DATE --bounds $GRID_FILE --bounds-query UNQ==$GRID_ID --out-path $OUT_DIR --epsg $EPSG --bounds-buffer $BUFFER --l7-stop_year $L7STOPYR --max-items -1 -w 4 -t 2
	
	done
	YEAR=$(($YEAR+1))
done

TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
TIMETOT=$(($(date -d "$TIMESTAMP" "+%s") - $(date -d "$TIMESTAMP0" "+%s") ))
echo Done at $TIMESTAMP >&2
echo full process took: $(($TIMETOT/60)) minutes >&2
echo core used: $SLURM_NTASKS >&2
conda deactivate
