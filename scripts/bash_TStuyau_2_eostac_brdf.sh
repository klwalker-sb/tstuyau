#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH -t 4-00:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o TStuyau_brdf.%N.%a.%j.out # STDOUT
#SBATCH -e TStuyau_brdf.%N.%a.%j.err # STDERR
#SBATCH --job-name="brdf"
#SBATCH --array=0-20%4
###################################################################
### Modify --array above with grid cell numbers to run as array job.

GRID_ID=$SLURM_ARRAY_TASK_ID
### note: if grid cell > 999, enter last three digits in array and use
### GRID_ID=$(($SLURM_ARRAY_TASK_ID + 2000))

#Set permissions on output files
umask 002

###################################################################
### Settables:

COEFFS="${PROJECT_DIR}/brdf_coeffs/coefficients"
BUFFER=100
START_YEAR=2010
END_YEAR=2020

###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel/"
PROJECT="paraguay_lc"
EPSG=8858

PROJECT_DIR="${MAIN_DIR}/${PROJECT}"
PROJECT_PATH="${PROJECT_DIR}/stac/grid/00${GRID_ID}/"
OUT_PATH="${PROJECT_DIR}/stac/grid/00${GRID_ID}/brdf"
GRID_FILE="${PROJECT_DIR}/project_grid_{EPSG}.gpkg"

###################################################################
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"
### activate the virtual environment
conda activate venv.tstuyau_dl

###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

TIMESTAMP0=`date "+%Y-%m-%d %H:%M:%S"`

y=$START_YEAR
while [ $y -ne $END_YEAR ]
do	
	START_DATE="${y}-1-01"
	END_DATE="${y}-12-31"
	echo  Working on $START_DATE to $END_DATE >&2 
	TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
	echo $TIMESTAMP >&2

	eostac brdf --start-date $START_DATE --end-date $END_DATE --project-path $PROJECT_PATH --out-path $OUT_PATH --threads 8 --apply-bandpass --coeffs-path $COEFFS

	y=$(($y+1))
done

TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
TIMETOT=$(($(date -d "$TIMESTAMP" "+%s") - $(date -d "$TIMESTAMP0" "+%s") ))
echo done at $TIMESTAMP >&2
echo full process took: $($TIMETOT/60) minutes >&2

conda deactivate
