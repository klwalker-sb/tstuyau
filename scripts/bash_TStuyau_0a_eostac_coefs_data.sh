#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH -t 4-00:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o TStuyau_coefs.%N.%a.%j.out # STDOUT
#SBATCH -e TStuyau_coefs.%N.%a.%j.err # STDERR
#SBATCH --job-name="TStuyau_coefs"
###################################################################
### Set permissions on output files
umask 002

###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel/
PROJECT="paraguay_lc"
OUT_PATH="${MAIN_DIR}/${PROJECT}/brdf_coeffs"
EPSG=8858

###################################################################
### Settables:

NSAMPS=50
BOUNDARY_FILE="${MAIN_DIR}/${PROJECT}/vector/AOI_main.zip"
START="2017-01-01"
END="2024-01-14"

###################################################################
###################################################################
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"
conda activate venv.tstuyau_dl

eostac adjust --start-date $START --end-date $END --out-path $OUT_PATH --geometry $BOUNDARY_FILE --n-samples $NSAMPS --resolution 30.0 --freq Y --freq-repeat 10 --workers 4 --threads 2 --epsg $EPSG

conda deactivate
