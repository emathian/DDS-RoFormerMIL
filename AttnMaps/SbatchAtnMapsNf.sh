#!/bin/bash
#SBATCH --job-name=NfAttnMaps
#SBATCH --output=NfAttnMaps-%j.out
#SBATCH --error=NfAttnMaps-%j.error
#SBATCH --partition=high_p
#SBATCH --account=gcs
nextflow run  AttnMapsNf.nf --resume