#!/bin/bash
#SBATCH --job-name=AttnMap
#SBATCH --output=AttnMap-%A_%a.out
#SBATCH --error=AttnMap-%A_%a.error
#SBATCH --mem=5G
#SBATCH --partition=high_p
#SBATCH --account=lungNENomics
#SBATCH --array=0-193  # Adjust this range based on the number of directories

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HaloAE
python AttnScoresPlotCaA1CaA2CaB.py --job_id $SLURM_ARRAY_TASK_ID 