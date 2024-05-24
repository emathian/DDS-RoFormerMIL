#!/bin/bash
#SBATCH --job-name=RunAllCaA1CaA2CaBModel
#SBATCH --array=1-10
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --error=RunAllCaA1CaA2CaBModel-%A_%a.error
#SBATCH --out=RunAllCaA1CaA2CaBModel-%A_%a.out
#SBATCH --account ryr@cpu



# Construct the filename using SLURM_TASK_ID
TASKID=$SLURM_ARRAY_TASK_ID
script_file="run_id_model_${TASKID}.sh"
echo $script_file
# Execute the script
sbatch "${script_file}"