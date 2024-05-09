#!/bin/bash
#SBATCH --job-name=RoformerA1A2
#SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --error=/gpfsscratch/rech/ryr/ueu39kt/data_RoFormerMIL_LNEN_LCNEC_training_CaA1CaA2_norm/RoformerA1A2-%j.error
#SBATCH --out=/gpfsscratch/rech/ryr/ueu39kt/data_RoFormerMIL_LNEN_LCNEC_training_CaA1CaA2_norm/RoformerA1A2-%j.out
#SBATCH --account ryr@v100
module load pytorch-gpu/py3/2.0.0
echo "PYTHONPATH=${PYTHONPATH}:${PWD}:${PWD}/CLAM" > .env
export PYTHONPATH=${PYTHONPATH}:${PWD}:${PWD}/CLAM

python ../../scripts/train.py --config-name training_LNEN_LCNEC_CaA1_CaA2_JZ   features_dir=/gpfsscratch/rech/ryr/ueu39kt/data_RoFormerMIL_LNEN_LCNEC_training_CaA1CaA2_norm/dataRoFormer/features split_dir=/gpfsscratch/rech/ryr/ueu39kt/data_RoFormerMIL_LNEN_LCNEC_training_CaA1CaA2_norm/dataRoFormer/split