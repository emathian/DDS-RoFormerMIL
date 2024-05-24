#!/bin/bash
#SBATCH --job-name=createSplit
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --error=createSplit-%j.error
#SBATCH --out=createSplit-%j.out
#SBATCH --account ryr@v100
module load pytorch-gpu/py3/2.0.0
ROOTDIR=/linkhome/rech/genkmw01/ueu39kt/DDS-RoFormerMIL
echo "PYTHONPATH=${PYTHONPATH}:${ROOTDIR}:${ROOTDIR}/CLAM" > .env
export PYTHONPATH=${PYTHONPATH}:${ROOTDIR}:${ROOTDIR}/CLAM


python /linkhome/rech/genkmw01/ueu39kt/DDS-RoFormerMIL/scripts/new_create_splits_seq.py  --config-name create_splits  features_dir=/gpfsscratch/rech/ryr/ueu39kt/data_RoFormerMIL_LNEN_LCNEC_training_CaA1CaA2LCNEC_norm/dataRoFormer/features split_dir=/gpfsscratch/rech/ryr/ueu39kt/data_RoFormerMIL_LNEN_LCNEC_training_CaA1CaA2LCNEC_norm/dataRoFormer/split
