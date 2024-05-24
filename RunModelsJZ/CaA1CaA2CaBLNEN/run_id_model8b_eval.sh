#!/bin/bash
#SBATCH --job-name=RoformerCaA1CaA2CaB_biggest_B
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --error=/gpfsscratch/rech/ryr/ueu39kt/data_RoFormerMIL_LNEN_LCNEC_training_CaA1CaA2CaB_norm/RoformerCaA1CaA2CaB_biggest_B-%j.error
#SBATCH --out=/gpfsscratch/rech/ryr/ueu39kt/data_RoFormerMIL_LNEN_LCNEC_training_CaA1CaA2CaB_norm/RoformerCaA1CaA2CaB_biggest_B-%j.out
#SBATCH --account ryr@v100
module load pytorch-gpu/py3/2.0.0
ROOTDIR=/linkhome/rech/genkmw01/ueu39kt/DDS-RoFormerMIL
echo "PYTHONPATH=${PYTHONPATH}:${ROOTDIR}:${ROOTDIR}/CLAM" > .env
export PYTHONPATH=${PYTHONPATH}:${ROOTDIR}:${ROOTDIR}/CLAM

                    
 python /linkhome/rech/genkmw01/ueu39kt/DDS-RoFormerMIL/scripts/eval.py  --config-name eval_CaA1CaA2CaB_chillpinguin \
                          features_dir=/gpfsscratch/rech/ryr/ueu39kt/data_RoFormerMIL_LNEN_LCNEC_training_CaA1CaA2CaB_norm/dataRoFormer/features \
                          split_dir=/gpfsscratch/rech/ryr/ueu39kt/data_RoFormerMIL_LNEN_LCNEC_training_CaA1CaA2CaB_norm/dataRoFormer/split \
                          model_dict.BTRoPEAMIL_s_pixel.hidden_dim=128 \
                          model_dict.BTRoPEAMIL_s_pixel.positional_encoder.n_attention_block=6 \
                          model_dict.BTRoPEAMIL_s_pixel.mil_head.attention_net.attention_dim=64 
