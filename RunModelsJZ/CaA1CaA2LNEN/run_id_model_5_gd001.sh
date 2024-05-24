#!/bin/bash
#SBATCH --job-name=RoformerA1A2_smallest_hig_reg
#SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --error=/gpfsscratch/rech/ryr/ueu39kt/data_RoFormerMIL_LNEN_LCNEC_training_CaA1CaA2_norm/RoformerA1A2_smallest_hig_reg_gd01-%j.error
#SBATCH --out=/gpfsscratch/rech/ryr/ueu39kt/data_RoFormerMIL_LNEN_LCNEC_training_CaA1CaA2_norm/RoformerA1A2_smallest_hig_reg_gd01-%j.out
#SBATCH --account ryr@v100
module load pytorch-gpu/py3/2.0.0
ROOTDIR=/linkhome/rech/genkmw01/ueu39kt/DDS-RoFormerMIL
echo "PYTHONPATH=${PYTHONPATH}:${ROOTDIR}:${ROOTDIR}/CLAM" > .env
export PYTHONPATH=${PYTHONPATH}:${ROOTDIR}:${ROOTDIR}/CLAM

python /linkhome/rech/genkmw01/ueu39kt/DDS-RoFormerMIL/scripts/train.py  --config-name training_LNEN_LCNEC_CaA1_CaA2_JZ \
                    features_dir=/gpfsscratch/rech/ryr/ueu39kt/data_RoFormerMIL_LNEN_LCNEC_training_CaA1CaA2_norm/dataRoFormer/features \
                    split_dir=/gpfsscratch/rech/ryr/ueu39kt/data_RoFormerMIL_LNEN_LCNEC_training_CaA1CaA2_norm/dataRoFormer/split \
                    model_dict.BTRoPEAMIL_s_pixel.positional_encoder.n_attention_block=4 \
                    model_dict.BTRoPEAMIL_s_pixel.mil_head.attention_net.attention_dim=16 \
                    model_dict.BTRoPEAMIL_s_pixel.positional_encoder.dropout=0.35 \
                    model_dict.BTRoPEAMIL_s_pixel.positional_encoder.resid_dropout=0.35 \
                    model_dict.BTRoPEAMIL_s_pixel.mil_head.dropout=0.35 \
                    training_args.lightning_module.optimizer.weight_decay=0.001 \
                    training_args.trainer.gradient_clip_val=0.01 \
                    k_folds.k_start=15 \
                    k_folds.k_end=20