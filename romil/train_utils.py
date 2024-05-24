import logging
import os
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from CLAM.datasets.dataset_generic import Generic_MIL_Dataset
from romil.lightning_datamodule import MILDatamodule
import sys
sys.path.append("romil")
from pytorch_lightning.loggers import WandbLogger
from torch import nn
log = logging.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
           
            
def xavier_init(model):
    for name, param in model.named_parameters():
        if "weight" in name:  # Initialize weights only, not biases
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.1)
        elif "bias" in name:
            nn.init.constant_(param, 0.0)
            
            
def train(
    dataset: Generic_MIL_Dataset, fold: int, args: DictConfig, split_csv_filename: Path, results_dir: str, 
) -> Dict[str, float]:
    """
    train for a single fold
    """
    print("\nTraining Fold {}!".format(fold))

    if (
        args["training_args"]["lightning_module"]["model"]["_target_"]
        == "models.model_bpa.RPEDSMIL"
    ):
        log.info(
            "DSMIL currently doesn't support batch size >1. Setting"
            " accumulate_grad_batches instead"
        )
        OmegaConf.update(
            args,
            "training_args.trainer.accumulate_grad_batches",
            args["training_args"]["datamodule_params"]["batch_size"],
        )
        OmegaConf.update(args, "training_args.datamodule_params.batch_size", 1)

    datamodule = MILDatamodule(
        dataset, split_csv_filename, **args["training_args"]["datamodule_params"]
    )

    model = hydra.utils.instantiate(args["training_args"]["lightning_module"],
                                    fold=fold, results_dir=results_dir)
    xavier_init(model)
    callbacks = [
        hydra.utils.instantiate(callback_cfg)
        for _, callback_cfg in args["training_args"]["callbacks"].items()
    ]
    ## TEST WNADB 
    #wandb_logger = WandbLogger(project="CaA1CaA2_v3", log_model="all")

    trainer = hydra.utils.instantiate(
        args["training_args"]["trainer"], callbacks=callbacks,
        gradient_clip_val=1,
        gradient_clip_algorithm="value",
        #logger =  wandb_logger,
        #detect_anomaly=True
        )
    
    #wandb_logger.watch(model, log_freq=20)
    # art = wandb.Artifact('model-test-debug', type='model')
    # wandb.log_artifact(art)
    
    if fold == 0:
        trainer.logger.log_hyperparams(args["training_args"])
        
    # MUTE MLFLOW LOGGER   
    trainer.logger.experiment.log_artifact(
        trainer.logger.run_id, split_csv_filename, f"fold_{fold}"
    )

    trainer.fit(model=model, datamodule=datamodule)
 
    return trainer.test(ckpt_path="best", datamodule=datamodule)[0]
    ## Warning test on last epoch
    #return trainer.test(ckpt_path="last", datamodule=datamodule)[0]


def seed_torch(seed=7):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
