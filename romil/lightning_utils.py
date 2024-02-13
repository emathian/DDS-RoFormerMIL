import os
from typing import Dict
import h5py
from pathlib import Path
import torch
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger


def get_multiclass_classif_results(
    preds: torch.Tensor, gts: torch.Tensor, n_classes: int, average: str
) -> Dict:
    """Use torchmetrics to compute binary classification metrics
    Args:
        preds (torch.Tensor) (n,c)
        gts (torch.Tensor) (n,)
        average: for now, torchmetrics support macro or weighted
    Returns:
        Dict: output results, including preds and gts tensors"""

    preds = preds.cpu()
    gts = gts.cpu()
    results = {
        "accuracy": torchmetrics.Accuracy(  # pylint: disable=not-callable
            "multiclass",
            num_classes=n_classes,
            average=average,
        )(preds, gts).item(),
        "recall": torchmetrics.Recall(  # pylint: disable=not-callable
            "multiclass", num_classes=n_classes, average=average
        )(preds, gts).item(),
        "precision": torchmetrics.Precision(  # pylint: disable=not-callable
            "multiclass", num_classes=n_classes, average=average
        )(preds, gts).item(),
        "f1": torchmetrics.F1Score(  # pylint: disable=not-callable
            "multiclass", num_classes=n_classes, average=average
        )(preds, gts).item(),
        "auc": torchmetrics.AUROC(  # pylint: disable=not-callable
            "multiclass",
            average=average,
            num_classes=n_classes,
        )(preds, gts).item(),
        "map": torchmetrics.AveragePrecision(  # pylint: disable=not-callable
            "multiclass",
            average=average,
            num_classes=n_classes,
        )(preds, gts).item(),
    }

    return results


def get_binary_classif_results(preds: torch.Tensor, gts: torch.Tensor) -> Dict:
    """Use torchmetrics to compute binary classification metrics
    Args:
        preds (torch.Tensor) (n,2) with logits or softmaxed
        gts (torch.Tensor) (n,)
    Returns:
        Dict: output results, including preds and gts tensors
    """

    preds = preds.cpu()
    gts = gts.cpu()
    results = {
        "accuracy": torchmetrics.Accuracy("binary")(  # pylint: disable=not-callable
            torch.softmax(preds, 1)[:, 1], gts
        ).item(),
        "recall": torchmetrics.Recall("binary")(  # pylint: disable=not-callable
            torch.softmax(preds, 1)[:, 1], gts
        ).item(),
        "precision": torchmetrics.Precision("binary")(  # pylint: disable=not-callable
            torch.softmax(preds, 1)[:, 1], gts
        ).item(),
        "f1": torchmetrics.F1Score("binary")(  # pylint: disable=not-callable
            torch.softmax(preds, 1)[:, 1], gts
        ).item(),
        "auc": torchmetrics.AUROC("binary")(  # pylint: disable=not-callable
            torch.softmax(preds, 1)[:, 1], gts
        ).item(),
        "map": torchmetrics.AveragePrecision("binary")(  # pylint: disable=not-callable
            torch.softmax(preds, 1)[:, 1], gts
        ).item(),
    }

    return results


def save_attention_matrix(coords, attention_scores, slide, eval_config):
    
    results_dir = Path(eval_config["results_dir"])
    output_path = os.path.join(results_dir, "attention_scores", slide+".h5")
    coords = coords.cpu().numpy()
    attention_scores = attention_scores.cpu().numpy()
    asset_dict = {   
        "coords": coords,
        "attention_scores": attention_scores,
            }
    
    save_hdf5(output_path, asset_dict, attr_dict=None, mode = "w")
    return -1


def save_hdf5(output_path, asset_dict, attr_dict=None, mode="a"):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(
                key,
                shape=data_shape,
                maxshape=maxshape,
                chunks=chunk_shape,
                dtype=data_type,
            )
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0] :] = val
    file.close()
    return output_path
class MLFlowLoggerCheckpointer(MLFlowLogger):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def after_save_checkpoint(self, model_checkpoint: ModelCheckpoint) -> None:
        """
        Called after model checkpoint callback saves a new checkpoint.
        """
        self.log_metrics(
            {
                f"best_{model_checkpoint.monitor}": model_checkpoint.best_model_score.item()
            }
        )
        self.experiment.log_artifact(
            self.run_id,
            model_checkpoint.best_model_path,
            self._prefix,
        )
        if os.path.exists(model_checkpoint.last_model_path):
            self.experiment.log_artifact(
                self.run_id,
                model_checkpoint.last_model_path,
                self._prefix,
            )
