# pylint: skip-file
# Inspired by https://github.com/gokul-pv/lightning-hydra-timm
import logging
import re
from typing import Any, Dict, List, Tuple
import einops
import torch
import os
import pandas as pd
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    ConfusionMatrix,
    MaxMetric,
    MeanMetric,
    MetricCollection,
    Precision,
    Recall,
    F1Score,

)

from CLAM.utils import instance_eval_utils
from CLAM.utils.pretty_cm import get_cm_image
from romil.models import RoMIL
from romil import lightning_utils
import h5py 
console_log = logging.getLogger("Lightning training")


class MILLitModule(LightningModule):
    """Wrapper around model_mil class
    doesn't change the execution of the model

    mil_mc not tested yet
    """

    def __init__(
        self,
        model: nn.Module,
        loss: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        use_instance_loss: bool,
        bag_loss_weight: float,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=False,
        lr_scheduler=None,
        k_sample=8,
        attn_scores=False,
        fold=0,
        results_dir="tmp_res",
    ) -> None:
        super().__init__()
        
       
        self.use_instance_loss = use_instance_loss
        if self.use_instance_loss:
            self.bag_loss_weight = bag_loss_weight
        else:
            self.bag_loss_weight = 1

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["model"])

        # model should also have model.n_classes
        # and optionally instance_classifiers
        # model.forward outputs a dict with logits/attention_scores/updated_features.
        # RPEDSMIL outputs also include a patches_predictions key

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # loss function
        self.criterion = loss
        metrics_params = {"task": "binary"}

        if model.n_classes > 2:
            metrics_params = {"task": "multiclass", "num_classes": model.n_classes}

        # metric objects for calculating and averaging across batches

        metrics = MetricCollection(
            [
                Accuracy(**metrics_params),
                Recall(**metrics_params),
                Precision(**metrics_params),
                AUROC(**metrics_params),
                AveragePrecision(**metrics_params),
                F1Score(**metrics_params, average='weighted')
            ]
        )

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # CM:
        self.train_cm = ConfusionMatrix(**metrics_params)
        self.val_cm = ConfusionMatrix(**metrics_params)
        self.test_cm = ConfusionMatrix(**metrics_params)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.subtyping = subtyping

        # for averaging loss across batches
        self.train_instance_loss = MeanMetric()
        self.val_instance_loss = MeanMetric()
        self.test_instance_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        
        # get attention scores 
        self.attn_scores = attn_scores
        self.test_process = False
        
        # For saving attention maps at the test time
        self.fold = fold
        self.results_dir = results_dir

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def forward(
        self, features: List[torch.Tensor], coords: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features (List[torch.Tensor]): List of features tensors (B elements of size Nb*D)
            coords (List[torch.Tensor]): List of features coordss (B elements of size Nb*2)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                logits (b, n_classes)
                attention_scores(b, n, n_classes)
                updated_features (b, n, d)
                patch_preds for DSMIL (1, n, n_classes)
        """
        return self.model(features, coords, self.test_process)

    def step(
        self, batch: Any,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        features (List[torch.Tensor]): List of features tensors (B elements of size Nb*D)
        coords (List[torch.Tensor]): List of features coordss (B elements of size Nb*2)
        """
        features, labels, coords, slide_id = batch

        assert not (
            isinstance(self.model, RoMIL.RoPEDSMIL) and len(features) > 1
        ), "DSMIL only implemented for batch size 1"
        
        outputs = self.forward(features, coords)
        
        # Bag-level logits (b,n_classes), attention_scores (b, n_classes, n_patches) updated_features (b, n_patches, D)
        # Updated features are features that went through linear projections before attention
        logits, attention_scores, updated_features = (
            outputs["logits"],
            outputs["attention_scores"],
            outputs["updated_features"],
        )

        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        bag_loss = self.criterion(logits, labels)

        if isinstance(self.model, RoMIL.RoPEDSMIL):
            patch_loss = self.criterion(
                outputs["patches_predictions"].max(1)[0], labels
            )
            bag_loss = 0.5 * bag_loss + 0.5 * patch_loss

        if not self.use_instance_loss or (
            isinstance(self.model, RoMIL.RoPEAMIL)
            or isinstance(self.model, RoMIL.RoPEDSMIL)
        ):
            instance_loss = torch.zeros_like(bag_loss)
        else:
            instance_loss = instance_eval_utils.instance_loss(
                updated_features,
                attention_scores,
                labels,
                self.model.instance_classifiers,
                self.model.n_classes,
                self.subtyping,
                self.k_sample,
                self.instance_loss_fn,
            )

        loss = (
            self.bag_loss_weight * bag_loss + (1 - self.bag_loss_weight) * instance_loss
        )
        if self.attn_scores:
            # If predict_step or test_step is called
            # Warning : atten_scores are returned only if eval.py is called. In this case ClassAttention matches with inneficient_attention_class,
            # ie. quadratic computation of the self attention. Attention scores are returned within the evaluate function of eval_utils.oy script.
            # To activate the stoarage of attention scores please set attn_scores : True in model_dict.yaml > RoPEAMIL_pixel > mil_head > attention_net  
            # Automatic activation while test step
            return loss, instance_loss, preds, labels, probs, attention_scores, coords, slide_id
        else:
            return loss, instance_loss, preds, labels, probs

    def training_step(self, batch: Any, _: int) -> Dict[str, Any]:
        loss, instance_loss, preds, targets, probas = self.step(batch)
        self.train_log(loss, instance_loss, preds, targets, probas)

        return {"loss": loss, "preds": preds, "targets": targets}

    def train_log(
        self,
        loss: torch.Tensor,
        instance_loss: torch.Tensor,
        preds: torch.Tensor,
        targets: torch.Tensor,
        probas: torch.Tensor,
    ) -> Dict[str, Any]:
        # update and log metrics
        self.train_loss(loss)
        if self.model.n_classes == 2:
            self.train_metrics(probas[..., 1], targets)

        else:
            self.train_metrics(probas, targets)

        self.log_dict(
            self.train_metrics,
            logger=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(targets),
            sync_dist=True,
        )

        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(targets),
            sync_dist=True,
        )

        self.train_cm(preds, targets)

        if self.use_instance_loss:
            self.train_instance_loss(instance_loss)
            self.log(
                "train/instance_loss",
                self.train_instance_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, _: int) -> Dict[str, Any]:     
        self.model.eval()
        loss, instance_loss, preds, targets, probas = self.step(batch)
        ## Activation of attn scores 
        # if self.current_epoch == 10:
        #     # Debug export all validation res at epoch == 10
        #     self.test_on()
        #     loss, instance_loss, preds, targets, probas, attention_scores, coords, slide_id = self.step(batch)
        #     self.val_export_all_res(preds, targets, probas, attention_scores, coords, slide_id)
        #     self.test_off()   
        self.val_log(loss, instance_loss, preds, targets, probas)
        self.model.train()
        return {"loss": loss, "preds": preds, "targets": targets}

    def val_export_all_res(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor, 
        probas: torch.Tensor,
        attention_scores: torch.Tensor, 
        coords: torch.Tensor,
        slide_id: list, 
        )-> None :
        
        print("slide_id!" )
        print(slide_id)
        print("attention_scores size!")
        print(attention_scores.size())
        attention_scores = einops.rearrange(torch.squeeze(attention_scores), "c n h -> n c h")
        #attention_scores = einops.rearrange(torch.squeeze(attention_scores), "c n -> n c ")
        output_attention_scores_dir = os.path.join(self.results_dir, "val_attention_scores", str(self.fold))
        os.makedirs(output_attention_scores_dir, exist_ok=True)
        lightning_utils.save_attention_matrix(coords[0], attention_scores,
                                          slide_id[0], output_attention_scores_dir )
      
        output_pred_dir = os.path.join(self.results_dir, "val_patients_preds.parquet", str(self.fold))
        os.makedirs(output_pred_dir, exist_ok=True)

        df_predictions = pd.DataFrame({"slide_id":[slide_id[0]],
                      "preds":[preds[0].cpu().numpy().tolist()],
                      "labels":[targets[0].cpu().numpy().tolist()],
                      "probas":[probas[0].cpu().numpy().tolist()]})
        df_predictions["fold"]  = self.fold
        df_predictions["split"]  = "val"
        df_predictions.to_parquet(output_pred_dir, 
                                   partition_cols=["fold", "split"],
                                   )
    
    def val_log(
        self,
        loss: torch.Tensor,
        instance_loss: torch.Tensor,
        preds: torch.Tensor,
        targets: torch.Tensor,
        probas: torch.Tensor,
    ) -> Dict[str, Any]:
        # update and log metrics
        self.val_loss(loss)
        if self.model.n_classes == 2:
            self.val_metrics(probas[..., 1], targets)
        else:
            self.val_metrics(probas, targets)

        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(targets),
            sync_dist=True,
        )

        self.log_dict(
            self.val_metrics,
            logger=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(targets),
            sync_dist=True,
        )

        self.val_cm(preds, targets)

        if self.use_instance_loss:
            self.val_instance_loss(instance_loss)
            self.log(
                "val/instance_loss",
                self.val_instance_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=len(targets),
                sync_dist=True,
            )
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, _: int) -> Dict[str, Any]:
        self.test_on()
        self.model.eval()
        loss, instance_loss, preds, targets, probas, attention_scores, coords, slide_id = self.step(batch)
        self.test_log(loss, instance_loss, preds, targets, probas)
        attention_scores = einops.rearrange(torch.squeeze(attention_scores), "c n h -> n c h")
        #attention_scores = einops.rearrange(torch.squeeze(attention_scores), "c n -> n c ")
        output_attention_scores_dir = os.path.join(self.results_dir, "attention_scores", str(self.fold))
        os.makedirs(output_attention_scores_dir, exist_ok=True)
        lightning_utils.save_attention_matrix(coords[0], attention_scores,
                                          slide_id[0], output_attention_scores_dir )
      
        output_pred_dir = os.path.join(self.results_dir, "patients_preds.parquet", str(self.fold))
        os.makedirs(output_pred_dir, exist_ok=True)

        df_predictions = pd.DataFrame({"slide_id":[slide_id[0]],
                      "preds":[preds[0].cpu().numpy().tolist()],
                      "labels":[targets[0].cpu().numpy().tolist()],
                      "probas":[probas[0].cpu().numpy().tolist()]})
        df_predictions["fold"]  = self.fold
        df_predictions["split"]  = "test"
        df_predictions.to_parquet(output_pred_dir, 
                                   partition_cols=["fold", "split"],
                                   )
        self.model.train()
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_on(self) -> None :
        self.test_process = True
        self.attn_scores = True
        
    def test_off(self) -> None :
        self.test_process = False
        self.attn_scores = False
        
        
    def test_log(
        self,
        loss: torch.Tensor,
        instance_loss: torch.Tensor,
        preds: torch.Tensor,
        targets: torch.Tensor,
        probas: torch.Tensor,
    ) -> Dict[str, Any]:
        # update and log metrics
        self.test_loss(loss)
        if self.model.n_classes == 2:
            self.test_metrics(probas[..., 1], targets)

        else:
            self.test_metrics(probas, targets)

        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log_dict(
            self.test_metrics,
            logger=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(targets),
            sync_dist=True,
        )

        self.test_cm(preds, targets)

        if self.use_instance_loss:
            self.test_instance_loss(instance_loss)
            self.log(
                "test/instance_loss",
                self.test_instance_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=len(targets),
                sync_dist=True,
            )
        return {"loss": loss, "preds": preds, "targets": targets}

    def predict_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        loss, instance_loss, preds, targets, probas, attention_scores, coords, slide_id = self.step(batch)
        return {"loss": loss, "preds": preds, "targets": targets, "logits": probas, "attention_scores": attention_scores, "coords": coords}

    def on_validation_epoch_end(
        self,
    ) -> None:
        if self.model.n_classes  <=2:
            acc = self.val_metrics.compute()["val_BinaryAccuracy"]  # get current val acc
        else:
            acc =  self.val_metrics.compute()["val_MulticlassAccuracy"]  
        if not self.trainer.sanity_checking:
            self.val_acc_best(acc)  # update best so far val acc
            # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            self.log(
                "val/acc_best",
                self.val_acc_best.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            # MUTE MLFLOW LOGGER
            if self.trainer.is_global_zero:
                console_log.info(
                    (
                        "Fold %s | Epoch %s|"
                        f" {' | '.join([f'{key}: {value.item()}' for key, value in self.trainer.callback_metrics.items()])} "
                    ),
                    # MUTE MLFLOW LOGGER
                    re.findall(r"\d+", self.logger._prefix)[
                        -1
                    ],  # Find current validation fold using the logger prefix (extract the last number in the string)
                    self.current_epoch,
                )

            if "ddp" not in self.trainer.strategy.__str__():
                self.log_cm(self.val_cm, "cm_val.png")

    def on_train_epoch_end(
        self,
    ) -> None:
        # For some reason, DDP training stops there if we try to compute the confusion matrix
        # Related issue opened in github #34
        if "ddp" not in self.trainer.strategy.__str__():
            self.log_cm(self.train_cm, "cm_train.png")

    def on_test_epoch_end(
        self,
    ) -> None:
        self.log_cm(self.test_cm, "cm_test.png")

    def log_cm(self, cm_metric, name):
        cm = cm_metric.compute().cpu().numpy().astype(int)
        cm_metric.reset()
        if self.model.n_classes  <=2:
            cm_img = get_cm_image(cm, [0, 1])
        else:
            cm_img = get_cm_image(cm, [0, 1, 2])
        self.logger.experiment.log_image(
            self.logger.run_id, cm_img, f"{self.logger._prefix}/{name}"
        )

    def configure_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.lr_scheduler is None:
            return {"optimizer": optimizer}
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": self.lr_scheduler["scheduler"](optimizer=optimizer),
                "monitor": self.lr_scheduler.get("monitor"),
            }
