from typing import Any, Dict, Optional, Tuple

from lightning import LightningModule
import torch
import torch.nn as nn
from torchmetrics import MeanMetric, MinMetric

from models.components.hr_net import HRNetLandmarks

class WingLoss(nn.Module):
    def __init__(self, w: float =0.039, epsilon: float = 0.0078):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * torch.log(torch.tensor(1.0 + w / epsilon))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        diff = (pred - target).abs()
        C = self.C.to(diff.device)
        loss_per_coord = torch.where(
            diff < self.w,
            self.w * torch.log(1.0 + diff / self.epsilon),
            diff - C,
        )
        
        if mask is None:
            return loss_per_coord.sum(dim=-1).mean()

        visible = mask.float()
        n_visible = visible.sum().clamp(min=1.0)

        return (loss_per_coord.sum(dim=-1)*visible).sum() / n_visible
    
class FacialLandmarkModule(LightningModule):

    def __init__(
        self,
        net,
        num_landmarks: int = 76,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss: str = "mse",
        wing_w: float = 10.0,
        wing_epsilon: float = 2.0,
        nme_norm_factor: Optional[float] = None,
        freeze_backbone: int = 0,
        compile: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = net

        if loss == "wing":
            self.criterion = WingLoss(w=wing_w, epsilon=wing_epsilon)
        elif loss == "mse":
            self.criterion = nn.MSELoss()
        elif loss == "smooth_l1":
            self.criterion = nn.SmoothL1Loss()

        self.norm_factor = nme_norm_factor or (2 ** 0.5)

        self.train_loss = MeanMetric()
        self.val_loss   = MeanMetric()
        self.test_loss  = MeanMetric()

        self.train_nme  = MeanMetric()
        self.val_nme    = MeanMetric()
        self.test_nme   = MeanMetric()

        self.val_nme_best = MinMetric()

    def _compute_nme(self, pred_heatmap: torch.Tensor, target_landmark: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred, _ = self.decode_heatmaps(pred_heatmap)

        dist = torch.norm(pred - target_landmark, dim=-1)  

        if mask is None:
            nme = dist.mean(dim=-1) / self.norm_factor
            return nme

        visible = mask.float()
        n_visible = visible.sum(dim=-1).clamp(min=1.0)

        nme = (dist*visible).sum(dim=-1) / n_visible / self.norm_factor

        return nme
    
    def on_train_start(self):
        self.val_loss.reset()
        self.val_nme.reset()
        self.val_nme_best.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


    def model_step(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, heatmaps, landmarks, mask = batch['image'], batch['heatmap'], batch['keypoint_rel'], batch['mask']

        if landmarks.dim() == 2:
            landmarks = landmarks.view(-1, self.hparams.num_landmarks, 2)  
        if mask.dim() == 1:
            mask = mask.view(-1, self.hparams.num_landmarks)

        pred = self.net(images)
        loss = self.criterion(pred, heatmaps)
        nme  = self._compute_nme(pred, landmarks, mask)
        return loss, nme, pred

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, nme, _ = self.model_step(batch)
        self.train_loss(loss)
        self.train_nme(nme.mean())
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/nme",  self.train_nme,  on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss, nme, _ = self.model_step(batch)
        self.val_loss(loss)
        self.val_nme(nme.mean())
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/nme",  self.val_nme,  on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        nme = self.val_nme.compute()
        self.val_nme_best(nme)
        self.log("val/nme_best", self.val_nme_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        loss, nme, _ = self.model_step(batch)
        self.test_loss(loss)
        self.test_nme(nme.mean())
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True)
        self.log("test/nme",  self.test_nme,  on_step=False, on_epoch=True)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        warmup_epochs = 5 
        
        cosine_epochs = max(1, self.trainer.max_epochs - warmup_epochs)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.01, 
            end_factor=1.0, 
            total_iters=warmup_epochs
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=self.hparams.lr * 1e-2,
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch",
                "frequency": 1
            },
        }
    
    def decode_heatmaps(self, heatmaps):
        B, C, H, W = heatmaps.shape
        
        heatmaps_reshaped = heatmaps.reshape(B, C, -1)
        
        maxvals, idx = torch.max(heatmaps_reshaped, dim=2)
        maxvals = maxvals.unsqueeze(-1)
        
        preds_x = (idx % W).float()
        preds_y = (idx // W).float()
        

        for b in range(B):
            for c in range(C):
                hm = heatmaps[b, c]
                px = int(preds_x[b, c].item())
                py = int(preds_y[b, c].item())
                
                if 0 < px < W - 1 and 0 < py < H - 1:
                    diff_x = hm[py, px + 1] - hm[py, px - 1]
                    diff_y = hm[py + 1, px] - hm[py - 1, px]
                    
                    preds_x[b, c] += torch.sign(diff_x) * 0.25
                    preds_y[b, c] += torch.sign(diff_y) * 0.25

        preds = torch.stack([preds_x, preds_y], dim=-1)
        preds = preds / W
        
        return preds, maxvals