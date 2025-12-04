"""Conditional Point Offset Network (CPONet) for UCA-3DAL.

This module implements the unified conditional offset regressor used in the
second stage of the framework. It consumes sparse voxel features produced by a
MinkUNet34C backbone and predicts per-point 3D offsets, optionally conditioned
on category embeddings via FiLM or concatenation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from .mink import Mink_unet


class CPONet(nn.Module):
    """Point-wise conditional offset regressor with shared MinkUNet backbone."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_classes: int = 0,
        class_embed_dim: int = 0,
        conditional_mode: str = "concat",
    ) -> None:
        super().__init__()
        self.backbone = Mink_unet(in_channels=in_channels, out_channels=out_channels, D=3, arch="MinkUNet34C")

        self.num_classes = int(num_classes)
        self.conditional_mode = str(conditional_mode).lower()
        self.class_embed_dim = class_embed_dim if (self.num_classes > 0 and class_embed_dim > 0) else 0

        if self.class_embed_dim > 0:
            self.class_embed = nn.Embedding(self.num_classes, self.class_embed_dim)
            if self.conditional_mode == "film":
                # FiLM produces per-channel scale and shift
                self.film = nn.Linear(self.class_embed_dim, out_channels * 2)
            else:
                self.film = None
        else:
            self.class_embed = None
            self.film = None

        feat_in = out_channels if self.film is not None else (
            out_channels + (self.class_embed_dim if self.class_embed is not None else 0)
        )
        self.linear_offset = nn.Sequential(
            nn.Linear(feat_in, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
            nn.Linear(out_channels, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.Linear(16, 3, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialization for Minkowski convs and BN in backbone."""

        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(
        self,
        feat_voxel: torch.Tensor,
        xyz_voxel: torch.Tensor,
        v2p_index: torch.Tensor,
        batch_count: torch.Tensor | None = None,
        category_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            feat_voxel: (N, C_in) voxel features.
            xyz_voxel: (N, 4) quantized coordinates.
            v2p_index: (P,) mapping from points to voxel features.
            batch_count: (B+1,) cumulative counts of points per batch item.
            category_ids: (B,) integer category ids for conditioning.
        Returns:
            (P, 3) predicted offsets.
        """

        cuda_cur_device = torch.cuda.current_device()
        inputs = ME.SparseTensor(feat_voxel, xyz_voxel, device=f"cuda:{cuda_cur_device}")
        voxel_feat = self.backbone(inputs)
        point_feat = voxel_feat.F[v2p_index]

        if self.class_embed is not None and batch_count is not None and category_ids is not None:
            # build per-point conditional embedding
            B = int(category_ids.shape[0])
            cond_list = []
            cat_emb = self.class_embed(category_ids.cuda())  # [B, D]
            for b in range(B):
                start = int(batch_count[b].item())
                end = int(batch_count[b + 1].item())
                n = end - start
                cond_list.append(cat_emb[b : b + 1].repeat(n, 1))
            cond = torch.cat(cond_list, dim=0)

            if self.film is not None and self.conditional_mode == "film":
                gamma_beta = self.film(cond)
                gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
                point_feat = point_feat * (1.0 + gamma) + beta
            else:
                point_feat = torch.cat([point_feat, cond], dim=1)

        pred_offset = self.linear_offset(point_feat)
        return pred_offset


def model_fn(batch: dict, model: CPONet, cfg) -> tuple[torch.Tensor, dict, dict, dict]:
    """Compute training loss for offset regression.

    Loss = L1 distance between predicted and ground-truth offsets
           + lambda_dir * cosine direction loss.
    """

    xyz_voxel = batch["xyz_voxel"]
    feat_voxel = batch["feat_voxel"]
    v2p_index = batch["v2p_index"]
    batch_count = batch["batch_count"]
    category_ids = batch.get("category_id", None)

    pred_offset = model(
        feat_voxel,
        xyz_voxel,
        v2p_index,
        batch_count=batch_count,
        category_ids=category_ids.cuda() if category_ids is not None else None,
    )

    gt_offsets = batch["batch_offset"].cuda()

    pt_diff = pred_offset - gt_offsets
    pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)
    valid = torch.ones(pt_dist.shape[0], device=pt_dist.device)
    offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

    gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)
    gt_unit = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
    pred_norm = torch.norm(pred_offset, p=2, dim=1)
    pred_unit = pred_offset / (pred_norm.unsqueeze(-1) + 1e-8)
    direction_diff = - (gt_unit * pred_unit).sum(-1)
    offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

    loss = offset_norm_loss + cfg.lambda_dir * offset_dir_loss

    with torch.no_grad():
        pred = {}
        visual_dict = {"loss": loss.item()}
        meter_dict = {"loss": (loss.item(), pred_offset.shape[0])}

    return loss, pred, visual_dict, meter_dict


def eval_fn(
    batch: dict,
    model: CPONet,
    category_ids: torch.Tensor | None = None,
    quantile: float = 0.95,
    score_method: str = "quantile",
):
    """Inference helper returning object-level score and raw offsets.

    Point-wise anomaly scores are the L1 norm of predicted offsets. The
    object-level score is aggregated by mean / max / quantile over points.
    """

    xyz_voxel = batch["xyz_voxel"]
    feat_voxel = batch["feat_voxel"]
    v2p_index = batch["v2p_index"]
    batch_count = batch.get("batch_count", None)

    with torch.no_grad():
        pred_offset = model(
            feat_voxel,
            xyz_voxel,
            v2p_index,
            batch_count=batch_count,
            category_ids=category_ids,
        )

    point_scores = torch.sum(torch.abs(pred_offset.detach().cpu()), dim=-1)

    if score_method == "mean":
        sample_score = torch.mean(point_scores)
    elif score_method == "max":
        sample_score = torch.max(point_scores)
    else:
        sample_score = torch.quantile(point_scores, quantile)

    return sample_score, pred_offset