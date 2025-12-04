"""Contrastive Prototype Encoder (CPE) for UCA-3DAL.

This module provides:
- CPE: shared MinkUNet34C backbone + projection head.
- SupConLoss: supervised contrastive loss.
- PrototypeMemory: EMA-updated class prototypes.
- PrototypeNCELoss: prototype-based NCE loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from .mink import Mink_unet


class CPE(nn.Module):
    """Contrastive encoder with MinkUNet34C backbone and MLP projection head.

    The encoder outputs L2-normalized embeddings used for contrastive learning
    and prototype estimation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        proj_dim: int = 128,
        arch: str = "MinkUNet34C",
    ) -> None:
        super().__init__()
        self.backbone = Mink_unet(in_channels=in_channels, out_channels=out_channels, D=3, arch=arch)
        self.proj = nn.Sequential(
            nn.Linear(out_channels, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, proj_dim, bias=True),
        )
        self.global_pool = ME.MinkowskiGlobalAvgPooling()

    def forward_embed(self, feat_voxel: torch.Tensor, xyz_voxel: torch.Tensor) -> torch.Tensor:
        """Encode a batch of sparse voxels into normalized embeddings.

        Args:
            feat_voxel: (N, C_in) voxel features.
            xyz_voxel: (N, 4) quantized coordinates (batch,x,y,z).
        Returns:
            (B, proj_dim) L2-normalized embeddings.
        """

        cuda_cur_device = torch.cuda.current_device()
        inputs = ME.SparseTensor(feat_voxel, xyz_voxel, device=f"cuda:{cuda_cur_device}")
        voxel_feat = self.backbone(inputs)           # SparseTensor [#vox, C]
        pooled = self.global_pool(voxel_feat)        # SparseTensor [B, C]
        x = pooled.F                                 # [B, C]
        z = self.proj(x)                             # [B, proj_dim]
        return F.normalize(z, dim=1)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020).

    Expects features of shape [B, V, D] where V is number of views.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        B, V, D = features.shape
        feats = F.normalize(features.view(B * V, D), dim=1)  # [B*V, D]

        labels = labels.view(-1, 1)                          # [B, 1]
        mask = torch.eq(labels, labels.T).float().to(device) # [B, B]
        mask = mask.repeat_interleave(V, dim=0).repeat_interleave(V, dim=1)  # [B*V, B*V]

        logits = torch.matmul(feats, feats.T) / self.temperature            # [B*V, B*V]
        logits_mask = torch.ones_like(mask) - torch.eye(B * V, device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos
        loss = loss.view(B, V).mean()
        return loss


class PrototypeMemory(nn.Module):
    """EMA-updated prototypes for each semantic category."""

    def __init__(self, num_classes: int, dim: int, momentum: float = 0.9) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.dim = int(dim)
        self.m = float(momentum)
        self.register_buffer("proto", torch.zeros(num_classes, dim))
        self.register_buffer("counts", torch.zeros(num_classes))

    @torch.no_grad()
    def update(self, z: torch.Tensor, y: torch.Tensor) -> None:
        """Update prototypes with a mini-batch of embeddings.

        Args:
            z: (N, D) normalized embeddings.
            y: (N,) integer class ids.
        """

        if z.numel() == 0:
            return
        y = y.view(-1)
        for cls in y.unique():
            cid = int(cls.item())
            idx = (y == cid)
            if idx.sum() == 0:
                continue
            z_mean = F.normalize(z[idx].mean(dim=0, keepdim=True), dim=1)
            old = self.proto[cid : cid + 1]
            new = F.normalize(self.m * old + (1.0 - self.m) * z_mean, dim=1)
            self.proto[cid : cid + 1] = new
            self.counts[cid] = self.counts[cid] + idx.sum()


class PrototypeNCELoss(nn.Module):
    """InfoNCE-style loss pulling samples to their class prototype."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z: torch.Tensor, y: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Compute NCE loss between embeddings and class prototypes.

        Args:
            z: (N, D) normalized embeddings.
            y: (N,) class ids.
            prototypes: (K, D) normalized prototypes.
        """

        logits = torch.matmul(z, prototypes.T) / self.temperature  # [N, K]
        loss = F.cross_entropy(logits, y)
        return loss