import os
import sys
import random
import time
from math import cos, pi

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

# ensure project root (containing datasets/ and tools/) is on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
# ensure local UCA-3DAL packages (tools, datasets, network, config) are searched first
for p in [THIS_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config.train_cpe_config import get_parser
from tools import log as log_tools
from network.cpe import CPE, SupConLoss, PrototypeMemory, PrototypeNCELoss


def cosine_lr_after_step(optimizer, base_lr, epoch, step_epoch, total_epochs, clip: float = 1e-6):
    """Cosine LR with a flat warm portion before `step_epoch`.

    Epoch index is assumed to start from 0.
    """

    if epoch < step_epoch:
        lr = base_lr
    else:
        lr = clip + 0.5 * (base_lr - clip) * (1 + cos(pi * ((epoch - step_epoch) / (total_epochs - step_epoch))))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_dataset(cfg):

    if cfg.dataset == "AnomalyShapeNet":
        from datasets.AnomalyShapeNet import Dataset
    elif cfg.dataset == "Real3D":
        from datasets.Real3D import Dataset
    elif cfg.dataset == "IEC3DAD":
        from datasets.IEC3DAD import Dataset
    else:
        raise RuntimeError(f"Unsupported dataset for CPE: {cfg.dataset}")
    return Dataset(cfg)


def restore_contrastive_checkpoint(model, optimizer, proto_module, logpath, pretrain_file: str = ""):
    """Load CPE + prototype checkpoint if available.

    Preference:
    1) explicit pretrain_file;
    2) <logpath>/latest.pth;
    3) highest numbered 00*.pth under logpath.
    """

    epoch = 0
    if not pretrain_file:
        latest = os.path.join(logpath, "latest.pth")
        if os.path.isfile(latest):
            pretrain_file = latest
        else:
            import glob

            files = sorted(glob.glob(os.path.join(logpath + "00*.pth")))
            if files:
                pretrain_file = files[-1]

    if pretrain_file and os.path.isfile(pretrain_file):
        ckpt = torch.load(pretrain_file, map_location="cuda")
        model.load_state_dict(ckpt["model"], strict=False)
        if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            for state in optimizer.state.values():
                if state is None:
                    continue
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        if proto_module is not None and "prototypes" in ckpt and ckpt["prototypes"] is not None:
            proto_module.load_state_dict(ckpt["prototypes"], strict=False)
        if "epoch" in ckpt:
            try:
                epoch = int(ckpt["epoch"]) + 1
            except Exception:
                epoch = 0
    return epoch, pretrain_file


def save_contrastive_latest(model, optimizer, proto_module, logpath, epoch: int) -> str:

    os.makedirs(logpath, exist_ok=True)
    latest_file = os.path.join(logpath, "latest.pth")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "prototypes": proto_module.state_dict() if proto_module is not None else None,
            "epoch": epoch,
        },
        latest_file,
    )
    return latest_file


def train_epoch_contrastive(train_loader, model, supcon_crit, proto_mod, proto_crit, optimizer, cfg, epoch, logger, writer):
    """Single training epoch for Stage-1 CPE."""

    model.train()
    iter_time = log_tools.AverageMeter()
    batch_time = log_tools.AverageMeter()
    am_loss = log_tools.AverageMeter()
    am_acc = log_tools.AverageMeter()

    start_time = time.time()
    end_time = time.time()

    try:
        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"[CPE] start epoch={epoch} lr={cur_lr}")
    except Exception:
        pass

    for i, batch in enumerate(train_loader):
        batch_time.update(time.time() - end_time)
        cosine_lr_after_step(optimizer, cfg.lr, epoch, cfg.step_epoch, cfg.epochs, clip=1e-6)

        z1 = model.forward_embed(batch["feat_voxel_view1"], batch["xyz_voxel_view1"])
        z2 = model.forward_embed(batch["feat_voxel_view2"], batch["xyz_voxel_view2"])
        labels = batch["labels"].cuda(non_blocking=True)

        feats = torch.stack([z1, z2], dim=1)  # [B, 2, D]
        loss = supcon_crit(feats, labels)

        if proto_mod is not None and proto_crit is not None and cfg.proto_loss_weight > 0:
            with torch.no_grad():
                proto_mod.update(torch.cat([z1, z2], dim=0), torch.cat([labels, labels], dim=0))
            proto = torch.nn.functional.normalize(proto_mod.proto, dim=1)
            proto_loss = proto_crit(torch.cat([z1, z2], dim=0), torch.cat([labels, labels], dim=0), proto)
            loss = loss + cfg.proto_loss_weight * proto_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # simple accuracy: nearest-prototype classification
        with torch.no_grad():
            if proto_mod is not None:
                proto = torch.nn.functional.normalize(proto_mod.proto, dim=1)
                z_mean = torch.nn.functional.normalize(0.5 * (z1 + z2), dim=1)
                logits = torch.matmul(z_mean, proto.T)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == labels).float().mean().item()
            else:
                acc = 0.0

        am_loss.update(loss.item(), labels.shape[0])
        am_acc.update(acc, labels.shape[0])

        current_iter = epoch * len(train_loader) + i + 1
        max_iter = cfg.epochs * len(train_loader)
        remain_iter = max_iter - current_iter
        iter_time.update(time.time() - end_time)
        end_time = time.time()
        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time_str = f"{int(t_h):02d}:{int(t_m):02d}:{int(t_s):02d}"

        line = (
            f"[CPE] epoch: {epoch}/{cfg.epochs} iter: {i+1}/{len(train_loader)} "
            f"loss: {am_loss.val:.4f}({am_loss.avg:.4f}) acc: {acc:.3f} "
            f"data_time: {batch_time.val:.2f}({batch_time.avg:.2f}) "
            f"iter_time: {iter_time.val:.2f}({iter_time.avg:.2f}) remain_time: {remain_time_str}\n"
        )
        sys.stdout.write(line)
        try:
            logger.info(line.strip())
        except Exception:
            pass
        if i == len(train_loader) - 1:
            print()

    logger.info(
        "[CPE] epoch: {}/{} train loss: {:.4f} time: {:.1f}s".format(
            epoch, cfg.epochs, am_loss.avg, time.time() - start_time
        )
    )
    lr = optimizer.param_groups[0]["lr"]
    writer.add_scalar("contrastive/train_loss", am_loss.avg, epoch)
    writer.add_scalar("contrastive/train_acc", am_acc.avg, epoch)
    writer.add_scalar("train/learning_rate", lr, epoch)

    return am_loss.avg, am_acc.avg


def main():
    cfg = get_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id

    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed(cfg.manual_seed)

    logger = log_tools.get_logger(cfg)
    writer = SummaryWriter(cfg.logpath)

    dataset = get_dataset(cfg)
    dataset.contrastiveLoader()
    logger.info(f"Contrastive training samples: {len(dataset.train_file_list)}")

    model = CPE(cfg.in_channels, cfg.out_channels, proj_dim=cfg.proj_dim).cuda()
    logger.info("#Model parameters: {}".format(sum(x.nelement() for x in model.parameters())))

    if cfg.optimizer == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optimizer == "SGD":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
            betas=(0.9, 0.99),
            weight_decay=cfg.weight_decay,
        )

    supcon_crit = SupConLoss(temperature=cfg.temperature).cuda()
    proto_mod = PrototypeMemory(num_classes=dataset.num_classes, dim=cfg.proj_dim, momentum=cfg.proto_m).cuda()
    proto_crit = PrototypeNCELoss(temperature=cfg.temperature).cuda()

    start_epoch, pretrain_file = restore_contrastive_checkpoint(model, optimizer, proto_mod, cfg.logpath, pretrain_file=cfg.pretrain)
    if pretrain_file:
        logger.info(f"Restore from {pretrain_file}")
    else:
        logger.info(f"Start from epoch {start_epoch}")

    best_metric = -1e9 if cfg.contrastive_best_metric == "acc" else 1e9
    best_file = None

    for epoch in range(start_epoch, cfg.epochs):
        epoch_loss, epoch_acc = train_epoch_contrastive(
            dataset.train_data_loader, model, supcon_crit, proto_mod, proto_crit, optimizer, cfg, epoch, logger, writer
        )
        save_contrastive_latest(model, optimizer, proto_mod, cfg.logpath, epoch)

        metric = epoch_acc if cfg.contrastive_best_metric == "acc" else epoch_loss
        improved = (metric > best_metric) if cfg.contrastive_best_metric == "acc" else (metric < best_metric)
        if improved:
            best_metric = metric
            best_file = os.path.join(cfg.logpath, "best.pth")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "prototypes": proto_mod.state_dict(),
                    "epoch": epoch,
                },
                best_file,
            )
            logger.info(
                f"[Best CPE] epoch={epoch} metric={cfg.contrastive_best_metric} value={best_metric:.6f} -> {best_file}"
            )


if __name__ == "__main__":
    main()