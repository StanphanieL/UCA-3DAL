import os
import sys
import random
from math import cos, pi

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

# ensure project root (containing datasets/ and tools/) is on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
for p in [THIS_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config.train_cponet_config import get_parser
from tools import log as log_tools
from network.cponet import CPONet, model_fn


def cosine_lr_after_step(optimizer, base_lr, epoch, step_epoch, total_epochs, clip: float = 1e-6):
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
        raise RuntimeError(f"Unsupported dataset for CPONet: {cfg.dataset}")
    return Dataset(cfg)


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
    dataset.trainLoader()
    logger.info(f"Training samples: {len(dataset.train_file_list)}")

    num_classes = getattr(dataset, "num_classes", 0)
    model = CPONet(
        cfg.in_channels,
        cfg.out_channels,
        num_classes=num_classes,
        class_embed_dim=cfg.class_embed_dim,
        conditional_mode=cfg.conditional_mode,
    ).cuda()

    # optionally load Stage-1 backbone weights
    if getattr(cfg, "contrastive_backbone", ""):
        try:
            ckpt = torch.load(cfg.contrastive_backbone, map_location="cuda")
            state = ckpt["model"] if "model" in ckpt else ckpt
            model_dict = model.state_dict()
            mapped = {k: v for k, v in state.items() if k.startswith("backbone.") and k in model_dict}
            model_dict.update(mapped)
            model.load_state_dict(model_dict, strict=False)
            logger.info(f"Loaded backbone from contrastive ckpt: {cfg.contrastive_backbone} ({len(mapped)} params)")
        except Exception as e:
            logger.info(f"Load contrastive backbone failed: {e}")

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

    start_epoch, pretrain_file = log_tools.checkpoint_restore(model, optimizer, cfg.logpath, pretrain_file=cfg.pretrain)
    if pretrain_file:
        logger.info(f"Restore from {pretrain_file}")
    else:
        logger.info(f"Start from epoch {start_epoch}")

    best_loss = 1e9

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        am = log_tools.AverageMeter()
        iter_time = log_tools.AverageMeter()
        batch_time = log_tools.AverageMeter()
        end_time = time.time()

        for i, batch in enumerate(dataset.train_data_loader):
            batch_time.update(time.time() - end_time)
            cosine_lr_after_step(optimizer, cfg.lr, epoch, cfg.step_epoch, cfg.epochs, clip=1e-6)

            loss, _, visual_dict, meter_dict = model_fn(batch, model, cfg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            am.update(visual_dict["loss"], 1)
            iter_time.update(time.time() - end_time)
            end_time = time.time()

            current_iter = epoch * len(dataset.train_data_loader) + i + 1
            max_iter = cfg.epochs * len(dataset.train_data_loader)
            remain_iter = max_iter - current_iter
            remain_time = remain_iter * iter_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time_str = f"{int(t_h):02d}:{int(t_m):02d}:{int(t_s):02d}"

            line = (
                f"[CPONet] epoch: {epoch}/{cfg.epochs} iter: {i+1}/{len(dataset.train_data_loader)} "
                f"loss: {am.val:.4f}({am.avg:.4f}) data_time: {batch_time.val:.2f}({batch_time.avg:.2f}) "
                f"iter_time: {iter_time.val:.2f}({iter_time.avg:.2f}) remain_time: {remain_time_str}\n"
            )
            sys.stdout.write(line)
            try:
                logger.info(line.strip())
            except Exception:
                pass
            if i == len(dataset.train_data_loader) - 1:
                print()

        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/loss", am.avg, epoch)
        writer.add_scalar("train/learning_rate", lr, epoch)

        latest = log_tools.checkpoint_save_newest(model, optimizer, cfg.logpath, epoch, save_freq=cfg.save_freq)
        logger.info(f"[CPONet] Saved latest checkpoint: {latest}")

        if am.avg < best_loss:
            best_loss = am.avg
            best_file = os.path.join(cfg.logpath, "best.pth")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, best_file)
            logger.info(f"[CPONet Best] epoch={epoch} loss={best_loss:.6f} -> {best_file}")


if __name__ == "__main__":
    import time

    main()