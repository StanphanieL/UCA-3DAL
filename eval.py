import os
import sys
import random
import time
from math import cos, pi
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    from tqdm import tqdm
except ImportError: 
    tqdm = lambda x, **kwargs: x

# make project root and this package importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
for p in [THIS_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tools import log as log_tools
from network.cponet import CPONet, eval_fn as cponet_eval_fn
from network.cpe import CPE, PrototypeMemory
from config.eval_config import get_parser as get_eval_parser


def get_dataset(cfg):
    if cfg.dataset == "AnomalyShapeNet":
        from datasets.AnomalyShapeNet import Dataset
    elif cfg.dataset == "Real3D":
        from datasets.Real3D import Dataset
    elif cfg.dataset == "IEC3DAD":
        from datasets.IEC3DAD import Dataset
    else:
        raise RuntimeError(f"Unsupported dataset: {cfg.dataset}")
    return Dataset(cfg)


def safe_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def safe_ap(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def predict_category(cpe: CPE, proto_mem: PrototypeMemory, batch: dict) -> int:
    """Nearest-prototype category prediction using CPE embeddings."""

    with torch.no_grad():
        z = cpe.forward_embed(batch["feat_voxel"], batch["xyz_voxel"])  # (1, D)
        proto = torch.nn.functional.normalize(proto_mem.proto, dim=1)     # (C, D)
        logits = torch.matmul(z, proto.T)                                 # (1, C)
        cid = int(torch.argmax(logits, dim=1).item())
    return cid


def build_geo_tta_views(xyz: np.ndarray, cfg) -> list:
    """Generate K geometric TTA views from original xyz coordinates."""

    import MinkowskiEngine as ME

    views = []
    base_xyz = xyz
    for _ in range(int(cfg.tta_views)):
        xyz_t = base_xyz.copy()

        deg = float(cfg.tta_rotate_deg)
        ax, ay, az = np.deg2rad(np.random.uniform(-deg, deg, size=3))
        Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]], dtype=np.float32)
        Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]], dtype=np.float32)
        Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]], dtype=np.float32)
        R = Rz @ Ry @ Rx
        xyz_t = xyz_t @ R.T

        s = 1.0 + np.random.uniform(-float(cfg.tta_scale), float(cfg.tta_scale))
        xyz_t = xyz_t * s

        if float(cfg.tta_jitter) > 0:
            xyz_t = xyz_t + np.random.normal(scale=float(cfg.tta_jitter), size=xyz_t.shape).astype(np.float32)

        q, f, _, inv = ME.utils.sparse_quantize(
            xyz_t.astype(np.float32),
            xyz_t.astype(np.float32),
            quantization_size=cfg.voxel_size,
            return_index=True,
            return_inverse=True,
        )
        xyz_voxel_t, feat_voxel_t = ME.utils.sparse_collate([q], [f])
        if isinstance(inv, np.ndarray):
            v2p_t = torch.from_numpy(inv).long()
        elif torch.is_tensor(inv):
            v2p_t = inv.long().cpu()
        else:
            v2p_t = torch.as_tensor(inv, dtype=torch.long)
        batch_count_t = torch.tensor([0, xyz_t.shape[0]], dtype=torch.int64)
        views.append({
            "xyz_voxel": xyz_voxel_t,
            "feat_voxel": feat_voxel_t,
            "v2p_index": v2p_t,
            "batch_count": batch_count_t,
        })
    return views


def _strip_module_prefix(state_dict):

    if not isinstance(state_dict, dict):
        return state_dict
    needs_strip = any(k.startswith("module.") for k in state_dict.keys())
    if not needs_strip:
        return state_dict
    return {k[len("module.") :]: v for k, v in state_dict.items()}


def main():
    parser = get_eval_parser()
    cfg = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed(cfg.manual_seed)

    logger = log_tools.get_logger(cfg)

    dataset = get_dataset(cfg)

    if getattr(cfg, "eval_category_only", ""):
        only_cat = str(cfg.eval_category_only).strip()
        if hasattr(dataset, "test_file_list"):
            before = len(dataset.test_file_list)
            filtered = []
            for p in dataset.test_file_list:
                norm_p = p.replace("\\", "/")
                parts = norm_p.split("/")
                true_cat = None
                if hasattr(dataset, "cat2id"):
                    for seg in parts:
                        if seg in dataset.cat2id:
                            true_cat = seg
                            break
                if true_cat is None:
                    true_cat = parts[-3] if len(parts) >= 3 else ""
                if (
                    true_cat == only_cat
                    or true_cat.lower() == only_cat.lower()
                    or true_cat.lower().endswith(only_cat.lower())
                ):
                    filtered.append(p)
            dataset.test_file_list = filtered
            after = len(dataset.test_file_list)
            logger.info(
                f"Evaluating category-only subset: '{only_cat}' | samples: {after} (was {before})"
            )

    dataset.testLoader()
    logger.info(f"Test samples: {len(dataset.test_file_list)}")

    ckpt_path = os.path.join(cfg.logpath, cfg.checkpoint_name)
    state = torch.load(ckpt_path, map_location="cuda")
    state_model = state["model"] if isinstance(state, dict) and "model" in state else state
    state_model = _strip_module_prefix(state_model)

    num_classes = getattr(dataset, "num_classes", 0)
    if "class_embed.weight" in state_model:
        num_classes = state_model["class_embed.weight"].shape[0]

    model = CPONet(
        cfg.in_channels,
        cfg.out_channels,
        num_classes=num_classes,
        class_embed_dim=cfg.class_embed_dim,
        conditional_mode=cfg.conditional_mode,
    ).cuda()
    missing, unexpected = model.load_state_dict(state_model, strict=False)
    if missing or unexpected:
        logger.info(
            f"Loaded CPONet from {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})"
        )
    else:
        logger.info(f"Loaded CPONet from {ckpt_path}")
    model.eval()

    # CPE + prototypes for category / cluster prediction
    cpe = CPE(cfg.in_channels, cfg.out_channels, proj_dim=cfg.proj_dim).cuda()
    proto_mem = None
    if cfg.contrastive_ckpt and os.path.isfile(cfg.contrastive_ckpt):
        ckpt_con = torch.load(cfg.contrastive_ckpt, map_location="cuda")
        cpe_state = ckpt_con["model"] if isinstance(ckpt_con, dict) and "model" in ckpt_con else ckpt_con
        cpe_state = _strip_module_prefix(cpe_state)
        cpe.load_state_dict(cpe_state, strict=False)
        cpe.eval()

        if isinstance(ckpt_con, dict) and "prototypes" in ckpt_con and ckpt_con["prototypes"] is not None:
            proto_state = ckpt_con["prototypes"]
            if isinstance(proto_state, dict) and "proto" in proto_state:
                num_classes_proto = proto_state["proto"].shape[0]
                dim_proto = proto_state["proto"].shape[1]
            else:
                num_classes_proto = getattr(dataset, "num_classes", 0)
                dim_proto = 128
            proto_mem = PrototypeMemory(num_classes=num_classes_proto, dim=dim_proto).cuda()
            proto_mem.load_state_dict(proto_state, strict=False)
        logger.info(f"Loaded CPE from {cfg.contrastive_ckpt}")
    else:
        logger.info("No contrastive_ckpt found; CPONet will run without conditioning")

    # conditioning / cluster-assigner status
    cond_src = getattr(cfg, "conditioning_source", "auto").lower()
    if cond_src not in {"auto", "true", "cluster", "none"}:
        cond_src = "auto"
    cluster_assigner_ready = proto_mem is not None
    logger.info(f"[Conditioning] source={cond_src}")
    if cluster_assigner_ready:
        logger.info("[Cluster] CPE prototypes available for predicted clusters/categories.")
    else:
        logger.info("[Cluster] prototypes unavailable -> conditioning falls back to true category or is disabled.")

    obj_labels = []
    obj_scores = []

    pt_scores_all = []
    pt_labels_all = []

    per_cat_obj_labels = {}
    per_cat_obj_scores = {}
    per_cat_pt_scores = {}
    per_cat_pt_labels = {}


    sample_cats = []         
    pred_clusters = []        
    true_cats = []           

    import MinkowskiEngine as ME

    iterator = tqdm(
        dataset.test_data_loader,
        total=len(dataset.test_file_list),
        desc="Evaluating",
        dynamic_ncols=True,
    )

    for i, batch in enumerate(iterator):
        sample_path = batch["fn"][0]
        sample_name = Path(sample_path).stem
        norm_path = sample_path.replace("\\", "/")
        parts = norm_path.split("/")
        cat_name = parts[-3] if len(parts) >= 3 else ""
        sample_cats.append(cat_name)

        # map to integer category id when available
        if hasattr(dataset, "cat2id") and cat_name in getattr(dataset, "cat2id", {}):
            true_id = dataset.cat2id[cat_name]
        else:
            true_id = -1
        true_cats.append(true_id)

        # object-level label: all datasets follow 0=normal, 1=anomaly in labels
        if "labels" in batch:
            y = int(batch["labels"][0].item())
        else:
            y = 0 if "good" in sample_name.lower() else 1
        obj_labels.append(y)

        # predicted cluster/category id via CPE/prototypes (if available)
        cid = -1
        if proto_mem is not None:
            if "xyz_voxel" not in batch or "feat_voxel" not in batch:
                raise RuntimeError("Dataset testMerge must provide 'xyz_voxel' and 'feat_voxel' for CPE.")
            cid = predict_category(cpe, proto_mem, batch)
        pred_clusters.append(cid)

        # decide conditioning id based on user option
        if cond_src == "true":
            cond_cid = true_id
        elif cond_src == "cluster":
            cond_cid = cid if cid >= 0 else -1
        elif cond_src == "none":
            cond_cid = -1
        else:  # auto
            cond_cid = cid if cid >= 0 else true_id
        cond_ids = None if cond_cid < 0 else torch.tensor([cond_cid], dtype=torch.long).cuda()

        # point coordinates at point level for smoothing / TTA
        xyz_tensor = batch.get("xyz_original", None)
        if xyz_tensor is not None:
            xyz_np = xyz_tensor.numpy()
        else:
            xyz_np = batch["xyz_voxel"].F[:, 1:].cpu().numpy()

        _, pred_offset = cponet_eval_fn(
            batch,
            model,
            category_ids=cond_ids,
            quantile=cfg.score_quantile,
            score_method=cfg.score_method,
        )

        # point-wise scores on base view
        pt_scores = torch.sum(torch.abs(pred_offset.detach().cpu()), dim=-1).numpy()

        # optional Geo-TTA (start from raw base scores)
        if cfg.tta_views > 0:
            tta_masks = []
            for tta_batch in build_geo_tta_views(xyz_np, cfg):
                for k in ["xyz_voxel", "feat_voxel", "v2p_index", "batch_count"]:
                    tta_batch[k] = tta_batch[k].to(pred_offset.device)
                _, pred_offset_t = cponet_eval_fn(
                    tta_batch,
                    model,
                    category_ids=cond_ids,
                    quantile=cfg.score_quantile,
                    score_method=cfg.score_method,
                )
                tta_mask = torch.sum(torch.abs(pred_offset_t.detach().cpu()), dim=-1).numpy()
                tta_masks.append(tta_mask)

            if tta_masks:
                L = pt_scores.shape[0]
                fused = [pt_scores] + [m[:L] for m in tta_masks]
                if cfg.tta_reduce == "max":
                    pt_scores = np.max(np.stack(fused, axis=0), axis=0)
                else:
                    pt_scores = np.mean(np.stack(fused, axis=0), axis=0)

        inds = None
        if getattr(cfg, "smooth_knn", 0) and cfg.smooth_knn > 0:
            try:
                from sklearn.neighbors import NearestNeighbors

                k = min(cfg.smooth_knn, xyz_np.shape[0])
                nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(xyz_np)
                inds = nbrs.kneighbors(xyz_np, return_distance=False)
            except Exception as e:
                logger.warning(f"[smooth_knn] failed to build index: {e}")
                inds = None

        if inds is not None:
            try:
                pt_scores = pt_scores[inds].mean(axis=1)
            except Exception as e:
                logger.warning(f"[smooth_knn] apply failed: {e}")

        if cfg.score_method == "mean":
            score_obj = float(np.mean(pt_scores))
        elif cfg.score_method == "max":
            score_obj = float(np.max(pt_scores))
        else:
            score_obj = float(np.quantile(pt_scores, cfg.score_quantile))

        obj_scores.append(score_obj)
        # per-category object-level accumulation
        per_cat_obj_labels.setdefault(cat_name, []).append(y)
        per_cat_obj_scores.setdefault(cat_name, []).append(score_obj)

        sample_pt_labels = None

        if cfg.dataset == "IEC3DAD" and "gt_mask" in batch:
            gt_mask = batch["gt_mask"].numpy()
            sample_pt_labels = gt_mask

        elif cfg.dataset == "AnomalyShapeNet":
            norm_path = sample_path.replace("\\", "/")
            parts = norm_path.split("/")
            cat_name = parts[-3] if len(parts) >= 3 else ""
            sample_name = Path(sample_path).stem
            if "positive" in sample_name.lower():
                sample_pt_labels = np.zeros_like(pt_scores, dtype=np.float32)
            else:
                original_gt_path = f"datasets/AnomalyShapeNet/dataset/pcd/{cat_name}/GT/{sample_name}.txt"
                gt_file = original_gt_path if os.path.exists(original_gt_path) else None
                if gt_file is None:
                    raise FileNotFoundError(
                        f"GT file not found for {sample_path}. Tried: {original_gt_path}"
                    )
                arr = np.loadtxt(gt_file, delimiter=",")
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                sample_pt_labels = arr[:, 3].astype(np.float32)

        elif cfg.dataset == "Real3D":
            norm_path = sample_path.replace("\\", "/")
            parts = norm_path.split("/")
            cat_name = parts[-3] if len(parts) >= 3 else ""
            sample_name = Path(sample_path).stem
            if "good" in sample_name.lower():
                sample_pt_labels = np.zeros_like(pt_scores, dtype=np.float32)
            else:
                gt_mask_path = f"{dataset.pcd_root}/{cat_name}/gt/"
                gt_file = os.path.join(gt_mask_path, sample_name + ".txt")
                try:
                    arr = np.loadtxt(gt_file)
                except Exception:
                    arr = np.loadtxt(gt_file, delimiter=",")
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                sample_pt_labels = arr[:, -1].astype(np.float32)

        if sample_pt_labels is not None:
            L = min(len(sample_pt_labels), pt_scores.shape[0])
            pt_scores_clip = pt_scores[:L]
            pt_labels_clip = sample_pt_labels[:L]
            pt_scores_all.append(pt_scores_clip)
            pt_labels_all.append(pt_labels_clip)
            per_cat_pt_scores.setdefault(cat_name, []).append(pt_scores_clip)
            per_cat_pt_labels.setdefault(cat_name, []).append(pt_labels_clip)

    if pt_scores_all and pt_labels_all:
        labels_arr = np.asarray(obj_labels)
        scores_arr = np.asarray(obj_scores)
        pred_masks = pt_scores_all
        gt_masks = pt_labels_all

        def normalize_array(x: np.ndarray, method: str) -> np.ndarray:
            x = np.asarray(x)
            if method == "zscore":
                mu = np.mean(x)
                sd = np.std(x) + 1e-12
                return (x - mu) / sd
            if method == "mad":
                med = np.median(x)
                mad = np.median(np.abs(x - med)) + 1e-12
                return (x - med) / mad
            # default: min-max
            x_min = np.min(x)
            x_max = np.max(x)
            d = (x_max - x_min) + 1e-12
            return (x - x_min) / d

        def compute_group_metrics(group_map):
            stats = {}
            for gid, idxs in sorted(group_map.items(), key=lambda x: x[0]):
                if isinstance(gid, int) and gid < 0:
                    continue
                if not idxs:
                    continue
                l = labels_arr[idxs]
                s = scores_arr[idxs]
                s_n = normalize_array(s, getattr(cfg, "cluster_norm_type", "minmax"))
                auc_obj = safe_auc(l, s_n)
                ap_obj = safe_ap(l, s_n)
                pts = np.concatenate([pred_masks[i] for i in idxs], axis=0)
                gts = np.concatenate([gt_masks[i] for i in idxs], axis=0)
                pts_n = normalize_array(pts, getattr(cfg, "cluster_norm_type", "minmax"))
                auc_pt = safe_auc(gts, pts_n)
                ap_pt = safe_ap(gts, pts_n)
                pos_rate = float(np.mean(gts)) if getattr(cfg, "print_pos_rate", False) else None
                stats[gid] = (auc_obj, auc_pt, ap_obj, ap_pt, len(idxs), pos_rate)
            macro = None
            valid = [v for v in stats.values() if not (np.isnan(v[0]) or np.isnan(v[1]))]
            if len(valid) > 0:
                mean_obj = float(np.nanmean([v[0] for v in valid]))
                mean_pt = float(np.nanmean([v[1] for v in valid]))
                mean_obj_ap = float(np.nanmean([v[2] for v in valid]))
                mean_pt_ap = float(np.nanmean([v[3] for v in valid]))
                macro = (mean_obj, mean_pt, mean_obj_ap, mean_pt_ap)
            return stats, macro

        from collections import defaultdict

        norm_type = getattr(cfg, "cluster_norm_type", "minmax")

        if cluster_assigner_ready and getattr(cfg, "cluster_norm", False):
            pass

        cat_groups = defaultdict(list)
        for idx, cat in enumerate(sample_cats):
            cat_groups[cat].append(idx)
        print(f"\n[Per-Category metrics (by true category) with category-wise {norm_type} normalization]")
        cat_stats, macro_cat = compute_group_metrics(cat_groups)
        for cat, v in cat_stats.items():
            if getattr(cfg, "print_pos_rate", False):
                print(
                    f"  [cat {cat}] N={v[4]} objAUC={v[0]} ptAUC={v[1]} objAP={v[2]} ptAP={v[3]} pos_rate={v[5]:.6f}"
                    if v[5] is not None
                    else f"  [cat {cat}] N={v[4]} objAUC={v[0]} ptAUC={v[1]} objAP={v[2]} ptAP={v[3]}"
                )
            else:
                print(f"  [cat {cat}] N={v[4]} objAUC={v[0]} ptAUC={v[1]} objAP={v[2]} ptAP={v[3]}")
        if macro_cat is not None:
            print(
                f"  [category-macro] objAUC={macro_cat[0]} ptAUC={macro_cat[1]} "
                f"objAP={macro_cat[2]} ptAP={macro_cat[3]}"
            )


if __name__ == "__main__":
    main()
