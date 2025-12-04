import argparse


def get_parser():
    """Argument parser for unified evaluation (Geo-TTA, scoring, dataset options)."""

    parser = argparse.ArgumentParser(description="UCA-3DAL evaluation with Geo-TTA")

    # basic
    parser.add_argument("--dataset", type=str, default="AnomalyShapeNet", choices=["AnomalyShapeNet", "Real3D", "IEC3DAD"])
    parser.add_argument("--gpu_id", type=str, default="0", help="CUDA_VISIBLE_DEVICES index to use")
    parser.add_argument("--manual_seed", type=int, default=42, help="random seed for reproducibility")

    # model
    parser.add_argument("--voxel_size", type=float, default=0.03, help="MinkowskiEngine quantization size (meters)")
    parser.add_argument("--in_channels", type=int, default=3, help="input feature dimension per point/voxel")
    parser.add_argument("--out_channels", type=int, default=32, help="backbone feature dimension")
    parser.add_argument("--class_embed_dim", type=int, default=32, help="dimension of category embedding for CPONet conditioning")
    parser.add_argument("--conditional_mode", type=str, default="film", choices=["concat", "film"], help="how category embedding conditions CPONet",)
    parser.add_argument("--proj_dim", type=int, default=128, help="projection head output dim for CPE (must match Stage-1 training)")

    # checkpoints
    parser.add_argument("--logpath", type=str, default="./log/cponet/", help="directory where CPONet checkpoints are stored")
    parser.add_argument("--checkpoint_name", type=str, default="best.pth", help="CPONet checkpoint filename to load from logpath")
    parser.add_argument("--contrastive_ckpt", type=str, default="./log/cpe/best.pth", help="Stage-1 CPE checkpoint with prototypes (for category prediction)",)

    # conditioning / cluster-based normalization
    parser.add_argument("--conditioning_source", type=str, default="auto", choices=["auto", "true", "cluster", "none"], help="how to obtain conditioning id for CPONet",)
    parser.add_argument("--cluster_norm", action="store_true", help="enable per-group (cluster/category) normalization reporting for metrics",)
    parser.add_argument("--cluster_norm_type", type=str, default="minmax", choices=["minmax", "zscore", "mad"], help="normalization method for cluster/category-wise metrics",)
    parser.add_argument("--print_pos_rate", action="store_true", help="print positive rate (fraction of anomalous points) for global / per-category metrics",)

    # dataloader / dataset options
    parser.add_argument("--batch_size", type=int, default=1, help="evaluation batch size (usually keep 1 for variable-size point clouds)")
    parser.add_argument("--num_workers", type=int, default=4, help="number of DataLoader workers")
    parser.add_argument("--data_repeat", type=int, default=1, help="repeat factor for test set (normally 1)")
    parser.add_argument("--mask_num", type=int, default=32, help="kept for compatibility with training-time sphere masks")
    parser.add_argument("--categories", type=str, default="all", help="comma-separated category list or 'all' for all categories")
    parser.add_argument("--category", type=str, default="", help="single category name (used only when --categories is empty)")
    parser.add_argument("--eval_category_only", type=str, default="", help="if non-empty, subset evaluation to this category name (matching dataset paths)")
    parser.add_argument("--pin_memory", action="store_true", help="enable DataLoader pin_memory for faster host-to-GPU transfer")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch_factor when num_workers > 0")
    parser.add_argument("--cache_io", action="store_true", help="enable npz caching of heavy point cloud I/O")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="directory to store cached npz files")
    parser.add_argument("--iec_root", type=str, default="datasets/IEC3DAD", help="root directory of IEC3DAD dataset")
    parser.add_argument("--real3d_train_source", type=str, default="ply", choices=["ply", "train", "train_cut"], help="Real3D normal training source used for CPE/CPONet training (ply / train / train_cut)",)

    # Geo-TTA
    parser.add_argument("--tta_views", type=int, default=3, help="number of additional Geo-TTA views (0 disables TTA)")
    parser.add_argument("--tta_rotate_deg", type=float, default=5.0, help="max random rotation angle (degrees) per axis for TTA")
    parser.add_argument("--tta_scale", type=float, default=0.05, help="isotropic scaling jitter range for TTA (e.g., 0.05 -> [0.95, 1.05])")
    parser.add_argument("--tta_jitter", type=float, default=0.002, help="Gaussian noise std added to coordinates during TTA")
    parser.add_argument("--tta_reduce", type=str, default="mean", choices=["mean", "max"], help="how to aggregate point scores across TTA views (mean or max)",)

    # scoring
    parser.add_argument("--score_method", type=str, default="quantile", choices=["mean", "max", "quantile"], help="object-level score from point scores: mean / max / quantile",)
    parser.add_argument("--score_quantile", type=float, default=0.99, help="quantile used when score_method='quantile'")

    # spatial smoothing for point scores
    parser.add_argument("--smooth_knn", type=int, default=16, help="kNN smoothing for point scores (0 disables)",)

    return parser
