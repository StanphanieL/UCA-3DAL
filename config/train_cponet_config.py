import argparse


def get_parser():
    """Argument parser for Stage-2 CPONet training (offset regression)."""

    parser = argparse.ArgumentParser(description="UCA-3DAL Stage-2: Conditional Point Offset Network (CPONet)")

    # basic setup
    parser.add_argument("--task", type=str, default="train", help="training task name")
    parser.add_argument("--manual_seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--gpu_id", type=str, default="0", help="CUDA visible devices")
    parser.add_argument("--logpath", type=str, default="./log/cponet/", help="directory to save logs and ckpts")
    parser.add_argument("--save_freq", type=int, default=100, help="epoch interval for numbered checkpoints")
    parser.add_argument("--pretrain", type=str, default="", help="optional CPONet checkpoint to resume")

    # dataset
    parser.add_argument("--dataset", type=str, default="AnomalyShapeNet", help="{AnomalyShapeNet, Real3D, IEC3DAD}")
    parser.add_argument("--categories", type=str, default="all", help="comma separated list or 'all'")
    parser.add_argument("--category", type=str, default="", help="(unused for unified setting)")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--data_repeat", type=int, default=10, help="repeat factor per epoch")
    parser.add_argument("--mask_num", type=int, default=32, help="kept for compatibility with dataset code")
    parser.add_argument("--voxel_size", type=float, default=0.03, help="quantization size for MinkowskiEngine")

    # dataset roots
    parser.add_argument("--iec_root", type=str, default="", help="root path of IEC3DAD dataset")
    parser.add_argument("--real3d_train_source", type=str, default="ply", choices=["ply", "train", "train_cut"], help="Real3D training normal source")

    # DataLoader / caching
    parser.add_argument("--pin_memory", action="store_true", help="enable DataLoader pin_memory")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch_factor")
    parser.add_argument("--cache_io", action="store_true", help="cache heavy I/O to npz")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="cache directory")

    # model
    parser.add_argument("--in_channels", type=int, default=3, help="input feature channels")
    parser.add_argument("--out_channels", type=int, default=32, help="backbone feature channels")
    parser.add_argument("--class_embed_dim", type=int, default=32, help="class embedding dimension (0 to disable)")
    parser.add_argument("--conditional_mode", type=str, default="film", choices=["concat", "film"], help="category conditioning mode")

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["Adam", "SGD", "AdamW"], help="optimizer type")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--step_epoch", type=int, default=10, help="LR decay start epoch")
    parser.add_argument("--multiplier", type=float, default=0.5, help="cosine LR floor scale")

    # offset loss
    parser.add_argument("--lambda_dir", type=float, default=1.0, help="weight for direction (cosine) loss term")

    # DPS (Diversified Pseudo-Anomaly Synthesis) parameters for AnomalyShapeNet
    parser.add_argument("--region_anom_prob", type=float, default=0.15, help="prob. to use region-style anomaly")
    parser.add_argument("--region_K_max", type=int, default=2, help="max regions per sample")
    parser.add_argument("--region_area_min", type=float, default=0.15, help="min area fraction per region")
    parser.add_argument("--region_area_max", type=float, default=0.25, help="max area fraction per region")
    parser.add_argument("--region_soft_min", type=float, default=0.05, help="min soft boundary ratio")
    parser.add_argument("--region_soft_max", type=float, default=0.20, help="max soft boundary ratio")
    parser.add_argument("--region_amp_min", type=float, default=0.05, help="min displacement amplitude")
    parser.add_argument("--region_amp_max", type=float, default=0.25, help="max displacement amplitude")
    parser.add_argument("--region_mix_sign_prob", type=float, default=0.8, help="prob. to mix convex/concave signs")

    # warm start from Stage-1 backbone
    parser.add_argument("--contrastive_backbone", type=str, default="", help="path to Stage-1 CPE checkpoint whose backbone will initialize CPONet")

    return parser