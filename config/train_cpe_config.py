import argparse


def get_parser():
    """Argument parser for Stage-1 CPE training (contrastive pre-training)."""

    parser = argparse.ArgumentParser(description="UCA-3DAL Stage-1: Contrastive Prototype Encoder (CPE)")

    # basic setup
    parser.add_argument("--task", type=str, default="contrastive", help="fixed training task name")
    parser.add_argument("--manual_seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=150, help="number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--gpu_id", type=str, default="0", help="CUDA visible devices")
    parser.add_argument("--logpath", type=str, default="./log/cpe/", help="directory to save logs and ckpts")
    parser.add_argument("--save_freq", type=int, default=100, help="epoch interval for numbered checkpoints")
    parser.add_argument("--pretrain", type=str, default="", help="optional path to existing CPE checkpoint")

    # dataset
    parser.add_argument("--dataset", type=str, default="AnomalyShapeNet", help="{AnomalyShapeNet, Real3D, IEC3DAD}")
    parser.add_argument("--categories", type=str, default="all", help="comma separated category list or 'all' for unified training")
    parser.add_argument("--category", type=str, default="", help="(unused for unified setting; kept for compatibility")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--data_repeat", type=int, default=10, help="repeat factor per epoch")
    parser.add_argument("--mask_num", type=int, default=32, help="kept for compatibility with dataset code")
    parser.add_argument("--voxel_size", type=float, default=0.03, help="quantization size for MinkowskiEngine")

    # dataset roots (for Real3D / IEC3DAD if needed)
    parser.add_argument("--iec_root", type=str, default="", help="root path of IEC3DAD dataset (category/train,test,gt)")
    parser.add_argument("--real3d_train_source", type=str, default="ply", choices=["ply", "train", "train_cut"], help="Real3D training normal source")

    # DataLoader / caching
    parser.add_argument("--pin_memory", action="store_true", help="enable DataLoader pin_memory")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch_factor")
    parser.add_argument("--cache_io", action="store_true", help="cache heavy I/O to npz")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="cache directory")

    # model
    parser.add_argument("--in_channels", type=int, default=3, help="input feature channels")
    parser.add_argument("--out_channels", type=int, default=32, help="backbone feature channels")
    parser.add_argument("--proj_dim", type=int, default=128, help="projection head output dim")

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["Adam", "SGD", "AdamW"], help="optimizer type")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--step_epoch", type=int, default=10, help="LR decay start epoch")
    parser.add_argument("--multiplier", type=float, default=0.5, help="cosine LR floor scale")

    # contrastive / prototype parameters
    parser.add_argument("--temperature", type=float, default=0.07, help="temperature for SupCon / NCE")
    parser.add_argument("--proto_m", type=float, default=0.9, help="EMA momentum for prototypes")
    parser.add_argument("--proto_loss_weight", type=float, default=0.2, help="weight for prototype NCE loss")
    parser.add_argument("--contrastive_best_metric", type=str, default="loss", choices=["loss", "acc"], help="criterion to select best checkpoint")

    return parser