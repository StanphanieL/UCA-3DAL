# UCA-3DAL

## UCA-3DAL: A Unified Contrastive Framework with Test-Time Adaptation for Robust 3D Anomaly Localization

This folder contains a clean, self-contained implementation of **UCA-3DAL** (CPE + CPONet + Geo-TTA) for 3D point cloud anomaly localization on **AnomalyShapeNet**, **Real3D-AD** and **IEC3D-AD**. It is designed to be uploaded as an independent GitHub repository.

---

## Environments

We recommend Python 3.8 and the following versions :

- Python 3.8
- PyTorch 1.9.0 + CUDA 11.1
- MinkowskiEngine 0.5.4

**Create Conda Environment**

```bash
conda create -n UCA-3DAL python=3.8
conda activate UCA-3DAL
```

**Install PyTorch and MinkowskiEngine**  (adapt paths / CUDA version as needed)

```bash
# PyTorch 1.9.0 with CUDA 11.1
conda install -c pytorch -c nvidia -c conda-forge pytorch=1.9.0 cudatoolkit=11.1 torchvision

# (Linux example) Install MinkowskiEngine 0.5.4 from source
conda install openblas-devel -c anaconda

# Uncomment and adapt if you need to set CUDA_HOME explicitly
# export CUDA_HOME=/usr/local/cuda-11.1

pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
  --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
  --install-option="--blas=openblas"
```

**Install Python dependencies**

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

Assume the project structure after cloning is:

```text
UCA-3DAL
├── datasets
│   ├── AnomalyShapeNet
│   │   └── dataset
│   │       ├── obj
│   │       └── pcd
│   ├── Real3D
│   │   ├── Real3D-AD-PCD
│   │   └── Real3D-AD-PLY
│   └── IEC3DAD
├── network
├── config
├── train_cpe.py
├── train_cponet.py
└── eval.py
```

Place datasets as follows:

- **AnomalyShapeNet**
  - `datasets/AnomalyShapeNet/dataset/obj/<cat>/*.obj`  (templates; filenames contain `template`)
  - `datasets/AnomalyShapeNet/dataset/pcd/<cat>/train/*.pcd`
  - `datasets/AnomalyShapeNet/dataset/pcd/<cat>/test/*.pcd`
  - `datasets/AnomalyShapeNet/dataset/pcd/<cat>/GT/*.txt` (per-point labels)

- **Real3D-AD**
  - `datasets/Real3D/Real3D-AD-PCD/<cat>/{train,train_cut,test,gt}`
  - `datasets/Real3D/Real3D-AD-PLY/<cat>/*.ply` (templates and GOOD scans)

- **IEC3D-AD**
  - `datasets/IEC3DAD/<category>/{train,test,gt}` with `.pcd` and `.txt` files

---

## Evaluation (with pretrained weights)

We provide pretrained checkpoints for quick reproduction of the main results in the following table:

| Dataset | CheckPoint |
|--------|--------------|
| AnomalyShapeNet (CPE + CPONet) | [Google Drive](https://drive.google.com/drive/folders/1zJ0DV-_OmdKnQwRbhluqwNV1jDctALvl?usp=drive_link) |
| IEC3D-AD (CPE + CPONet) | [Google Drive](https://drive.google.com/drive/folders/1lc1-ccDbsz5_3faWxfrmgirMukDyREAv?usp=drive_link) |
| Real3D-AD (CPE + CPONet) | [Google Drive](https://drive.google.com/drive/folders/1mYSfD1wZtKMxjNSMuY1H0EUCgT3jrqRg?usp=sharing) |

Download the corresponding CPE and CPONet weights, place them under some log directory (e.g., `./log/<Dataset>/stage1_CPE/` and `./log/<Dataset>/stage2_CPON/`), and then run `eval.py` as follows.

### Direct evaluation with `eval.py`

Example: evaluate **AnomalyShapeNet** on a subset of categories:

```bash
python eval.py \
  --dataset AnomalyShapeNet \
  --logpath ./log/AnomalyShapeNet/stage2_CPON/ \
  --checkpoint_name best.pth \
  --contrastive_ckpt ./log/AnomalyShapeNet/stage1_CPE/best.pth \
  --categories ashtray0,bottle0,bottle1
```

Use `all` to evaluate all categories in this dataset:

```bash
python eval.py \
  --dataset AnomalyShapeNet \
  --logpath ./log/AnomalyShapeNet/stage2_CPON/ \
  --checkpoint_name best.pth \
  --contrastive_ckpt ./log/AnomalyShapeNet/stage1_CPE/best.pth \
  --categories all
```

For **Real3D-AD** or **IEC3D-AD**, change `--dataset` and (if needed)
`--iec_root` / `--categories` accordingly.

### Per-category evaluation with shell scripts

For convenience, we also provide three shell scripts that loop over all categories and evaluate them one by one.

- **AnomalyShapeNet**: edit `run_eval_AnomalyShapeNet.sh` to set `CKPT_DIR` /
  `CKPT_NAME` / `CPE_CKPT`, then run:

  ```bash
  bash run_eval_AnomalyShapeNet.sh
  ```

- **Real3D-AD**: edit `run_eval_Real3D.sh` to set `CKPT_DIR` / `CKPT_NAME` /
  `CPE_CKPT`, then run:

  ```bash
  bash run_eval_Real3D.sh
  ```

- **IEC3D-AD**: edit `run_eval_IEC3DAD.sh` to set `CKPT_DIR` / `CKPT_NAME` /
  `CPE_CKPT` / `IEC_ROOT`, then run:

  ```bash
  bash run_eval_IEC3DAD.sh
  ```

---

## Training (optional)

If you want to train the models yourself instead of using the pretrained checkpoints, you can use the following commands.

UCA-3DAL is trained in two stages:

1. Stage-1: **CPE** (Contrastive Prototype Encoder)
2. Stage-2: **CPONet** (Conditional Point Offset Network with DPS pseudo anomalies)

### (1) Stage-1: CPE Training

Example: unified CPE training on **AnomalyShapeNet** over all categories:

```bash
python train_cpe.py \
  --dataset AnomalyShapeNet \
  --categories all \
  --logpath ./log/AnomalyShapeNet/stage1_CPE/ \
  --epochs 150 --batch_size 32
```

This will save checkpoints under `./log/AnomalyShapeNet/stage1_CPE/`:

- `latest.pth` – latest checkpoint
- `best.pth`   – best checkpoint (by loss or accuracy, see config)

For **Real3D-AD** and **IEC3D-AD**, change `--dataset` and (optionally) `--categories` / `--iec_root` accordingly.

Key options are defined in `config/train_cpe_config.py`.

### (2) Stage-2: CPONet Training

After Stage-1, initialize CPONet from the CPE backbone and train the conditional offset regressor with diversified pseudo anomalies (region-style + local):

```bash
python train_cponet.py \
  --dataset AnomalyShapeNet \
  --categories all \
  --logpath ./log/AnomalyShapeNet/stage2_CPON/ \
  --contrastive_backbone ./log/AnomalyShapeNet/stage1_CPE/best.pth
```

For **Real3D-AD** and **IEC3D-AD**, similarly by changing `--dataset` and the corresponding log paths.

All CPONet options are defined in `config/train_cponet_config.py`.

---



## Citation

If you find this project helpful for your research, please consider citing the UCA-3DAL paper :

```bibtex
@inproceedings{UCA3DAL,
  title   = {UCA-3DAL: A Unified Contrastive Framework with Test-Time Adaptation for Robust 3D Anomaly Localization},
  author  = {Linna Liu， Hanzhe Liang, Can Gao, Wei Zhuo, Linlin Shen, Jinbao Wang},
  booktitle = {...},
  year    = {2025}
}
```

## License

```
MIT License

Copyright (c) 2025 Linna Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.