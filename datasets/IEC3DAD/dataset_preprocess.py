import os
import glob
import random
import hashlib

import numpy as np
import open3d as o3d
import torch
import MinkowskiEngine as ME
from torch.utils.data import DataLoader

from . import transform as aug_transform


class Dataset:
    """IEC3D-AD dataset wrapper for UCA-3DAL.

    Directory layout (per category):
        <root>/<category>/
            train/*.pcd      # training normal samples
            test/*.pcd       # test samples (good and defective)
            gt/*.txt         # per-point labels for anomalies (x y z label)

    The root directory is specified by ``cfg.iec_root``.
    """

    def __init__(self, cfg):
        self.batch_size = cfg.batch_size
        self.dataset_workers = getattr(cfg, 'num_workers', getattr(cfg, 'num_works', 4))
        self.data_repeat = cfg.data_repeat
        self.voxel_size = cfg.voxel_size
        self.mask_num = cfg.mask_num

        self.pin_memory = getattr(cfg, "pin_memory", False)
        self.prefetch_factor = getattr(cfg, "prefetch_factor", 2)

        self.region_anom_prob = float(getattr(cfg, "region_anom_prob", 0.15))
        self.region_anom_enable = self.region_anom_prob > 0.0
        self.region_K_max = int(getattr(cfg, "region_K_max", 2))
        self.region_area_min = float(getattr(cfg, "region_area_min", 0.05))
        self.region_area_max = float(getattr(cfg, "region_area_max", 0.10))
        self.region_soft_min = float(getattr(cfg, "region_soft_min", 0.05))
        self.region_soft_max = float(getattr(cfg, "region_soft_max", 0.10))
        self.region_amp_min = float(getattr(cfg, "region_amp_min", 0.05))
        self.region_amp_max = float(getattr(cfg, "region_amp_max", 0.10))
        self.region_mix_sign_prob = float(getattr(cfg, "region_mix_sign_prob", 0.2))

        self.cache_io = getattr(cfg, "cache_io", False)
        self.root = getattr(cfg, "iec_root", "datasets/IEC3DAD")
        self.cache_dir = os.path.join(getattr(cfg, "cache_dir", "./cache"), "IEC3DAD")
        if self.cache_io:
            os.makedirs(self.cache_dir, exist_ok=True)

        if not os.path.isdir(self.root):
            raise RuntimeError(f"IEC3DAD root not found: {self.root}")

        all_categories = sorted(
            [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        )
        single_category = getattr(cfg, "category", "")
        if hasattr(cfg, "categories") and cfg.categories:
            if cfg.categories.strip().lower() == "all":
                self.category_list = all_categories
            else:
                requested = [c.strip() for c in cfg.categories.split(",") if c.strip()]
                for c in requested:
                    assert c in all_categories, f"Unknown IEC3DAD category {c}"
                self.category_list = requested
        else:
            assert single_category in all_categories, f"Unknown IEC3DAD category {single_category}"
            self.category_list = [single_category]

        self.cat2id = {c: i for i, c in enumerate(self.category_list)}
        self.num_classes = len(self.category_list)

        # training files: train/*.pcd
        self.train_file_list = [] 
        for c in self.category_list:
            train_dir = os.path.join(self.root, c, "train")
            pattern = os.path.join(train_dir, "*.pcd")
            files = sorted(glob.glob(pattern))
            if len(files) == 0:
                raise RuntimeError(f"[IEC3DAD] No training PCD under {pattern}")
            files = files * self.data_repeat
            self.train_file_list += [(p, self.cat2id[c]) for p in files]

        # test files: test/*.pcd + object-level labels (good/defect)
        self.test_file_list = []  
        self.test_labels = []      # 0=good, 1=defect
        for c in self.category_list:
            test_dir = os.path.join(self.root, c, "test")
            pattern = os.path.join(test_dir, "*.pcd")
            files = sorted(glob.glob(pattern))
            for p in files:
                self.test_file_list.append(p)
                fname = os.path.basename(p).lower()
                label = 0 if "good" in fname else 1
                self.test_labels.append(label)

        # transforms
        self.NormalizeCoord = aug_transform.NormalizeCoord()
        self.CenterShift = aug_transform.CenterShift(apply_z=True)
        self.RandomRotate_z = aug_transform.RandomRotate(angle=[-1, 1], axis="z", center=[0, 0, 0], p=1.0)
        self.RandomRotate_y = aug_transform.RandomRotate(angle=[-1, 1], axis="y", p=1.0)
        self.RandomRotate_x = aug_transform.RandomRotate(angle=[-1, 1], axis="x", p=1.0)
        self.SphereCropMask = aug_transform.SphereCropMask(part_num=self.mask_num)

        # contrastive views
        self.contrast_aug = aug_transform.Compose([
            self.CenterShift,
            self.RandomRotate_z,
            self.RandomRotate_y,
            self.RandomRotate_x,
            aug_transform.RandomScale(0.9, 1.1, p=0.5),
            aug_transform.RandomJitter(sigma=0.005, clip=0.02, p=0.5),
            self.NormalizeCoord,
        ])

        # offset training views
        self.train_aug_compose = aug_transform.Compose([
            self.CenterShift,
            self.RandomRotate_z,
            self.RandomRotate_y,
            self.RandomRotate_x,
            self.NormalizeCoord,
            self.SphereCropMask,
        ])

        # test views
        self.test_aug_compose = aug_transform.Compose([
            self.CenterShift,
            self.NormalizeCoord,
        ])

    def _worker_init_fn_(self, worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        np.random.seed(np_seed)
        random.seed(np_seed)

    def trainLoader(self):
        train_set = list(range(len(self.train_file_list)))
        self.train_data_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            collate_fn=self.trainMerge,
            num_workers=self.dataset_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn_,
            persistent_workers=(self.dataset_workers > 0),
            prefetch_factor=self.prefetch_factor if self.dataset_workers > 0 else None,
        )

    def contrastiveLoader(self):
        train_set = list(range(len(self.train_file_list)))
        self.train_data_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            collate_fn=self.contrastiveMerge,
            num_workers=self.dataset_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn_,
            persistent_workers=(self.dataset_workers > 0),
            prefetch_factor=self.prefetch_factor if self.dataset_workers > 0 else None,
        )

    def testLoader(self):
        test_set = list(range(len(self.test_file_list)))
        self.test_data_loader = DataLoader(
            test_set,
            batch_size=1,
            collate_fn=self.testMerge,
            num_workers=self.dataset_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn_,
            persistent_workers=(self.dataset_workers > 0),
        )

    def _load_pcd_with_normals(self, fn_path, knn=16):
        cache_key = None
        if self.cache_io:
            rel = fn_path.replace("/", "_").replace("\\", "_")
            h = hashlib.md5(rel.encode("utf-8")).hexdigest()[:16]
            cache_key = os.path.join(self.cache_dir, f"pcd_{h}.npz")

        if self.cache_io and cache_key and os.path.isfile(cache_key):
            arr = np.load(cache_key)
            coord = arr["coord"]
            normals = arr["normal"]
            return coord, normals

        pcd = o3d.io.read_point_cloud(fn_path)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
        )
        coord = np.asarray(pcd.points).astype(np.float32)
        normals = np.asarray(pcd.normals).astype(np.float32)

        if self.cache_io and cache_key:
            try:
                np.savez(cache_key, coord=coord, normal=normals)
            except Exception:
                pass

        return coord, normals


    def generate_pseudo_anomaly(self, points, normals, center, distance_to_move=0.08):
        distances_to_center = np.linalg.norm(points - center, axis=1)
        max_distance = np.max(distances_to_center) + 1e-12
        movement_ratios = 1 - (distances_to_center / max_distance)
        dmin, dmax = movement_ratios.min(), movement_ratios.max()
        denom = (dmax - dmin) + 1e-12
        movement_ratios = (movement_ratios - dmin) / denom

        directions = np.ones(points.shape[0])
        movements = movement_ratios * distance_to_move * directions
        new_points = points + np.abs(normals) * movements[:, np.newaxis]
        return new_points

    def generate_region_anomaly(self, xyz: np.ndarray, normals: np.ndarray):
        N = xyz.shape[0]
        if N == 0:
            return xyz.copy(), np.zeros_like(xyz)

        nrm = normals
        if nrm is None or nrm.shape[0] != N:
            nrm = np.zeros_like(xyz, dtype=np.float32)
        else:
            nn = np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8
            nrm = nrm / nn

        K = max(1, int(np.random.randint(1, max(2, self.region_K_max + 1))))
        offset = np.zeros_like(xyz, dtype=np.float32)

        for _ in range(K):
            ci = np.random.randint(0, N)
            c = xyz[ci]
            alpha = float(np.random.uniform(self.region_area_min, self.region_area_max))
            d = np.linalg.norm(xyz - c, axis=1)
            r = np.percentile(d, min(99.0, max(1.0, alpha * 100.0))) + 1e-8
            soft = float(np.random.uniform(self.region_soft_min, self.region_soft_max))
            r_hard = r * (1.0 - soft)
            A = float(np.random.uniform(self.region_amp_min, self.region_amp_max))
            sign = 1.0 if (np.random.rand() < 0.5) else -1.0

            distances_to_center = np.linalg.norm(xyz - c, axis=1)
            max_distance = np.max(distances_to_center) + 1e-12
            movement_ratios = 1 - (distances_to_center / max_distance)
            dmin, dmax = movement_ratios.min(), movement_ratios.max()
            denom = (dmax - dmin) + 1e-12
            movement_ratios = (movement_ratios - dmin) / denom

            w = np.zeros((N,), dtype=np.float32)
            inner = d <= r_hard
            w[inner] = 1.0
            band = (d > r_hard) & (d <= r)
            t = (r - d[band]) / max(1e-6, (r - r_hard))
            w[band] = 0.5 * (1.0 - np.cos(np.clip(t, 0.0, 1.0) * np.pi))

            final_movement_ratios = movement_ratios * w

            if np.random.rand() < self.region_mix_sign_prob:
                cj = xyz[np.random.randint(0, N)]
                dj = np.linalg.norm(xyz - cj, axis=1)
                rj = 0.5 * r
                mask_j = dj <= rj
                sign_map = np.ones((N,), dtype=np.float32) * sign
                sign_map[mask_j] = -sign
            else:
                sign_map = np.ones((N,), dtype=np.float32) * sign

            movements = final_movement_ratios * A * sign_map
            disp = movements[:, None] * np.abs(nrm)
            offset += disp.astype(np.float32)

        new_xyz = xyz + offset
        return new_xyz.astype(np.float32), offset.astype(np.float32)

    def contrastiveMerge(self, id_list):
        file_name = []
        labels = []
        xyz_voxel_1, feat_voxel_1 = [], []
        xyz_voxel_2, feat_voxel_2 = [], []

        for idx in id_list:
            fn_path, cat_id = self.train_file_list[idx]
            file_name.append(fn_path)
            labels.append(cat_id)

            coord, _ = self._load_pcd_with_normals(fn_path)

            p1 = {"coord": coord.copy()}
            p1 = self.contrast_aug(p1)
            xyz1 = p1["coord"].astype(np.float32)

            p2 = {"coord": coord.copy()}
            p2 = self.contrast_aug(p2)
            xyz2 = p2["coord"].astype(np.float32)

            q1, f1, _, _ = ME.utils.sparse_quantize(
                xyz1,
                xyz1,
                quantization_size=self.voxel_size,
                return_index=True,
                return_inverse=True,
            )
            q2, f2, _, _ = ME.utils.sparse_quantize(
                xyz2,
                xyz2,
                quantization_size=self.voxel_size,
                return_index=True,
                return_inverse=True,
            )
            xyz_voxel_1.append(q1)
            feat_voxel_1.append(f1)
            xyz_voxel_2.append(q2)
            feat_voxel_2.append(f2)

        xyz_voxel_1_batch, feat_voxel_1_batch = ME.utils.sparse_collate(xyz_voxel_1, feat_voxel_1)
        xyz_voxel_2_batch, feat_voxel_2_batch = ME.utils.sparse_collate(xyz_voxel_2, feat_voxel_2)
        labels = torch.from_numpy(np.array(labels)).long()
        return {
            "xyz_voxel_view1": xyz_voxel_1_batch,
            "feat_voxel_view1": feat_voxel_1_batch,
            "xyz_voxel_view2": xyz_voxel_2_batch,
            "feat_voxel_view2": feat_voxel_2_batch,
            "labels": labels,
            "fn": file_name,
        }


    def trainMerge(self, id_list):
        file_name = []
        xyz_voxel, feat_voxel = [], []
        xyz_original, xyz_shifted = [], []
        v2p_index_batch = []
        gt_offset_list = []
        category_ids = []

        total_voxel_num = 0
        total_point_num = 0
        batch_count = [0]

        for idx in id_list:
            fn_path, cat_id = self.train_file_list[idx]
            file_name.append(fn_path)
            category_ids.append(cat_id)

            coord, vertex_normals = self._load_pcd_with_normals(fn_path)

            mask = np.ones(coord.shape[0]) * -1
            Point_dict = {"coord": coord, "normal": vertex_normals, "mask": mask}
            Point_dict, centers = self.train_aug_compose(Point_dict)

            xyz = Point_dict["coord"].astype(np.float32)
            normal = Point_dict["normal"].astype(np.float32)
            mask = Point_dict["mask"].astype(np.int32)
            mask[mask == (self.mask_num + 1)] = self.mask_num - 1

            xyz_original.append(torch.from_numpy(xyz))

            use_region = self.region_anom_enable and (random.random() < self.region_anom_prob)
            if use_region:
                new_xyz, gt_offset = self.generate_region_anomaly(xyz, normal)
            else:
                num_shift = 1
                mask_range = np.arange(0, self.mask_num // 2)
                shift_index = np.random.choice(mask_range, num_shift, replace=False)
                mask[np.isin(mask, shift_index)] = -1

                shift_xyz = xyz[mask == -1].copy()
                shift_normal = normal[mask == -1].copy()
                shifted_xyz = self.generate_pseudo_anomaly(
                    shift_xyz,
                    shift_normal,
                    centers[shift_index[0]],
                    distance_to_move=np.random.uniform(0.06, 0.12),
                )

                new_xyz = xyz.copy()
                new_xyz[mask == -1] = shifted_xyz
                gt_offset = new_xyz - xyz

            gt_offset_list.append(torch.from_numpy(gt_offset))
            xyz_shifted.append(torch.from_numpy(new_xyz))

            quantized_coords, feats_all, index, inverse_index = ME.utils.sparse_quantize(
                new_xyz,
                new_xyz,
                quantization_size=self.voxel_size,
                return_index=True,
                return_inverse=True,
            )

            v2p_index = inverse_index + total_voxel_num
            total_voxel_num += index.shape[0]
            total_point_num += inverse_index.shape[0]
            batch_count.append(total_point_num)

            xyz_voxel.append(quantized_coords)
            feat_voxel.append(feats_all)
            v2p_index_batch.append(v2p_index)

        xyz_voxel_batch, feat_voxel_batch = ME.utils.sparse_collate(xyz_voxel, feat_voxel)
        xyz_original = torch.cat(xyz_original, 0).to(torch.float32)
        xyz_shifted = torch.cat(xyz_shifted, 0).to(torch.float32)
        v2p_index_batch = torch.cat(v2p_index_batch, 0).to(torch.int64)
        batch_count = torch.from_numpy(np.array(batch_count))
        batch_offset = torch.cat(gt_offset_list, 0).to(torch.float32)
        return {
            "xyz_voxel": xyz_voxel_batch,
            "feat_voxel": feat_voxel_batch,
            "xyz_original": xyz_original,
            "fn": file_name,
            "v2p_index": v2p_index_batch,
            "xyz_shifted": xyz_shifted,
            "batch_count": batch_count,
            "batch_offset": batch_offset,
            "category_id": torch.tensor(category_ids, dtype=torch.long),
        }


    def testMerge(self, id_list):
        file_name = []
        xyz_voxel, feat_voxel = [], []
        xyz_original = []
        v2p_index_batch = []
        labels = []
        gt_mask_list = []

        total_voxel_num = 0
        total_point_num = 0
        batch_count = [0]

        for idx in id_list:
            fn_path = self.test_file_list[idx]
            file_name.append(fn_path)
            label_pc = int(self.test_labels[idx])
            labels.append(label_pc)

            parts = fn_path.replace("\\", "/").split("/")
            if len(parts) >= 3:
                cat_name = parts[-3]
            else:
                cat_name = self.category_list[0]

            if label_pc == 0:
                pcd = o3d.io.read_point_cloud(fn_path)
                coord = np.asarray(pcd.points).astype(np.float32)
                gt_mask = np.zeros(coord.shape[0], dtype=np.float32)
            else:
                base = os.path.splitext(os.path.basename(fn_path))[0]
                gt_txt = os.path.join(self.root, cat_name, "gt", base + ".txt")
                if not os.path.isfile(gt_txt):
                    pcd = o3d.io.read_point_cloud(fn_path)
                    coord = np.asarray(pcd.points).astype(np.float32)
                    gt_mask = np.zeros(coord.shape[0], dtype=np.float32)
                else:
                    arr = np.loadtxt(gt_txt)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    coord = arr[:, 0:3].astype(np.float32)
                    gt_mask = arr[:, 3].astype(np.float32)

            Point_dict = {"coord": coord}
            Point_dict = self.test_aug_compose(Point_dict)
            xyz = Point_dict["coord"].astype(np.float32)

            quantized_coords, feats_all, index, inverse_index = ME.utils.sparse_quantize(
                xyz,
                xyz,
                quantization_size=self.voxel_size,
                return_index=True,
                return_inverse=True,
            )

            v2p_index = inverse_index + total_voxel_num
            total_voxel_num += index.shape[0]
            total_point_num += inverse_index.shape[0]
            batch_count.append(total_point_num)

            xyz_voxel.append(quantized_coords)
            feat_voxel.append(feats_all)
            xyz_original.append(torch.from_numpy(xyz))
            v2p_index_batch.append(v2p_index)
            gt_mask_list.append(torch.from_numpy(gt_mask.astype(np.float32)))

        xyz_voxel_batch, feat_voxel_batch = ME.utils.sparse_collate(xyz_voxel, feat_voxel)
        xyz_original = torch.cat(xyz_original, 0).to(torch.float32)
        v2p_index_batch = torch.cat(v2p_index_batch, 0).to(torch.int64)
        gt_mask = torch.cat(gt_mask_list, 0).to(torch.float32)
        labels = torch.from_numpy(np.array(labels))
        batch_count = torch.from_numpy(np.array(batch_count))

        return {
            "xyz_voxel": xyz_voxel_batch,
            "feat_voxel": feat_voxel_batch,
            "xyz_original": xyz_original,
            "fn": file_name,
            "v2p_index": v2p_index_batch,
            "labels": labels,
            "batch_count": batch_count,
            "gt_mask": gt_mask,
        }


__all__ = ["Dataset"]
