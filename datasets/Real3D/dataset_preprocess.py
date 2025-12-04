import math
import glob
import os
import random
import re
import hashlib

import numpy as np
import open3d as o3d
import torch
import MinkowskiEngine as ME
from torch.utils.data import DataLoader

from . import transform as aug_transform


class Dataset:

    def __init__(self, cfg):
        self.batch_size = cfg.batch_size
        self.dataset_workers = getattr(cfg, 'num_workers', getattr(cfg, 'num_works', 4))
        self.data_repeat = cfg.data_repeat
        self.voxel_size = cfg.voxel_size
        self.mask_num = cfg.mask_num

        self.pin_memory = getattr(cfg, 'pin_memory', False)
        self.prefetch_factor = getattr(cfg, 'prefetch_factor', 2)

        self.cache_io = getattr(cfg, 'cache_io', False)
        self.cache_dir = os.path.join(getattr(cfg, 'cache_dir', './cache'), 'Real3D')
        if self.cache_io:
            os.makedirs(self.cache_dir, exist_ok=True)


        self.pcd_root = 'datasets/Real3D/Real3D-AD-PCD'


        self.real3d_train_source = getattr(cfg, 'real3d_train_source', 'ply')

        # region_anom_prob > 0, region-style anomalies are enabled; when it
        # is 0, only legacy local anomalies are used.
        self.region_anom_prob = float(getattr(cfg, 'region_anom_prob', 0.15))
        self.region_anom_enable = self.region_anom_prob > 0.0
        self.region_K_max = int(getattr(cfg, 'region_K_max', 2))
        self.region_area_min = float(getattr(cfg, 'region_area_min', 0.05))
        self.region_area_max = float(getattr(cfg, 'region_area_max', 0.10))
        self.region_soft_min = float(getattr(cfg, 'region_soft_min', 0.05))
        self.region_soft_max = float(getattr(cfg, 'region_soft_max', 0.10))
        self.region_amp_min = float(getattr(cfg, 'region_amp_min', 0.05))
        self.region_amp_max = float(getattr(cfg, 'region_amp_max', 0.10))
        self.region_mix_sign_prob = float(getattr(cfg, 'region_mix_sign_prob', 0.2))

        default_list = ['airplane', 'candybar', 'car', 'chicken', 'diamond', 'duck', 'fish', 'gemstone',
                        'seahorse', 'shell', 'starfish', 'toffees']
        if hasattr(cfg, 'categories') and cfg.categories:
            if cfg.categories.strip().lower() == 'all':
                self.category_list = default_list
            else:
                requested = [c.strip() for c in cfg.categories.split(',') if c.strip()]
                for c in requested:
                    assert c in default_list, f'Unknown category {c} for Real3D'
                self.category_list = requested
        else:
            assert cfg.category in default_list
            self.category_list = [cfg.category]
        self.cat2id = {c: i for i, c in enumerate(self.category_list)}
        self.num_classes = len(self.category_list)

        # train files (source controlled by self.real3d_train_source)
        #   ply       -> use PLY templates as training normals
        #   train     -> use PCD/<cat>/train/*.pcd (360° templates)
        #   train_cut -> use PCD/<cat>/train_cut/*.pcd (single-view training)
        self.train_file_list = []  # list of (path, cat_id)
        if self.real3d_train_source == 'ply':
            for c in self.category_list:
                pattern = f"datasets/Real3D/Real3D-AD-PLY/{c}/*.ply"
                data_list = glob.glob(pattern)
                is_train = re.compile(r'template')
                train_files = list(filter(is_train.search, data_list))
                train_files.sort()
                train_files = train_files * self.data_repeat
                if len(train_files) == 0:
                    raise RuntimeError(f"[Real3D] No training templates found. Searched pattern={pattern} and filtered by 'template'. Category={c}")
                self.train_file_list += [(p, self.cat2id[c]) for p in train_files]
        elif self.real3d_train_source == 'train':
            for c in self.category_list:
                train_dir = os.path.join(self.pcd_root, c, 'train')
                pattern = os.path.join(train_dir, '*.pcd')
                files = sorted(glob.glob(pattern))
                if len(files) == 0:
                    raise RuntimeError(f"[Real3D] No train PCD found under {pattern}")
                files = files * self.data_repeat
                self.train_file_list += [(p, self.cat2id[c]) for p in files]
        elif self.real3d_train_source == 'train_cut':
            for c in self.category_list:
                train_cut_dir = os.path.join(self.pcd_root, c, 'train_cut')
                pattern = os.path.join(train_cut_dir, '*.pcd')
                files = sorted(glob.glob(pattern))
                if len(files) == 0:
                    raise RuntimeError(f"[Real3D] No train_cut PCD found under {pattern}")
                files = files * self.data_repeat
                self.train_file_list += [(p, self.cat2id[c]) for p in files]
        else:
            raise RuntimeError(f"[Real3D] Unknown real3d_train_source: {self.real3d_train_source}")

        self.test_file_list = []
        for c in self.category_list:
            test_files = glob.glob(f"{self.pcd_root}/{c}/test/*.pcd")
            test_files.sort()
            self.test_file_list += test_files

        self.NormalizeCoord = aug_transform.NormalizeCoord()
        self.CenterShift = aug_transform.CenterShift(apply_z=True)
        self.RandomRotate_z = aug_transform.RandomRotate(angle=[-1, 1], axis="z", center=[0, 0, 0], p=1.0)
        self.RandomRotate_y = aug_transform.RandomRotate(angle=[-1, 1], axis="y", p=1.0)
        self.RandomRotate_x = aug_transform.RandomRotate(angle=[-1, 1], axis="x", p=1.0)
        self.SphereCropMask = aug_transform.SphereCropMask(part_num=self.mask_num)

        self.train_aug_compose = aug_transform.Compose([
            self.CenterShift,
            self.RandomRotate_z,
            self.RandomRotate_y,
            self.RandomRotate_x,
            self.NormalizeCoord,
            self.SphereCropMask,
        ])
        self.contrast_aug = aug_transform.Compose([
            self.CenterShift,
            self.RandomRotate_z,
            self.RandomRotate_y,
            self.RandomRotate_x,
            aug_transform.RandomScale(0.9, 1.1, p=0.5),
            aug_transform.RandomJitter(sigma=0.005, clip=0.02, p=0.5),
            self.NormalizeCoord,
        ])

        self.test_aug_compose = aug_transform.Compose([self.CenterShift, self.NormalizeCoord])

    def trainLoader(self):
        train_set = list(range(len(self.train_file_list)))
        self.train_data_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            collate_fn=self.trainMerge,
            num_workers=self.dataset_workers,
            shuffle=True,
            sampler=None,
            drop_last=True,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn_,
            persistent_workers=(self.dataset_workers > 0),
            prefetch_factor=self.prefetch_factor if self.dataset_workers > 0 else None,
        )

    def contrastiveLoader(self):
        # build file list for contrastive stage from training files
        self._contrastive_files = self.train_file_list 
        train_set = list(range(len(self._contrastive_files)))
        self.train_data_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            collate_fn=self.contrastiveMerge,
            num_workers=self.dataset_workers,
            shuffle=True,
            sampler=None,
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
            sampler=None,
            drop_last=False,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn_,
            persistent_workers=(self.dataset_workers > 0),
            prefetch_factor=self.prefetch_factor if self.dataset_workers > 0 else None,
        )

    def _worker_init_fn_(self, worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        np.random.seed(np_seed)
        random.seed(np_seed)

    def _load_pcd_with_normals(self, fn_path, knn=16):
        """Load PCD and estimate normals, with optional npz caching."""
        cache_key = None
        if self.cache_io:
            rel = fn_path.replace('/', '_').replace('\\', '_')
            h = hashlib.md5(rel.encode('utf-8')).hexdigest()[:16]
            cache_key = os.path.join(self.cache_dir, f'pcd_{h}.npz')
        if self.cache_io and cache_key and os.path.isfile(cache_key):
            arr = np.load(cache_key)
            coord = arr['coord']
            normals = arr['normal']
            return coord, normals
        # read PCD and estimate normals
        pcd = o3d.io.read_point_cloud(fn_path)
        try:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=int(knn)))
        except Exception:
            coord = np.asarray(pcd.points).astype(np.float32)
            normals = np.zeros_like(coord, dtype=np.float32)
        else:
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
        """
            Region-style large-area anomalies with soft boundary.
        """
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

            if np.random.rand() < getattr(self, 'region_mix_sign_prob', 0.2):
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
        xyz_voxel_1 = []
        feat_voxel_1 = []
        xyz_voxel_2 = []
        feat_voxel_2 = []

        for idx in id_list:
            fn_path, cat_id = self._contrastive_files[idx]
            file_name.append(fn_path)
            labels.append(cat_id)

            if fn_path.lower().endswith('.pcd'):
                coord, _ = self._load_pcd_with_normals(fn_path)
            else:
                try:
                    cache_key = None
                    if self.cache_io:
                        rel = fn_path.replace('/', '_').replace('\\', '_')
                        h = hashlib.md5(rel.encode('utf-8')).hexdigest()[:16]
                        cache_key = os.path.join(self.cache_dir, f'mesh_{h}.npz')
                    if self.cache_io and cache_key and os.path.isfile(cache_key):
                        arr = np.load(cache_key)
                        coord = arr['coord']
                    else:
                        obj = o3d.io.read_triangle_mesh(fn_path)
                        obj.compute_vertex_normals()
                        coord = np.asarray(obj.vertices)
                        if self.cache_io and cache_key:
                            try:
                                np.savez(cache_key, coord=coord.astype(np.float32))
                            except Exception:
                                pass
                except Exception as e:
                    pcd = o3d.io.read_point_cloud(fn_path)
                    coord = np.asarray(pcd.points)

            Point_dict_1 = {'coord': coord.copy()}
            Point_dict_1 = self.contrast_aug(Point_dict_1)
            xyz1 = Point_dict_1['coord'].astype(np.float32)

            Point_dict_2 = {'coord': coord.copy()}
            Point_dict_2 = self.contrast_aug(Point_dict_2)
            xyz2 = Point_dict_2['coord'].astype(np.float32)

            q1, f1, _, _ = ME.utils.sparse_quantize(
                xyz1, xyz1, quantization_size=self.voxel_size, return_index=True, return_inverse=True
            )
            xyz_voxel_1.append(q1)
            feat_voxel_1.append(f1)

            q2, f2, _, _ = ME.utils.sparse_quantize(
                xyz2, xyz2, quantization_size=self.voxel_size, return_index=True, return_inverse=True
            )
            xyz_voxel_2.append(q2)
            feat_voxel_2.append(f2)

        xyz_voxel_1_batch, feat_voxel_1_batch = ME.utils.sparse_collate(xyz_voxel_1, feat_voxel_1)
        xyz_voxel_2_batch, feat_voxel_2_batch = ME.utils.sparse_collate(xyz_voxel_2, feat_voxel_2)
        labels = torch.from_numpy(np.array(labels)).long()
        return {
            'xyz_voxel_view1': xyz_voxel_1_batch,
            'feat_voxel_view1': feat_voxel_1_batch,
            'xyz_voxel_view2': xyz_voxel_2_batch,
            'feat_voxel_view2': feat_voxel_2_batch,
            'labels': labels,
            'fn': file_name,
        }


    def trainMerge(self, id_list):
        file_name = []
        xyz_voxel = []
        feat_voxel = []
        xyz_original = []
        v2p_index_batch = []
        gt_offset_list = []
        xyz_shifted = []
        category_ids = []
        total_voxel_num = 0
        batch_count = [0]
        total_point_num = 0

        for idx in id_list:
            fn_path, cat_id = self.train_file_list[idx]
            file_name.append(fn_path)
            category_ids.append(cat_id)

            if fn_path.lower().endswith('.pcd'):
                coord, vertex_normals = self._load_pcd_with_normals(fn_path)
            else:
                cache_key = None
                if self.cache_io:
                    rel = fn_path.replace('/', '_').replace('\\', '_')
                    h = hashlib.md5(rel.encode('utf-8')).hexdigest()[:16]
                    cache_key = os.path.join(self.cache_dir, f'mesh_{h}.npz')
                if self.cache_io and cache_key and os.path.isfile(cache_key):
                    arr = np.load(cache_key)
                    coord = arr['coord']
                    vertex_normals = arr['normal'] if 'normal' in arr.files else None
                    if vertex_normals is None:
                        obj = o3d.io.read_triangle_mesh(fn_path)
                        obj.compute_vertex_normals()
                        vertex_normals = np.asarray(obj.vertex_normals)
                else:
                    obj = o3d.io.read_triangle_mesh(fn_path)
                    obj.compute_vertex_normals()
                    coord = np.asarray(obj.vertices)
                    vertex_normals = np.asarray(obj.vertex_normals)
                    if self.cache_io and cache_key:
                        try:
                            np.savez(cache_key, coord=coord.astype(np.float32), normal=vertex_normals.astype(np.float32))
                        except Exception:
                            pass

            mask = np.ones(coord.shape[0]) * -1
            Point_dict = {'coord': coord, 'normal': vertex_normals, 'mask': mask}
            Point_dict, centers = self.train_aug_compose(Point_dict)
            xyz = Point_dict['coord'].astype(np.float32)
            normal = Point_dict['normal'].astype(np.float32)
            mask = Point_dict['mask'].astype(np.int32)
            mask[mask == (self.mask_num + 1)] = self.mask_num - 1

            xyz_original.append(torch.from_numpy(xyz))

            use_region = getattr(self, 'region_anom_enable', False) and (random.random() < getattr(self, 'region_anom_prob', 0.15))
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
            total_voxel_num = total_voxel_num + index.shape[0]

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
        return {'xyz_voxel': xyz_voxel_batch, 'feat_voxel': feat_voxel_batch, 'xyz_original': xyz_original,
                'fn': file_name, 'v2p_index': v2p_index_batch, 'xyz_shifted': xyz_shifted, 'batch_count': batch_count, 'batch_offset': batch_offset, 'category_id': torch.tensor(category_ids, dtype=torch.long)}

    def testMerge(self, id_list):
        file_name = []
        xyz_voxel = []
        feat_voxel = []
        xyz_original = []
        v2p_index_batch = []
        labels = []

        total_voxel_num = 0
        total_point_num = 0
        batch_count = [0]
        for idx in id_list:
            fn_path = self.test_file_list[idx]
            file_name.append(self.test_file_list[idx])

            c = fn_path.split('/')[-3]

            if 'good' in fn_path:
                cache_key = None
                if self.cache_io:
                    rel = fn_path.replace('/', '_').replace('\\', '_')
                    h = hashlib.md5(rel.encode('utf-8')).hexdigest()[:16]
                    cache_key = os.path.join(self.cache_dir, f'pcd_{h}.npz')
                if self.cache_io and cache_key and os.path.isfile(cache_key):
                    arr = np.load(cache_key)
                    coord = arr['xyz']
                else:
                    pcd = o3d.io.read_point_cloud(fn_path)
                    coord = np.asarray(pcd.points)
                    if self.cache_io and cache_key:
                        try:
                            np.savez(cache_key, xyz=coord.astype(np.float32))
                        except Exception:
                            pass
            else:
                sample_name = fn_path.split('/')[-1].split('.')[0]
                gt_mask_path = f'{self.pcd_root}/{c}/gt/'
                gt_file = gt_mask_path + sample_name + '.txt'
                cache_key = None
                if self.cache_io:
                    rel = gt_file.replace('/', '_').replace('\\', '_')
                    h = hashlib.md5(rel.encode('utf-8')).hexdigest()[:16]
                    cache_key = os.path.join(self.cache_dir, f'gt_{h}.npz')
                if self.cache_io and cache_key and os.path.isfile(cache_key):
                    arr = np.load(cache_key)
                    coord = arr['xyz']
                else:
                    coord = np.loadtxt(gt_file)[:, 0:3]
                    if self.cache_io and cache_key:
                        try:
                            np.savez(cache_key, xyz=coord.astype(np.float32))
                        except Exception:
                            pass

            Point_dict = {'coord': coord}
            Point_dict = self.test_aug_compose(Point_dict)

            xyz = Point_dict['coord'].astype(np.float32)

            quantized_coords, feats_all, index, inverse_index = ME.utils.sparse_quantize(xyz, xyz,
                                                                                        quantization_size=self.voxel_size,
                                                                                        return_index=True,
                                                                                        return_inverse=True)

            v2p_index = inverse_index + total_voxel_num
            total_voxel_num = total_voxel_num + index.shape[0]
            total_point_num += inverse_index.shape[0]
            batch_count.append(total_point_num)

            xyz_voxel.append(quantized_coords)
            feat_voxel.append(feats_all)
            xyz_original.append(torch.from_numpy(xyz))
            v2p_index_batch.append(v2p_index)
            if 'good' in fn_path:
                labels.append(0)
            else:
                labels.append(1)

        xyz_voxel_batch, feat_voxel_batch = ME.utils.sparse_collate(xyz_voxel, feat_voxel)
        xyz_original = torch.cat(xyz_original, 0).to(torch.float32)
        v2p_index_batch = torch.cat(v2p_index_batch, 0).to(torch.int64)
        labels = torch.from_numpy(np.array(labels))
        batch_count = torch.from_numpy(np.array(batch_count))
        return {'xyz_voxel': xyz_voxel_batch, 'feat_voxel': feat_voxel_batch, 'xyz_original': xyz_original,
                'fn': file_name, 'v2p_index': v2p_index_batch, 'labels': labels, 'batch_count': batch_count}


__all__ = ["Dataset"]
