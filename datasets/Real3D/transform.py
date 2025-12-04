import random
import numpy as np


class NormalizeCoord(object):
    def __call__(self, data_dict):
        if "coord" in data_dict:
            centroid = np.mean(data_dict["coord"], axis=0)
            data_dict["coord"] -= centroid
            m = np.max(np.sqrt(np.sum(data_dict["coord"] ** 2, axis=1)))
            data_dict["coord"] = data_dict["coord"] / max(m, 1e-12)
        return data_dict


class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict:
            x_min, y_min, z_min = data_dict["coord"].min(axis=0)
            x_max, y_max, _ = data_dict["coord"].max(axis=0)
            if self.apply_z:
                shift = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, z_min]
            else:
                shift = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, 0.0]
            data_dict["coord"] -= shift
        return data_dict


class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1.0
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict:
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        return data_dict


class RandomJitter(object):
    def __init__(self, sigma=0.005, clip=0.02, p=0.5):
        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        if "coord" in data_dict:
            jitter = np.clip(self.sigma * np.random.randn(*data_dict["coord"].shape), -self.clip, self.clip)
            data_dict["coord"] = data_dict["coord"] + jitter
        return data_dict


class RandomScale(object):
    def __init__(self, scale_low=0.9, scale_high=1.1, p=0.5):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.p = p

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        if "coord" in data_dict:
            s = np.random.uniform(self.scale_low, self.scale_high)
            data_dict["coord"] = data_dict["coord"] * s
        return data_dict


class SphereCropMask(object):
    def __init__(self, part_num=64):
        self.part_num = part_num

    def __call__(self, data_dict):
        assert "coord" in data_dict and "mask" in data_dict
        part_point_num = data_dict["coord"].shape[0] // self.part_num
        centers = []
        for p_i in range(self.part_num):
            null_mask = np.argwhere(data_dict["mask"] == -1)
            center = data_dict["coord"][null_mask[np.random.randint(null_mask.shape[0])]]
            idx_crop = np.argsort(
                np.sum(np.square(data_dict["coord"][null_mask.reshape(-1)] - center), 1)
            )[:part_point_num]
            data_dict["mask"][null_mask[idx_crop]] = p_i
            centers.append(center)
        data_dict["mask"][data_dict["mask"] == -1] = self.part_num + 1
        return data_dict, centers


class Compose(object):
    def __init__(self, aug_list=None):
        self.transforms = list(aug_list or [])

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict