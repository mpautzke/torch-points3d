import os
import os.path as osp
from itertools import repeat, product
import numpy as np
import h5py
import torch
import random
import glob
from plyfile import PlyData, PlyElement
from torch_geometric.data import InMemoryDataset, Data, extract_zip, Dataset
from torch_geometric.data.dataset import files_exist
from torch_geometric.data import DataLoader
from torch_geometric.datasets import S3DIS as S3DIS1x1
import torch_geometric.transforms as T
import logging
from sklearn.neighbors import NearestNeighbors, KDTree
from tqdm.auto import tqdm as tq
import csv
import pandas as pd
import pickle
import gdown
import shutil
from torch_geometric.nn import knn_interpolate
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_geometric.io import read_txt_array
# from torch_points3d.datasets.segmentation.nexplore import shift_and_quantize, OBJECT_COLOR

from torch_points3d.datasets.samplers import BalancedRandomSampler
import torch_points3d.core.data_transform as cT
from torch_points3d.datasets.base_dataset import BaseDataset

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)



# def to_ply(pos, label, file):
#     assert len(label.shape) == 1
#     assert pos.shape[0] == label.shape[0]
#     pos = np.asarray(pos)
#     colors = OBJECT_COLOR[np.asarray(label)]
#     ply_array = np.ones(
#         pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
#     )
#     ply_array["x"] = pos[:, 0]
#     ply_array["y"] = pos[:, 1]
#     ply_array["z"] = pos[:, 2]
#     ply_array["red"] = colors[:, 0]
#     ply_array["green"] = colors[:, 1]
#     ply_array["blue"] = colors[:, 2]
#     el = PlyElement.describe(ply_array, "S3DIS")
#     PlyData([el], byte_order=">").write(file)

################################### UTILS #######################################

# def object_name_to_label(object_class):
#     """convert from object name in S3DIS to an int"""
#     object_label = OBJECT_LABEL.get(object_class.lower(), OBJECT_LABEL["other"])
#     return object_label

# def read_s3dis_format(train_file, shift_quantize = False, verbose = False, include_labels = True):
#     """extract data from a room folder"""
#     raw_path = osp.join(train_file)
#     room_ver = pd.read_csv(raw_path, sep=" ", header=None).values
#
#     xyz = np.ascontiguousarray(room_ver[:, 0:3], dtype="float64")
#
#     shift_vector = np.array([0,0,0])
#     if shift_quantize:
#         xyz, shift_vector = shift_and_quantize(xyz)
#
#     # outliers = remove_outliers(xyz)
#
#     try:
#         rgb = np.ascontiguousarray(room_ver[:, 3:6], dtype="uint8")
#     except ValueError:
#         rgb = np.zeros((room_ver.shape[0], 3), dtype="uint8")
#         log.warning("WARN - corrupted rgb data for file %s" % raw_path)
#
#     n_ver = len(room_ver)
#     semantic_labels = np.zeros((n_ver,), dtype="int64")
#     instance_labels = np.zeros((n_ver,), dtype="int64")
#
#     if not include_labels:
#         return torch.from_numpy(xyz), torch.from_numpy(rgb), torch.from_numpy(semantic_labels)
#
#     train_file_dir = osp.split(train_file)[0]
#
#     nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(xyz)
#     objects = glob.glob(osp.join(train_file_dir, "annotations/*.txt"))
#     i_object = 1
#     for single_object in objects:
#         object_name = os.path.splitext(os.path.basename(single_object))[0]
#         if object_name == "remainder":  # expand to a list
#             continue
#         if verbose:
#             log.debug("adding object " + str(i_object) + " : " + object_name)
#         object_class = object_name.split("_")[0]
#         object_label = object_name_to_label(object_class)
#         obj_ver = pd.read_csv(single_object, sep=" ", header=None).values
#         obj_xyz = np.ascontiguousarray(obj_ver[:, 0:3], dtype="float64")
#         obj_xyz, _ = shift_and_quantize(obj_xyz, manual_shift=shift_vector)
#         _, obj_ind = nn.kneighbors(obj_xyz)
#         semantic_labels[obj_ind] = object_label
#         instance_labels[obj_ind] = i_object
#         i_object = i_object + 1
#
#     return (
#         torch.from_numpy(xyz),
#         torch.from_numpy(rgb),
#         torch.from_numpy(semantic_labels),  # actual label
#     )
#
# #TODO Would need to modify original file if we remove outliers as we depend on original index
# def remove_outliers(xyz, x_c = 5, y_c = 5, z_c = 20):
#     # a = np.array(x)
#     x_index = outlier_index(xyz[:,0], x_c)
#     y_index = outlier_index(xyz[:,1], y_c)
#     z_index = outlier_index(xyz[:,2], z_c)
#
#     xy_index = np.union1d(x_index, y_index)
#     xyz_index = np.union1d(xy_index, z_index)
#
#     return xyz_index
#
# def outlier_index(a, outlierConstant):
#     upper_quartile = np.percentile(a, 75)
#     lower_quartile = np.percentile(a, 25)
#     IQR = (upper_quartile - lower_quartile) * outlierConstant
#     quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
#     results = np.where((a <= quartileSet[0]) | (a >= quartileSet[1]))
#     return results

class NexploreS3DISOriginalFused(Dataset):
    def __init__(
        self,
        root,
        fname,
        split="train",
        radius=20,
        sample_per_epoch=-1,
        transform=None,
        pre_transform=None,
        pre_collate_transform=None,
        pre_filter=None,
        verbose=False,
        debug=False,
        include_labels=True
    ):
        self.num_classes = S3DIS_NUM_CLASSES
        self.object_label = INV_OBJECT_LABEL
        self.object_label_map = INV_OBJECT_LABEL_MAP
        self.include_labels = include_labels
        self.fname = fname
        self.path = os.path.join(root, fname)
        self.transform = transform
        self.pre_collate_transform = pre_collate_transform
        self.verbose = verbose
        self.debug = debug
        self._split = split
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self.root = root
        super(NexploreS3DISOriginalFused, self).__init__(root, transform, pre_transform, pre_filter)

        self._load_data(self.processed_file_names[0])
        self.raw_test_data = self.read_raw_data(include_labels=self.include_labels) #TODO do not hold in mem
        self.shifted_test_data = self.read_raw_data(shift_quantize=True, include_labels=self.include_labels, remove_dups=True)

    def read_raw_data(self, shift_quantize=False, include_labels=True, remove_dups=False):
        xyz, rgb, semantic_labels = self.read_s3dis_format(
            self.path, shift_quantize=shift_quantize, include_labels=include_labels, remove_dups=remove_dups
        )

        return Data(pos=xyz.numpy(), y=semantic_labels.numpy(), rgb=rgb.numpy())

    @property
    def center_labels(self):
        if hasattr(self.data, "center_label"):
            return self.data.center_label
        else:
            return None

    @property
    def raw_file_names(self):
        return [self.path]

    @property
    def pre_processed_path(self):
        pre_processed_file_names = "preprocessed.pt"
        return os.path.join(self.processed_dir, pre_processed_file_names)

    @property
    def transformed_path(self):
        transformed_file_names = "transformed.pt"
        return os.path.join(self.processed_dir, transformed_file_names)

    @property
    def raw_areas_paths(self):
        return [self.path]

    #used for processed_paths
    @property
    def processed_file_names(self):
        return (
            [os.path.join(self.processed_dir, f"{self.fname}.pt")]
        )

    @property
    def raw_test_data(self):
        return self._raw_test_data

    @raw_test_data.setter
    def raw_test_data(self, value):
        self._raw_test_data = value

    def download(self):
        raw_folders = os.listdir(self.raw_dir)
        if len(raw_folders) == 0:
            # if not os.path.exists(osp.join(self.root, self.zip_name)):
            log.info("WARNING: You need to download data from sharepoint and put it in the data root folder")

    def process(self):
        if not os.path.exists(self.processed_file_names[0]):
            xyz, rgb, semantic_labels = self.read_s3dis_format(
                self.path, shift_quantize=True, include_labels=False, remove_dups=True
            )

            rgb_norm = rgb.float() / 255.0
            data = Data(pos=xyz, y=semantic_labels, rgb=rgb_norm)

            if self.pre_collate_transform:
                log.info("pre_collate_transform ...")
                log.info(self.pre_collate_transform)
                data = self.pre_collate_transform([data])

            grid_sampler = cT.GridSphereSampling(self._radius, self._radius, center=False)
            data = grid_sampler(data)
            data = [d for d in data if len(d.origin_id) > 10]

            self._save_data(data)

    def _load_data(self, path):
        self.data, self.slices = torch.load(path)

    def read_s3dis_format(self, train_file, shift_quantize=False, verbose=False, include_labels=True, remove_dups=False):
        """extract data from a room folder"""
        raw_path = osp.join(train_file)
        room_ver = pd.read_csv(raw_path, sep=" ", header=None).values

        xyz = np.ascontiguousarray(room_ver[:, 0:3], dtype="float64")

        if remove_dups:
            total_count = len(xyz)
            unq_index = np.unique(xyz, axis=0, return_index=True)[1]
            # keeping the same sort order runs much faster through knn
            unq_index = np.sort(unq_index, axis=0)
            xyz = np.ascontiguousarray(room_ver[unq_index, 0:3], dtype="float64")
            unq_count = len(xyz)
            print(f"Removed {total_count - unq_count} duplicate points in {raw_path}")

        shift_vector = np.array([0, 0, 0])
        if shift_quantize:
            xyz, shift_vector = self.shift_and_quantize(xyz)

        # outliers = remove_outliers(xyz)

        try:
            if remove_dups:
                rgb = np.ascontiguousarray(room_ver[unq_index, 3:6], dtype="uint8")
            else:
                rgb = np.ascontiguousarray(room_ver[:, 3:6], dtype="uint8")
        except ValueError:
            rgb = np.zeros((xyz.shape[0], 3), dtype="uint8")
            log.warning("WARN - corrupted rgb data for file %s" % raw_path)

        n_ver = len(xyz)
        semantic_labels = np.zeros((n_ver,), dtype="int64")
        instance_labels = np.zeros((n_ver,), dtype="int64")

        if not include_labels:
            return torch.from_numpy(xyz), torch.from_numpy(rgb), torch.from_numpy(semantic_labels)

        train_file_dir = osp.split(train_file)[0]

        nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(xyz)
        objects = glob.glob(osp.join(train_file_dir, "annotations/*.txt"))
        i_object = 1
        for single_object in objects:
            object_name = os.path.splitext(os.path.basename(single_object))[0]
            if object_name == "remainder":  # expand to a list
                continue
            if verbose:
                log.debug("adding object " + str(i_object) + " : " + object_name)
            object_class = object_name.split("_")[0]
            object_label = self.object_name_to_label(object_class)
            obj_ver = pd.read_csv(single_object, sep=" ", header=None).values
            obj_xyz = np.ascontiguousarray(obj_ver[:, 0:3], dtype="float64")
            if remove_dups:
                obj_unq_index = np.unique(obj_xyz, axis=0, return_index=True)[1]
                obj_unq_index = np.sort(obj_unq_index, axis=0)
                obj_xyz = obj_xyz[obj_unq_index]
            if shift_quantize:
                obj_xyz, _ = self.shift_and_quantize(obj_xyz, manual_shift=shift_vector)
            _, obj_ind = nn.kneighbors(obj_xyz)
            n = obj_ind[:, [0]]
            semantic_labels[n] = object_label
            instance_labels[n] = i_object
            i_object = i_object + 1

        return (
            torch.from_numpy(xyz),
            torch.from_numpy(rgb),
            torch.from_numpy(semantic_labels),  # actual label
        )

    def object_name_to_label(self, object_class):
        """convert from object name in S3DIS to an int"""
        log.info(self.object_label_map.keys())
        for key in self.object_label_map.keys():
            if object_class.lower() in self.object_label_map[key]:
                return key
        # 0 reserved for other
        return 0

    def shift_and_quantize(self, xyz, qtype=np.float32, manual_shift=None):
        if manual_shift is None:
            shift_threshold = 100
            x = xyz[:, 0].min()
            y = xyz[:, 1].min()
            z = xyz[:, 2].min()

            shift_x = self.calc_shift(x, shift_threshold)
            shift_y = self.calc_shift(y, shift_threshold)
            shift_z = self.calc_shift(z, shift_threshold)

            shift = np.array([shift_x, shift_y, shift_z])
        else:
            shift = np.array(manual_shift)
        scale = 1

        out = np.add(xyz, shift)
        out = np.multiply(out, scale)

        return np.ascontiguousarray(out.astype(qtype)), shift

    def calc_shift(self, number, threshold):
        shift = 0
        if number / threshold > 1:
            r = int(number % threshold)
            shift = int(number - r) * -1.00
        return shift


class NexploreS3DISSphere(NexploreS3DISOriginalFused):
    """ Small variation of S3DISOriginalFused that allows random sampling of spheres
    within an Area during training and validation. Spheres have a radius of 2m. If sample_per_epoch is not specified, spheres
    are taken on a 2m grid.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    test_area: int
        number between 1 and 6 that denotes the area used for testing
    train: bool
        Is this a train split or not
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    sample_per_epoch
        Number of spheres that are randomly sampled at each epoch (-1 for fixed grid)
    radius
        radius of each sphere
    pre_transform
    transform
    pre_filter
    """

    def __init__(self, root, *args, **kwargs):
        self._spheres = None
        super().__init__(root, *args, **kwargs)

    def __len__(self):
        return len(self._spheres)

    def get(self, idx):
        return self._spheres[idx].clone()

    def process(self):  # We have to include this method, otherwise the parent class skips processing
        super().process()

    def download(self):  # We have to include this method, otherwise the parent class skips download
        super().download()

    def _save_data(self, test_data_list):
        torch.save(test_data_list, self.processed_file_names[0])

    def _load_data(self, path):
        self._spheres = torch.load(path)

class NexploreS3DISFusedForwardDataset(BaseDataset):

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        dataset_cls = NexploreS3DISSphere

        if self.dataset_opt.object_labels is None:
            self.dataset_opt.object_labels = self.dataset_opt.fallback_object_labels
        if self.dataset_opt.object_labels_map is None:
            self.dataset_opt.object_labels_map = self.dataset_opt.fallback_object_labels_map

        global INV_OBJECT_LABEL
        temp_dict = {}
        for index, label in enumerate(self.dataset_opt.object_labels):
            temp_dict[index] = label

        INV_OBJECT_LABEL = temp_dict
        self.INV_OBJECT_LABEL = INV_OBJECT_LABEL

        global INV_OBJECT_LABEL_MAP
        temp_dict = {}
        for index, label in enumerate(self.dataset_opt.object_labels_map):
            temp_dict[index] = label

        INV_OBJECT_LABEL_MAP = temp_dict
        self.INV_OBJECT_LABEL_MAP = INV_OBJECT_LABEL_MAP

        global S3DIS_NUM_CLASSES
        S3DIS_NUM_CLASSES = len(INV_OBJECT_LABEL.keys())

        self.test_dataset = dataset_cls(
            dataset_opt.dataroot,
            fname=dataset_opt.dataset_name,
            sample_per_epoch=dataset_opt.samples,
            split="test",
            radius=dataset_opt.radius,
            pre_collate_transform=self.pre_collate_transform,
            transform=self.test_transform,
            include_labels=dataset_opt.include_labels
        )

        if dataset_opt.class_weight_method:
            self.add_weights(class_weight_method=dataset_opt.class_weight_method)

        self.softmax = torch.nn.Softmax(dim=1)
        self.confidence_threshold = dataset_opt.confidence_threshold
        self.prediction_selection_mode = PredictionTracker.MODE[dataset_opt.prediction_selection_mode.lower()]

    @property
    def test_data(self):
        return self.test_dataset[0].raw_test_data

    @staticmethod
    def to_ply(pos, label, file):
        """ Allows to save s3dis predictions to disk using s3dis color scheme

        Parameters
        ----------
        pos : torch.Tensor
            tensor that contains the positions of the points
        label : torch.Tensor
            predicted label
        file : string
            Save location
        """
        pass
        # to_ply(pos, label, file)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        from torch_points3d.metrics.s3dis_tracker import S3DISTracker

        return S3DISTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)

    def predict_original_samples(self, batch, conv_type, output, results={}):
        num_sample = BaseDataset.get_num_samples(batch, conv_type)
        if conv_type == "DENSE":
            output = output.reshape(num_sample, -1, output.shape[-1])  # [B,N,L]

        setattr(batch, "_pred", output)
        for b in range(num_sample):
            predicted = BaseDataset.get_sample(batch, "_pred", b, conv_type).reshape(-1, output.shape[-1])
            # labels = np.zeros((predicted.shape[0]), dtype=np.int64)
            origindid = BaseDataset.get_sample(batch, SaveOriginalPosId.KEY, b, conv_type).cpu().numpy()

            predicted = self.softmax(predicted)
            confidence, prediction = predicted.max(1)
            confidence[confidence < self.confidence_threshold] = 0  # threshold
            confidence = confidence.cpu().numpy()
            prediction = prediction.cpu().numpy()

            for index, oid in enumerate(origindid):
                pred_tracker = results.get(oid)
                if pred_tracker is None:
                    pred_tracker = PredictionTracker(mode=self.prediction_selection_mode)
                    pred_tracker.add_prediction(prediction[index], confidence[index])
                    results[oid] = pred_tracker
                else:
                    pred_tracker.add_prediction(prediction[index], confidence[index])

        return results

class PredictionTracker:
    import enum

    MODE = {
        "first": 0,
        "best": 1,
        "vote": 2
    }

    def __init__(self, mode = MODE["vote"]):
        self.predictions = []
        self.confidence = []
        self.mode = mode

    def add_prediction(self, prediction, confidence):
        self.predictions.append(prediction)
        self.confidence.append(confidence)

    def get_prediction(self):
        np_predictions = np.array(self.predictions)
        np_confidence = np.array(self.confidence)
        if self.mode == PredictionTracker.MODE["first"]:
            return np_predictions[0]
        if self.mode == PredictionTracker.MODE["best"]:
            max_index = np.where(np_confidence == np.amax(np_confidence))[0][0]
            return np_predictions[max_index]
        if self.mode == PredictionTracker.MODE["vote"]:
            unq, counts = np.unique(np_predictions, return_counts=True)
            max_count_index = np.where(counts == np.amax(counts))[0][0]
            return unq[max_count_index]


