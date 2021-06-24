import copy
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
from queue import Queue

from torch_points3d.datasets.samplers import BalancedRandomSampler
import torch_points3d.core.data_transform as cT
from torch_points3d.datasets.base_dataset import BaseDataset

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

S3DIS_NUM_CLASSES = 2

INV_OBJECT_LABEL = {
    0: "other",
    1: "road",
    # 2: "powerpole",
    # 3: "cable"
}

OBJECT_COLOR = np.asarray(
    [
        [233, 229, 107],  # 'road' .-> .yellow
        [95, 156, 196],  # 'powerpole' .-> . blue
        [179, 116, 81],  # 'cable'  ->  brown
        # [241, 149, 131],  # 'beam'  ->  salmon
        # [81, 163, 148],  # 'column'  ->  bluegreen
        # [77, 174, 84],  # 'window'  ->  bright green
        # [108, 135, 75],  # 'door'   ->  dark green
        # [41, 49, 101],  # 'chair'  ->  darkblue
        # [79, 79, 76],  # 'table'  ->  dark grey
        # [223, 52, 52],  # 'bookcase'  ->  red
        # [89, 47, 95],  # 'sofa'  ->  purple
        # [81, 109, 114],  # 'board'   ->  grey
        # [233, 233, 229],  # 'clutter'  ->  light grey
        [0, 0, 0],  # unlabelled .->. black
    ]
)

OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}

#segment types
ROOM_TYPES = {
    "segment": 0,
    "other": 1,
}

#validation segments
VALIDATION_ROOMS = [
    "segment",
]

################################### UTILS #######################################

#GOOD
def object_name_to_label(object_class):
    """convert from object name in S3DIS to an int"""

    object_label = OBJECT_LABEL.get(object_class.lower(), OBJECT_LABEL["other"])
    return object_label

#TODO segment maybe needed for very large files (100gb+)
def read_s3dis_format(train_file, room_name, label_out=True, verbose=False, debug=False, manual_shift=None):
    """extract data from a room folder"""
    room_type, room_idx = room_name.split("_")
    room_label = ROOM_TYPES[room_type]
    raw_path = osp.join(train_file, f"{room_name}.txt")
    if debug:
        reader = pd.read_csv(raw_path, delimiter="\n")
        RECOMMENDED = 10
        for idx, row in enumerate(reader.values):
            row = row[0].split(" ")
            if len(row) != RECOMMENDED:
                log.info("1: {} row {}: {}".format(raw_path, idx, row))

            try:
                for r in row:
                    r = float(r)
            except:
                log.info("2: {} row {}: {}".format(raw_path, idx, row))

        return True
    else:
        room_ver = pd.read_csv(raw_path, sep=" ", header=None).values
        xyz = np.ascontiguousarray(room_ver[:, 0:3], dtype="float64")

        #Shifts to local and quantizes to float32 by default
        xyz, shift_vector = shift_and_quantize(xyz, manual_shift=manual_shift)

        try:
            rgb = np.ascontiguousarray(room_ver[:, 3:6], dtype="uint8")
        except ValueError:
            rgb = np.zeros((room_ver.shape[0], 3), dtype="uint8")
            log.warning("WARN - corrupted rgb data for file %s" % raw_path)
        if not label_out:
            return xyz, rgb
        n_ver = len(room_ver)
        del room_ver
        nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(xyz)
        semantic_labels = np.zeros((n_ver,), dtype="int64")
        room_label = np.asarray([room_label])
        instance_labels = np.zeros((n_ver,), dtype="int64")
        objects = glob.glob(osp.join(train_file, "annotations/*.txt"))
        i_object = 1
        for single_object in objects:
            object_name = os.path.splitext(os.path.basename(single_object))[0]
            if object_name == "remainder": #expand to a list
                continue
            if verbose:
                log.debug("adding object " + str(i_object) + " : " + object_name)
            object_class = object_name.split("_")[0]
            object_label = object_name_to_label(object_class)
            obj_ver = pd.read_csv(single_object, sep=" ", header=None).values
            obj_xyz = np.ascontiguousarray(obj_ver[:, 0:3], dtype="float64")
            obj_xyz, _ = shift_and_quantize(obj_xyz, manual_shift=shift_vector)
            _, obj_ind = nn.kneighbors(obj_xyz)
            semantic_labels[obj_ind] = object_label
            instance_labels[obj_ind] = i_object
            i_object = i_object + 1

        return (
            torch.from_numpy(xyz),
            torch.from_numpy(rgb),
            torch.from_numpy(semantic_labels), #actual label
            torch.from_numpy(instance_labels), #index of instance
            torch.from_numpy(room_label),
            shift_vector
        )

def shift_and_quantize(xyz, qtype = np.float32, manual_shift = None):
    if manual_shift is None:
        shift_threshold = 100
        x = xyz[:,0].min()
        y = xyz[:,1].min()
        z = xyz[:,2].min()

        shift_x = calc_shift(x, shift_threshold)
        shift_y = calc_shift(y, shift_threshold)
        shift_z = calc_shift(z, shift_threshold)

        shift = np.array([shift_x, shift_y, shift_z])
    else:
        shift = np.array(manual_shift)
    scale = 1

    out = np.add(xyz, shift)
    out = np.multiply(out, scale)

    return out.astype(qtype), shift

def calc_shift(number, threshold):
    shift = 0
    if number / threshold > 1:
        r = int(number % threshold)
        shift = int(number - r) * -1.00
    return shift

def to_ply(pos, label, file):
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    pos = np.asarray(pos)
    colors = OBJECT_COLOR[np.asarray(label)]
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    el = PlyElement.describe(ply_array, "S3DIS")
    PlyData([el], byte_order=">").write(file)

#Changed from InMemoryDataset to Dataset
class NexploreS3DISOriginalFused(Dataset):
    """ Original S3DIS dataset. Each area is loaded individually and can be processed using a pre_collate transform. 
    This transform can be used for example to fuse the area into a single space and split it into 
    spheres or smaller regions. If no fusion is applied, each element in the dataset is a single room by default.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    test_area: int
        number between 1 and 6 that denotes the area used for testing
    split: str
        can be one of train, trainval, val or test
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    pre_transform
    transform
    pre_filter
    """

    num_classes = S3DIS_NUM_CLASSES

    def __init__(
        self,
        root,
        test_area=8,
        split="train",
        sample_per_epoch=-1,
        radius=2,
        transform=None,
        pre_transform=None,
        pre_collate_transform=None,
        lowres_subsampling = 0.5,
        pre_filter=None,
        keep_instance=False,
        verbose=False,
        debug=False,
        train_areas=[],
        val_areas=[],
        test_areas=[]
    ):
        assert len(test_areas) > 0
        assert len(train_areas) > 0
        if val_areas is None:
            val_areas = []
        self.lowres_subsampling = lowres_subsampling
        self.train_areas = list(train_areas)
        self.val_areas = list(val_areas)
        if len(self.val_areas) <= 0:
            self.val_areas = self.train_areas
        self.test_areas = list(test_areas)
        self.transform = transform
        self.pre_collate_transform = pre_collate_transform
        self.test_area = test_area
        self.keep_instance = keep_instance
        self.verbose = verbose
        self.debug = debug
        self._split = split
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        # self.areas_paths = []

        #init root to call super later
        self.root = root
        #Paths need super to be called first
        if split == "train":
            self.areas_paths = self.raw_areas_paths
        elif split == "val":
            self.areas_paths = self.raw_val_areas_paths
        elif split == "test":
            self.areas_paths = self.raw_test_areas_paths
        # else:
        #     raise ValueError((f"Split {split} found, but expected either " "train, val, or test"))

        super(NexploreS3DISOriginalFused, self).__init__(root, transform, pre_transform, pre_filter)

        # self._load_data(path) #load on get

        # if split == "test":
        #     self.raw_test_data = torch.load(self.raw_areas_paths[test_area - 1])

    #Overload the same so we can call super later
    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    # Overload the same so we can call super later
    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def split_areas_paths(self):
        return self.areas_paths

    @property
    def center_labels(self):
        if hasattr(self.data, "center_label"):
            return self.data.center_label
        else:
            return None

    @property
    def raw_file_names(self):
        return self.train_areas + self.test_areas

    @property
    def pre_processed_path(self):
        pre_processed_file_names = "preprocessed.pt"
        return os.path.join(self.processed_dir, pre_processed_file_names)

    @property
    def raw_areas_paths(self):
        return [os.path.join(self.processed_dir,"train" ,f"{name}.pt") for name in self.train_areas]

    @property
    def raw_val_areas_paths(self):
        return [os.path.join(self.processed_dir, "val", f"{name}.pt") for name in self.val_areas]

    @property
    def raw_test_areas_paths(self):
        return [os.path.join(self.processed_dir, "test", f"{name}.pt") for name in self.test_areas]

    #used for processed_paths
    @property
    def processed_file_names(self):
        test_area = self.test_area
        return (
            # ["{}_{}.pt".format(s, test_area) for s in ["train", "val", "test", "trainval"]]
            self.raw_areas_paths
            + self.raw_test_areas_paths
            + [self.pre_processed_path]
        )


    #Loaded in the init method.  Raw data with full precision
    @property
    def raw_test_data(self):
        return self._raw_test_data

    @raw_test_data.setter
    def raw_test_data(self, value):
        self._raw_test_data = value

    #simple method to output a warning about not seeing data
    def download(self):
        raw_folders = os.listdir(self.raw_dir)
        if len(raw_folders) == 0:
            # if not os.path.exists(osp.join(self.root, self.zip_name)):
            log.info(f"No data found in {self.raw_dir}")
            log.info("WARNING: You need to download data from sharepoint and put it in the data root folder")


    def process(self):
        #TODO IF ELSE is ugly.  All could be refactored

        if self._split == "train":
            for area in self.train_areas:
                if os.path.exists(os.path.join(self.processed_dir, "train", area + ".pt")):
                    continue  # skip if already exists

                train_data_list = []
                manual_shift = None
                dirs = os.listdir(osp.join(self.raw_dir, area))
                dirs.pop() #last segment used for val

                for segment_name in dirs:
                    segment_path = osp.join(self.raw_dir, area, segment_name)

                    print(f"Processing training data {area}, {segment_name}")
                    if os.path.isdir(segment_path):
                        # area_idx = folders.index(area)
                        segment_type, segment_idx = segment_name.split("_")

                        xyz, rgb, semantic_labels, instance_labels, room_label, last_shift_vector = read_s3dis_format(
                            segment_path, segment_name, label_out=True, verbose=self.verbose, debug=self.debug, manual_shift=manual_shift
                        )

                        # all segments use same shift
                        if manual_shift is None:
                            manual_shift = last_shift_vector

                        if self.debug:
                            pass

                        rgb_norm = rgb.float() / 255.0
                        data = Data(pos=xyz, y=semantic_labels, rgb=rgb_norm)
                        # TODO implement better way to select validation segments

                        if self.keep_instance:
                            data.instance_labels = instance_labels

                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue

                        train_data_list.append(data)

                if self.pre_collate_transform:
                    log.info("pre_collate_transform ...")
                    log.info(self.pre_collate_transform)
                    train_data_list = self.pre_collate_transform(train_data_list)

                grid_sampler = cT.GridSphereSampling(self._radius, self._radius, center=False)
                if self._sample_per_epoch > 0:
                    train_data_list = self._gen_low_res(self.lowres_subsampling, train_data_list)
                else:
                    train_data_list = grid_sampler(train_data_list)
                    train_data_list = [d for d in train_data_list if len(d.origin_id) > 10]

                self._save_data(train_data_list, os.path.join(self.processed_dir, "train", area + ".pt"))


        elif self._split == "val":
            for area in self.val_areas:
                if os.path.exists(os.path.join(self.processed_dir, "val", area + ".pt")):
                    continue  # skip if already exists

                val_data_list = []
                manual_shift = None
                dirs = os.listdir(osp.join(self.raw_dir, area))
                dirs = dirs[-1:] # last segment used for val

                for segment_name in dirs:
                    segment_path = osp.join(self.raw_dir, area, segment_name)

                    print(f"Processing val data {area}, {segment_name}")
                    if os.path.isdir(segment_path):
                        # area_idx = folders.index(area)
                        segment_type, segment_idx = segment_name.split("_")

                        xyz, rgb, semantic_labels, instance_labels, room_label, last_shift_vector = read_s3dis_format(
                            segment_path, segment_name, label_out=True, verbose=self.verbose, debug=self.debug,
                            manual_shift=manual_shift
                        )

                        # all segments use same shift
                        if manual_shift is None:
                            manual_shift = last_shift_vector

                        if self.debug:
                            pass

                        rgb_norm = rgb.float() / 255.0
                        data = Data(pos=xyz, y=semantic_labels, rgb=rgb_norm)
                        # TODO implement better way to select validation segments

                        if self.keep_instance:
                            data.instance_labels = instance_labels

                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue

                        val_data_list.append(data)

                if self.pre_collate_transform:
                    log.info("pre_collate_transform ...")
                    log.info(self.pre_collate_transform)
                    val_data_list = self.pre_collate_transform(val_data_list)

                grid_sampler = cT.GridSphereSampling(self._radius, self._radius, center=False)
                if self._sample_per_epoch > 0:
                    val_data_list = self._gen_low_res(self.lowres_subsampling, val_data_list)
                else:
                    val_data_list = grid_sampler(val_data_list)
                    val_data_list = [d for d in val_data_list if len(d.origin_id) > 10]

                self._save_data(val_data_list, os.path.join(self.processed_dir, "val", area + ".pt"))

        elif self._split == "test":
            for area in self.test_areas:
                if os.path.exists(os.path.join(self.processed_dir, "test", area + ".pt")):
                    continue  # skip if already exists

                test_data_list = []
                manual_shift = None
                dirs = os.listdir(osp.join(self.raw_dir, area))
                for segment_name in dirs:
                    print(f"Processing test data {area}, {segment_name}")

                    segment_path = osp.join(self.raw_dir, area, segment_name)
                    if os.path.isdir(segment_path):

                        xyz, rgb, semantic_labels, instance_labels, room_label, last_shift_vector = read_s3dis_format(
                            segment_path, segment_name, label_out=True, verbose=self.verbose, debug=self.debug,
                            manual_shift=manual_shift
                        )

                        if manual_shift is None:
                            manual_shift = last_shift_vector

                        rgb_norm = rgb.float() / 255.0
                        data = Data(pos=xyz, y=semantic_labels, rgb=rgb_norm)

                        if self.keep_instance:
                            data.instance_labels = instance_labels

                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue

                        test_data_list.append(data)

                if self.pre_collate_transform:
                    log.info("pre_collate_transform ...")
                    log.info(self.pre_collate_transform)
                    test_data_list = self.pre_collate_transform(test_data_list)

                grid_sampler = cT.GridSphereSampling(self._radius, self._radius, center=False)
                if self._sample_per_epoch > 0:
                    test_data_list = self._gen_low_res(self.lowres_subsampling, test_data_list)
                else:
                    test_data_list = grid_sampler(test_data_list)
                    test_data_list = [d for d in test_data_list if len(d.origin_id) > 10]

                self._save_data(test_data_list, os.path.join(self.processed_dir, "test", area + ".pt"))

    def _load_data(self, path):
        self.data, self.slices = torch.load(path)

    def _gen_low_res(self, resolution, data_list):
        _grid_sphere_sampling = cT.GridSampling3D(size=resolution)
        low_res_data = _grid_sphere_sampling(copy.deepcopy(data_list))
        low_res_tree = KDTree(np.asarray(low_res_data.pos), leaf_size=10)
        setattr(low_res_data, cT.SphereSampling.KDTREE_KEY, low_res_tree)
        tree = KDTree(np.asarray(data_list.pos), leaf_size=50)
        setattr(data_list, cT.SphereSampling.KDTREE_KEY, tree)
        return [data_list, low_res_data]


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
        # self._sample_per_epoch = sample_per_epoch
        # self._radius = radius

        self.meta = None
        self.spheres = []
        self.sample_data = []
        self.file_index = None
        self.last_index = 0
        super().__init__(root, *args, **kwargs)
        self._grid_sphere_sampling = cT.GridSampling3D(size=self._radius / 10.0)

    def _calc_meta(self):
        print(f"Calculate Meta: {self.split_areas_paths}")
        if self._sample_per_epoch > 0:
            self._calc_meta_sample()
        else:
            self._calc_meta_full()

    def _calc_meta_full(self):
        self.meta = {}
        for index, path in enumerate(self.split_areas_paths):
            mta = None
            if index == 0:
                mta = {"start": 0, "end": self._determine_length(path) - 1}
            else:
                start = self.meta[index - 1]["end"] + 1
                mta = {"start": start, "end": start + self._determine_length(path) - 1}

            self.meta[index] = mta

        return self.meta

    def _calc_meta_sample(self):
        self.meta = {}
        for index, path in enumerate(self.split_areas_paths):
            mta = None
            if index == 0:
                mta = {"start": 0, "end": self._sample_per_epoch}
            else:
                start = self.meta[index - 1]["end"] + 1
                mta = {"start": start, "end": start + self._sample_per_epoch}

            self.meta[index] = mta

        return self.meta

    def __len__(self):
        if self.meta is None:
            self._calc_meta()
        return self.meta[list(self.meta.keys())[-1]]["end"]

    def get(self, idx):
        #Quick fix for 0 index bug
        if self.last_index > 1 and idx == 0:
            idx = self.last_index
        self.last_index = idx
        # print(f"index: {idx}, split: {self._split}")
        meta_index = [i for i in self.meta.keys() if self.meta[i]["start"] <= idx <= self.meta[i]["end"]][0]

        # load next file
        if meta_index != self.file_index:
            print(f"loading new file for {self._split}")
            self.file_index = meta_index
            if self._sample_per_epoch > 0:
                self.sample_data = self._load_sample_data(self.split_areas_paths[self.file_index])
            else:
                data = self._load_spheres(self.split_areas_paths[self.file_index])
                self.spheres = data


        if self._sample_per_epoch > 0:
            return self._get_random()
        else:
            return self.spheres[idx - self.meta[meta_index]["start"]].clone()

    def process(self):  # We have to include this method, otherwise the parent class skips processing
        super().process()

    def download(self):  # We have to include this method, otherwise the parent class skips download
        super().download()

    #Implement using get random
    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        sphere = None
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        #TODO fail after timeout
        while sphere is None:
            valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
            centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
            centre = valid_centres[centre_idx]
            sphere_sampler = cT.SphereSampling(self._radius, centre[:3], align_origin=False)
            sphere_test = sphere_sampler(self.sample_data)
            if len(sphere_test.origin_id) > 10:
                sphere = sphere_test
        return sphere

    def _save_data(self, data_list, path):
        base, file = os.path.split(path)
        os.makedirs(base, exist_ok=True)
        torch.save(data_list, path)

    def _determine_length(self, path):
        _datas = torch.load(path)
        _spheres_count = len(_datas)
        return _spheres_count

    #if samples_per_epoch = -1, we store data as spheres
    def _load_spheres(self, path):
        _spheres = torch.load(path)
        # _spheres = [d for d in _spheres if len(d.origin_id) > 10]
        return _spheres

    #if samples_per_epoch > 0, we store data as full pointcloud
    def _load_sample_data(self, path):
        d = torch.load(path)
        _data = d[0]
        _low_res = d[1]
        self._centres_for_sampling = []

        centres = torch.empty((_low_res.pos.shape[0], 5), dtype=torch.float)
        centres[:, :3] = _low_res.pos
        centres[:, 3] = 0
        centres[:, 4] = _low_res.y
        self._centres_for_sampling.append(centres)

        self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
        uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
        uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
        self._label_counts = uni_counts / np.sum(uni_counts)
        self._labels = uni

        return _data

class NexploreS3DISFusedDataset(BaseDataset):
    """ Wrapper around S3DISSphere that creates train and test datasets.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - fold: test_area parameter
            - pre_collate_transform
            - train_transforms
            - test_transforms
    """

    INV_OBJECT_LABEL = INV_OBJECT_LABEL

    FORWARD_CLASS = "forward.nexplore.NexploreS3DISFusedForwardDataset"

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        sampling_format = dataset_opt.get("sampling_format", "sphere")
        dataset_cls = NexploreS3DISSphere

        self.train_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=500,
            test_area=self.dataset_opt.fold,
            radius=20,
            split="train",
            lowres_subsampling=self.dataset_opt.lowres_subsampling,
            train_areas=dataset_opt.train_areas,
            val_areas=dataset_opt.val_areas,
            test_areas=dataset_opt.test_areas,
            pre_collate_transform=self.pre_collate_transform,
            transform=self.train_transform,
        )

        self.val_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=100,
            test_area=self.dataset_opt.fold,
            radius=20,
            split="val",
            lowres_subsampling=self.dataset_opt.lowres_subsampling,
            train_areas=dataset_opt.train_areas,
            val_areas=dataset_opt.val_areas,
            test_areas=dataset_opt.test_areas,
            pre_collate_transform=self.pre_collate_transform,
            transform=self.val_transform,
        )

        self.test_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=100,
            test_area=self.dataset_opt.fold,
            radius=20,
            split="test",
            lowres_subsampling=self.dataset_opt.lowres_subsampling,
            train_areas=dataset_opt.train_areas,
            val_areas=dataset_opt.val_areas,
            test_areas=dataset_opt.test_areas,
            pre_collate_transform=self.pre_collate_transform,
            transform=self.test_transform,
        )

        if dataset_opt.class_weight_method:
            self.add_weights(class_weight_method=dataset_opt.class_weight_method)

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
        to_ply(pos, label, file)

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
