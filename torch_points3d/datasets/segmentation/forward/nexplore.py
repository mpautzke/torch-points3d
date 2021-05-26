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
from torch_points3d.datasets.segmentation.nexplore_old import shift_and_quantize, OBJECT_COLOR

from torch_points3d.datasets.samplers import BalancedRandomSampler
import torch_points3d.core.data_transform as cT
from torch_points3d.datasets.base_dataset import BaseDataset

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

S3DIS_NUM_CLASSES = 3

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

################################### UTILS #######################################

def read_s3dis_format(train_file, shift_quantize = False):
    """extract data from a room folder"""
    raw_path = osp.join(train_file)
    room_ver = pd.read_csv(raw_path, sep=" ", header=None).values

    xyz = np.ascontiguousarray(room_ver[:, 0:3], dtype="float64")

    shift_vector = np.array([0,0,0])
    if shift_quantize:
        #TODO do we need to keep shift_vector
        xyz, shift_vector = shift_and_quantize(xyz)

    # outliers = remove_outliers(xyz)

    try:
        rgb = np.ascontiguousarray(room_ver[:, 3:6], dtype="uint8")
    except ValueError:
        rgb = np.zeros((room_ver.shape[0], 3), dtype="uint8")
        log.warning("WARN - corrupted rgb data for file %s" % raw_path)

    n_ver = len(room_ver)
    semantic_labels = np.zeros((n_ver,), dtype="int64")

    return torch.from_numpy(xyz), torch.from_numpy(rgb), torch.from_numpy(semantic_labels)

#TODO Would need to modify original file if we remove outliers as we depend on original index
def remove_outliers(xyz, x_c = 5, y_c = 5, z_c = 20):
    # a = np.array(x)
    x_index = outlier_index(xyz[:,0], x_c)
    y_index = outlier_index(xyz[:,1], y_c)
    z_index = outlier_index(xyz[:,2], z_c)

    xy_index = np.union1d(x_index, y_index)
    xyz_index = np.union1d(xy_index, z_index)

    return xyz_index

def outlier_index(a, outlierConstant):
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    results = np.where((a <= quartileSet[0]) | (a >= quartileSet[1]))
    return results

class NexploreS3DISOriginalFused(InMemoryDataset):
    num_classes = S3DIS_NUM_CLASSES

    def __init__(
        self,
        root,
        fname,
        split="train",
        transform=None,
        pre_transform=None,
        pre_collate_transform=None,
        pre_filter=None,
        verbose=False,
        debug=False,
    ):
        self.path = os.path.join(root, fname)
        self.transform = transform
        self.pre_collate_transform = pre_collate_transform
        self.verbose = verbose
        self.debug = debug
        self._split = split
        super(NexploreS3DISOriginalFused, self).__init__(root, transform, pre_transform, pre_filter)
        self._load_data(self.processed_paths[2])
        self.raw_test_data = self.read_raw_data() #TODO do not hold in mem
        self.shifted_test_data = self.read_raw_data(shift_quantize = True) #TODO do not hold in mem

    def read_raw_data(self, shift_quantize = False):
        xyz, rgb, semantic_labels = read_s3dis_format(
            self.path, shift_quantize = False
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
        # return [os.path.join(self.processed_dir, "raw_area_%i.pt" % i) for i in range(6)]


    #used for processed_paths
    @property
    def processed_file_names(self):
        return (
            [self.path]
            + [self.pre_processed_path]
            + [self.transformed_path]
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
        if not os.path.exists(self.pre_processed_path):
        # if True:
            xyz, rgb, semantic_labels = read_s3dis_format(
                self.path, shift_quantize=True
            )

            rgb_norm = rgb.float() / 255.0
            data = Data(pos=xyz, y=semantic_labels, rgb=rgb_norm)

            #Placeholder for multi segment
            data_list = [[] for _ in range(1)]
            data_list[0].append(data)

            # raw_areas = cT.PointCloudFusion()(data_list)
            # for i, area in enumerate(raw_areas):
            #     if area.__len__() > 0:
            #         torch.save(area, self.raw_areas_paths[i])

            for area_datas in data_list:
                # Apply pre_transform
                if self.pre_transform is not None:
                    for data in area_datas:
                        data = self.pre_transform(data)
            torch.save(data_list, self.pre_processed_path)
        else:
            data_list = torch.load(self.pre_processed_path)

        test_data_list = data_list[0]

        if self.pre_collate_transform:
            log.info("pre_collate_transform ...")
            log.info(self.pre_collate_transform)
            test_data_list = self.pre_collate_transform(test_data_list)

        self._save_data(test_data_list)

    def _save_data(self, test_data_list):
        torch.save(self.collate(test_data_list), self.processed_paths[2])

    def _load_data(self, path):
        self.data, self.slices = torch.load(path)


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

    def __init__(self, root, sample_per_epoch=100, radius=2, *args, **kwargs):
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._grid_sphere_sampling = cT.GridSampling3D(size=radius / 10.0)
        super().__init__(root, *args, **kwargs)

    def __len__(self):
        if self._sample_per_epoch > 0:
            return self._sample_per_epoch
        else:
            return len(self._test_spheres)

    def get(self, idx):
        if self._sample_per_epoch > 0:
            return self._get_random()
        else:
            return self._test_spheres[idx].clone()

    def process(self):  # We have to include this method, otherwise the parent class skips processing
        super().process()

    def download(self):  # We have to include this method, otherwise the parent class skips download
        super().download()

    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        area_data = self._datas[centre[3].int()]
        sphere_sampler = cT.SphereSampling(self._radius, centre[:3], align_origin=False)
        return sphere_sampler(area_data)

    def _save_data(self, test_data_list):
        torch.save(test_data_list, self.processed_paths[2])

    def _load_data(self, path):
        self._datas = torch.load(path)
        if not isinstance(self._datas, list):
            self._datas = [self._datas]
        if self._sample_per_epoch > 0:
            self._centres_for_sampling = []
            for i, data in enumerate(self._datas):
                assert not hasattr(
                    data, cT.SphereSampling.KDTREE_KEY
                )  # Just to make we don't have some out of date data in there
                low_res = self._grid_sphere_sampling(data.clone())
                centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                self._centres_for_sampling.append(centres)
                tree = KDTree(np.asarray(data.pos), leaf_size=10)
                setattr(data, cT.SphereSampling.KDTREE_KEY, tree)

            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            self._labels = uni
        else:
            grid_sampler = cT.GridSphereSampling(self._radius, self._radius, center=False)
            self._test_spheres = grid_sampler(self._datas)
            self._test_spheres = [d for d in self._test_spheres if d.origin_id.__len__() >= 10000]

class NexploreS3DISFusedForwardDataset(BaseDataset):

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        dataset_cls = NexploreS3DISSphere

        self.test_dataset = dataset_cls(
            dataset_opt.dataroot,
            fname=dataset_opt.dataset_name,
            sample_per_epoch=dataset_opt.samples,
            split="test",
            radius=dataset_opt.radius,
            pre_collate_transform=self.pre_collate_transform,
            transform=self.test_transform,
        )

        if dataset_opt.class_weight_method:
            self.add_weights(class_weight_method=dataset_opt.class_weight_method)

        # self.raw_data = self.read_raw_data()

    # def read_raw_data(self):
    #     xyz, rgb, semantic_labels = read_s3dis_format(
    #         self.path, "segment", label_out=False, verbose=self.verbose, debug=self.debug
    #     )
    #
    #     return Data(pos=xyz, y=semantic_labels, rgb=rgb)

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

    def predict_original_samples(self, batch, conv_type, output):
        """ Takes the output generated by the NN and upsamples it to the original data
        Arguments:
            batch -- processed batch
            conv_type -- Type of convolutio (DENSE, PARTIAL_DENSE, etc...)
            output -- output predicted by the model
        """
        full_res_results = {}
        num_sample = BaseDataset.get_num_samples(batch, conv_type)
        if conv_type == "DENSE":
            output = output.reshape(num_sample, -1, output.shape[-1])  # [B,N,L]

        setattr(batch, "_pred", output)
        for b in range(num_sample):
            predicted = BaseDataset.get_sample(batch, "_pred", b, conv_type).reshape(-1, output.shape[-1])
            origindid = BaseDataset.get_sample(batch, SaveOriginalPosId.KEY, b, conv_type).cpu().numpy()
            #TODO need to take original pos and interpolate with transformed pos
            # full_prediction = knn_interpolate(predicted, sample_raw_pos[origindid], sample_raw_pos, k=3)
            labels = predicted.max(1)[1].cpu().numpy()
            for index, id in enumerate(origindid):
                full_res_results[id] = labels[index]
        return full_res_results

class _ForwardS3dis(torch.utils.data.Dataset):
    """ Dataset to run forward inference on Shapenet kind of data data. Runs on a whole folder.
    Arguments:
        path: folder that contains a set of files of a given category
        category: index of the category to use for forward inference. This value depends on how many categories the model has been trained one.
        transforms: transforms to be applied to the data
        include_normals: wether to include normals for the forward inference
    """

    def __init__(self, path, category: int, transforms=None, include_normals=True):
        super().__init__()
        self._category = category
        self._path = path
        self._files = sorted(glob.glob(os.path.join(self._path, "*.txt")))
        self._transforms = transforms
        self._include_normals = include_normals
        assert os.path.exists(self._path)
        if self.__len__() == 0:
            raise ValueError("Empty folder %s" % path)

    def __len__(self):
        return len(self._files)

    def _read_file(self, filename):
        raw = read_txt_array(filename)
        pos = raw[:, :3]
        x = raw[:, 3:6]
        if raw.shape[1] == 7:
            y = raw[:, 6].type(torch.long)
        else:
            y = None
        return Data(pos=pos, x=x, y=y)

    def get_raw(self, index):
        """ returns the untransformed data associated with an element
        """
        return self._read_file(self._files[index])

    @property
    def num_features(self):
        feats = self[0].x
        if feats is not None:
            return feats.shape[-1]
        return 0

    def get_filename(self, index):
        return os.path.basename(self._files[index])

    def __getitem__(self, index):
        data = self._read_file(self._files[index])
        category = torch.ones(data.pos.shape[0], dtype=torch.long) * self._category
        setattr(data, "category", category)
        setattr(data, "sampleid", torch.tensor([index]))
        if not self._include_normals:
            data.x = None
        if self._transforms is not None:
            data = self._transforms(data)
        return data

