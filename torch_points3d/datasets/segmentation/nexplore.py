import copy
import os
import os.path as osp
from itertools import repeat, product
import numpy as np
import h5py
import datetime
import time
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
import os
import multiprocessing
from queue import Queue
from timeit import default_timer as timer

from torch_points3d.datasets.samplers import BalancedRandomSampler
import torch_points3d.core.data_transform as cT
from torch_points3d.datasets.base_dataset import BaseDataset

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

# if not 'INV_OJBECT_LABEL_MAP' in globals():
#     INV_OBJECT_LABEL_MAP = {}
# S3DIS_NUM_CLASSES = 0
# INV_OBJECT_LABEL = {}


OBJECT_COLOR = np.asarray(
    [
        [233, 229, 107],  # 'road' .-> .yellow
        [95, 156, 196],  # 'powerpole' .-> . blue
        [179, 116, 81],  # 'cable'  ->  brown
        [241, 149, 131],  # 'beam'  ->  salmon
        [81, 163, 148],  # 'column'  ->  bluegreen
        [77, 174, 84],  # 'window'  ->  bright green
        [108, 135, 75],  # 'door'   ->  dark green
        [41, 49, 101],  # 'chair'  ->  darkblue
        [79, 79, 76],  # 'table'  ->  dark grey
        [223, 52, 52],  # 'bookcase'  ->  red
        [89, 47, 95],  # 'sofa'  ->  purple
        [81, 109, 114],  # 'board'   ->  grey
        [233, 233, 229],  # 'clutter'  ->  light grey
        [0, 0, 0],  # unlabelled .->. black
    ]
)

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
# def object_name_to_label(object_class):
#     """convert from object name in S3DIS to an int"""
#     log.info(INV_OBJECT_LABEL_MAP.keys())
#     for key in INV_OBJECT_LABEL_MAP.keys():
#         if object_class.lower() in INV_OBJECT_LABEL_MAP[key]:
#             return key
#     #0 reserved for other
#     return 0
#
#     # object_label = OBJECT_LABEL.get(object_class.lower(), OBJECT_LABEL["other"])
#     # return object_label

# #TODO segment maybe needed for very large files (100gb+)
# def read_s3dis_format(train_file, room_name, label_out=True, verbose=False, debug=False, manual_shift=None):
#     """extract data from a room folder"""
#     room_type, room_idx = room_name.split("_")
#     room_label = ROOM_TYPES[room_type]
#     raw_path = osp.join(train_file, f"{room_name}.txt")
#     if debug:
#         reader = pd.read_csv(raw_path, delimiter="\n")
#         RECOMMENDED = 10
#         for idx, row in enumerate(reader.values):
#             row = row[0].split(" ")
#             if len(row) != RECOMMENDED:
#                 log.info("1: {} row {}: {}".format(raw_path, idx, row))
#
#             try:
#                 for r in row:
#                     r = float(r)
#             except:
#                 log.info("2: {} row {}: {}".format(raw_path, idx, row))
#
#         return True
#     else:
#         room_ver = pd.read_csv(raw_path, sep=" ", header=None).values
#         xyz = np.ascontiguousarray(room_ver[:, 0:3], dtype="float64")
#
#         #Shifts to local and quantizes to float32 by default
#         xyz, shift_vector = shift_and_quantize(xyz, manual_shift=manual_shift)
#
#         try:
#             rgb = np.ascontiguousarray(room_ver[:, 3:6], dtype="uint8")
#         except ValueError:
#             rgb = np.zeros((room_ver.shape[0], 3), dtype="uint8")
#             log.warning("WARN - corrupted rgb data for file %s" % raw_path)
#         if not label_out:
#             return xyz, rgb
#         n_ver = len(room_ver)
#         del room_ver
#         nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(xyz)
#         semantic_labels = np.zeros((n_ver,), dtype="int64")
#         room_label = np.asarray([room_label])
#         instance_labels = np.zeros((n_ver,), dtype="int64")
#         objects = glob.glob(osp.join(train_file, "annotations/*.txt"))
#         i_object = 1
#         for single_object in objects:
#             object_name = os.path.splitext(os.path.basename(single_object))[0]
#             if object_name == "remainder": #expand to a list
#                 continue
#             if verbose:
#                 log.debug("adding object " + str(i_object) + " : " + object_name)
#             object_class = object_name.split("_")[0]
#             object_label = object_name_to_label(object_class)
#             if object_label == 0:
#                 continue
#             obj_ver = pd.read_csv(single_object, sep=" ", header=None).values
#             obj_xyz = np.ascontiguousarray(obj_ver[:, 0:3], dtype="float64")
#             obj_xyz, _ = shift_and_quantize(obj_xyz, manual_shift=shift_vector)
#             _, obj_ind = nn.kneighbors(obj_xyz)
#             semantic_labels[obj_ind] = object_label
#             instance_labels[obj_ind] = i_object
#             i_object = i_object + 1
#
#         return (
#             torch.from_numpy(xyz),
#             torch.from_numpy(rgb),
#             torch.from_numpy(semantic_labels), #actual label
#             torch.from_numpy(instance_labels), #index of instance
#             torch.from_numpy(room_label),
#             shift_vector
#         )
#
# def shift_and_quantize(xyz, qtype = np.float32, manual_shift = None):
#     if manual_shift is None:
#         shift_threshold = 100
#         x = xyz[:,0].min()
#         y = xyz[:,1].min()
#         z = xyz[:,2].min()
#
#         shift_x = calc_shift(x, shift_threshold)
#         shift_y = calc_shift(y, shift_threshold)
#         shift_z = calc_shift(z, shift_threshold)
#
#         shift = np.array([shift_x, shift_y, shift_z])
#     else:
#         shift = np.array(manual_shift)
#     scale = 1
#
#     out = np.add(xyz, shift)
#     out = np.multiply(out, scale)
#
#     return out.astype(qtype), shift
#
# def calc_shift(number, threshold):
#     shift = 0
#     if number / threshold > 1:
#         r = int(number % threshold)
#         shift = int(number - r) * -1.00
#     return shift
#
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

    _pp_seconds = 0.0

    def __init__(
        self,
        root,
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
        self.num_classes = S3DIS_NUM_CLASSES
        self.object_label = INV_OBJECT_LABEL
        self.object_label_map = INV_OBJECT_LABEL_MAP
        print(f"num_classes: {self.num_classes}")
        # assert len(test_areas) > 0
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

    def concatenate(self, pos, rgb, preds):
        pos_rgb = np.concatenate((pos, rgb), axis=1)
        pos_rgb_preds = np.concatenate((pos_rgb, preds), axis=1)

        return pos_rgb_preds

    def save_file(self, path, filename, results):

        output_timer_start = timer()
        os.makedirs(path, exist_ok=True)
        out_file = f"{filename}.txt"
        print(f"Writing {out_file}...")
        path = os.path.join(path, out_file)
        # np.save(path, results)  #These are faster

        np.savetxt(path, results, fmt='%s')
        # res_df = pd.DataFrame(results)
        # res_df.to_csv(path, sep=' ', index=False, header=False)
        output_timer_end = timer()
        print(f"{out_file} elapsed time: {round(output_timer_end - output_timer_start, 2)} seconds")

    def process_train(self, area):
        st = datetime.datetime.utcnow()
        log.info("Starting train preprocessing on %s"%area)
        if os.path.exists(os.path.join(self.processed_dir, "train", area + ".pt")):
            log.info("Train preprocessing for %s already exists. Skipping."%area)
            return  # skip if already exists

        train_data_list = []
        manual_shift = None
        dirs = os.listdir(osp.join(self.raw_dir, area))
        if len(dirs) > 1:
            dirs.pop() #last segment used for val

        for segment_name in dirs:
            segment_path = osp.join(self.raw_dir, area, segment_name)

            log.info(f"Read train data {area}, {segment_name}")
            if os.path.isdir(segment_path):
                # area_idx = folders.index(area)
                segment_type, segment_idx = segment_name.split("_")

                s3_st = datetime.datetime.utcnow()
                xyz, rgb, semantic_labels, instance_labels, room_label, last_shift_vector = self.read_s3dis_format(
                    segment_path, segment_name, label_out=True, verbose=self.verbose, debug=self.debug, manual_shift=manual_shift
                )

                # pos_out = np.array(xyz, dtype=np.str)
                # rgb_out = np.array(rgb, dtype=np.str)
                # values_out = np.array(semantic_labels, dtype=np.str).reshape((-1, 1))
                # out = self.concatenate(pos_out, rgb_out, values_out)
                # self.save_file(os.path.join(self.processed_dir, "train"), f"{area}_debug", out)
                s3_tt = (datetime.datetime.utcnow() - s3_st).total_seconds()
                log.info("s3dis train read for %s took %.1f seconds"%(area,s3_tt))

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
            log.info("Starting train pre_collate_transform for %s:"%area)
            log.info(self.pre_collate_transform)
            train_data_list = self.pre_collate_transform(train_data_list)

        grid_sampler = cT.GridSphereSampling(self._radius, self._radius, center=False)
        if self._sample_per_epoch > 0:
            train_data_list = self._gen_low_res(self.lowres_subsampling, train_data_list)
        else:
            train_data_list = grid_sampler(train_data_list)
            train_data_list = [d for d in train_data_list if len(d.origin_id) > 10]

        log.info("Saving train data for %s"%area)
        self._save_data(train_data_list, os.path.join(self.processed_dir, "train", area + ".pt"))
        tt = (datetime.datetime.utcnow() - st).total_seconds()
        self._pp_seconds += tt
        log.info("Completed preprocessing train data for %s in %.1f seconds !"%(area,tt))

    def process_val(self, area):
        st = datetime.datetime.utcnow()
        log.info("Starting val preprocessing on %s"%area)
        if os.path.exists(os.path.join(self.processed_dir, "val", area + ".pt")):
            log.info("Val preprocessing for %s already exists. Skipping."%area)
            return  # skip if already exists

        val_data_list = []
        manual_shift = None
        dirs = os.listdir(osp.join(self.raw_dir, area))
        dirs = dirs[-1:] # last segment used for val

        for segment_name in dirs:
            segment_path = osp.join(self.raw_dir, area, segment_name)

            log.info(f"Read val data {area}, {segment_name}")
            if os.path.isdir(segment_path):
                # area_idx = folders.index(area)
                segment_type, segment_idx = segment_name.split("_")

                s3_st = datetime.datetime.utcnow()
                xyz, rgb, semantic_labels, instance_labels, room_label, last_shift_vector = self.read_s3dis_format(
                    segment_path, segment_name, label_out=True, verbose=self.verbose, debug=self.debug,
                    manual_shift=manual_shift
                )
                s3_tt = (datetime.datetime.utcnow() - s3_st).total_seconds()
                log.info("s3dis val read for %s took %.1f seconds"%(area,s3_tt))

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
            log.info("Starting val pre_collate_transform for %s:"%area)
            log.info(self.pre_collate_transform)
            val_data_list = self.pre_collate_transform(val_data_list)

        grid_sampler = cT.GridSphereSampling(self._radius, self._radius, center=False)
        if self._sample_per_epoch > 0:
            val_data_list = self._gen_low_res(self.lowres_subsampling, val_data_list)
        else:
            val_data_list = grid_sampler(val_data_list)
            val_data_list = [d for d in val_data_list if len(d.origin_id) > 10]

        log.info("Saving val data for %s"%area)
        self._save_data(val_data_list, os.path.join(self.processed_dir, "val", area + ".pt"))
        tt = (datetime.datetime.utcnow() - st).total_seconds()
        self._pp_seconds += tt
        log.info("Completed preprocessing val data for %s in %.1f seconds !"%(area,tt))

    def process_test(self, area):
        st = datetime.datetime.utcnow()
        log.info("Starting test preprocessing on %s"%area)
        if os.path.exists(os.path.join(self.processed_dir, "test", area + ".pt")):
            log.info("Test preprocessing for %s already exists. Skipping."%area)
            return  # skip if already exists

        test_data_list = []
        manual_shift = None
        dirs = os.listdir(osp.join(self.raw_dir, area))
        for segment_name in dirs:
            log.info(f"Read test data {area}, {segment_name}")

            segment_path = osp.join(self.raw_dir, area, segment_name)
            if os.path.isdir(segment_path):

                s3_st = datetime.datetime.utcnow()
                xyz, rgb, semantic_labels, instance_labels, room_label, last_shift_vector = self.read_s3dis_format(
                    segment_path, segment_name, label_out=True, verbose=self.verbose, debug=self.debug,
                    manual_shift=manual_shift
                )
                s3_tt = (datetime.datetime.utcnow() - s3_st).total_seconds()
                log.info("s3dis test read for %s took %.1f seconds"%(area,s3_tt))

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
            log.info("Starting test pre_collate_transform for %s:"%area)
            log.info(self.pre_collate_transform)
            test_data_list = self.pre_collate_transform(test_data_list)

        grid_sampler = cT.GridSphereSampling(self._radius, self._radius, center=False)
        if self._sample_per_epoch > 0:
            test_data_list = self._gen_low_res(self.lowres_subsampling, test_data_list)
        else:
            test_data_list = grid_sampler(test_data_list)
            test_data_list = [d for d in test_data_list if len(d.origin_id) > 10]

        log.info("Saving test data for %s"%area)
        self._save_data(test_data_list, os.path.join(self.processed_dir, "test", area + ".pt"))
        tt = (datetime.datetime.utcnow() - st).total_seconds()
        self._pp_seconds += tt
        log.info("Completed preprocessing test data for %s in %.1f seconds !"%(area,tt))

    def process(self):
        if self._split == "train":
            this_func = self.process_train
            QUEUE = [x for x in self.train_areas]
        elif self._split == "val":
            this_func = self.process_val
            QUEUE = [x for x in self.val_areas]
        elif self._split == "test":
            this_func = self.process_test
            QUEUE = [x for x in self.test_areas]

        NPROC = os.cpu_count()

        THREADS = [None]*NPROC
        NCPU = -1

        st = datetime.datetime.utcnow()

        # Kick off processing threads
        while True:
            NCPU += 1
            if len(QUEUE) == 0 or NCPU >= NPROC:
                break # exit if no more items to queue or if all queue slots are full
            area = QUEUE.pop(0)
            THREADS[NCPU] = multiprocessing.Process(target=this_func,args=(area,))
            log.info("Queueing new job '%s:%s' in thread %d" % (self._split, area, NCPU))

        log.info("threading: queued %d jobs with %d jobs remaining" % (NCPU, len(QUEUE)))

        # Start threads now
        for t in THREADS:
            if t is not None:
                t.start()
        time.sleep(1) # wait a moment to make sure threads have started

        # Start waiting for threads and queueing new items if necessary
        while True:
            for t in range(len(THREADS)):
                if THREADS[t] is None: continue # skip empty threads
                if not THREADS[t].is_alive():
                    # try to queue another thread, or else continue
                    if len(QUEUE) > 0:
                        area = QUEUE.pop(0)
                        THREADS[t] = multiprocessing.Process(target=this_func,args=(area,))
                        log.info("Queueing new job '%s:%s' in thread %d" % (self._split, area, t))
                        THREADS[t].start()
                    else:
                        print("thread %d is done and no more jobs, closing."%t, flush=True)
                        THREADS[t] = None
                        continue

            # if all threads have died, then exit
            deadThreadCount = len([x for x in THREADS if x is None])
            if deadThreadCount >= NPROC:
                log.info("No threads left and no more queue items. Exiting threader.")
                break

            time.sleep(1) # check threads once per second

        tt = (datetime.datetime.utcnow() - st).total_seconds()
        log.info("Preprocessing for %s completed."%self._split)
        log.info("Wall clock time: %.1f seconds; total core time: %.1f seconds; speedup: %.2f" % (tt, self._pp_seconds, (self._pp_seconds / tt)))

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

    def object_name_to_label(self, object_class):
        """convert from object name in S3DIS to an int"""
        log.info(self.object_label_map.keys())
        for key in self.object_label_map.keys():
            if object_class.lower() in self.object_label_map[key]:
                return key
        # 0 reserved for other
        return 0

    # TODO segment maybe needed for very large files (100gb+)
    def read_s3dis_format(self, train_file, room_name, label_out=True, verbose=False, debug=False, manual_shift=None):
        k = 1
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
            print(f"loading {raw_path}")
            xyz = np.ascontiguousarray(room_ver[:, 0:3], dtype="float64")
            total_count = len(xyz)
            unq_index = np.unique(xyz, axis=0, return_index=True)[1]
            #keeping the same sort order runs much faster through knn
            unq_index = np.sort(unq_index, axis=0)
            xyz = np.ascontiguousarray(room_ver[unq_index, 0:3], dtype="float64")
            unq_count = len(xyz)
            print(f"Removed {total_count - unq_count} duplicate points")

            # Shifts to local and quantizes to float32 by default
            xyz, shift_vector = self.shift_and_quantize(xyz, manual_shift=manual_shift)
            shift_unq_index = np.unique(xyz, axis=0, return_index=True)[1]
            print(f"duplicates after shift: {unq_count - len(shift_unq_index)}")

            try:
                rgb = np.ascontiguousarray(room_ver[unq_index, 3:6], dtype="uint8")
            except ValueError:
                rgb = np.zeros((xyz.shape[0], 3), dtype="uint8")
                log.warning("WARN - corrupted rgb data for file %s" % raw_path)
            if not label_out:
                return xyz, rgb
            n_ver = len(xyz)
            del room_ver
            del unq_index
            nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(xyz)
            semantic_labels = np.zeros((n_ver,), dtype="int64")
            # semantic_labels[semantic_labels == 0] = -1
            room_label = np.asarray([room_label])
            instance_labels = np.zeros((n_ver,), dtype="int64")
            objects = glob.glob(osp.join(train_file, "annotations/*.txt"))
            i_object = 1
            for single_object in objects:
                object_name = os.path.splitext(os.path.basename(single_object))[0]
                print(f"Annotation: {object_name}")
                if object_name == "remainder":  # expand to a list
                    continue
                if verbose:
                    log.debug("adding object " + str(i_object) + " : " + object_name)
                object_class = object_name.split("_")[0]
                object_label = self.object_name_to_label(object_class)
                if object_label == 0:
                    continue
                obj_ver = pd.read_csv(single_object, sep=" ", header=None).values
                obj_xyz = np.ascontiguousarray(obj_ver[:, 0:3], dtype="float64")
                obj_unq_index = np.unique(obj_xyz, axis=0, return_index=True)[1]
                obj_unq_index = np.sort(obj_unq_index, axis=0)
                obj_xyz = obj_xyz[obj_unq_index]
                obj_xyz, _ = self.shift_and_quantize(obj_xyz, manual_shift=shift_vector)
                _, obj_ind = nn.kneighbors(obj_xyz)
                n = obj_ind[:, [0]]
                # all_ind = np.arange(len(obj_xyz))
                # for i in range(1, k):
                #     unq, n_unq_index = np.unique(n, return_index=True)
                #     dup_n_index = np.setdiff1d(all_ind, n_unq_index)
                #     print(f"duplicate after {i} passes: {len(dup_n_index)}")
                #     if(len(dup_n_index) <= 0):
                #         break
                #     n2 = obj_ind[dup_n_index, [i]]
                #     n[dup_n_index] = n2.reshape(n[dup_n_index].shape)

                semantic_labels[n] = object_label
                instance_labels[n] = i_object
                i_object = i_object + 1

            # classified_xyz = xyz[semantic_labels != -1]
            # classified_labels = xyz[semantic_labels != -1]
            # nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", p=1, radius=0.00000001, leaf_size=50).fit(classified_xyz)
            # unclassified_xyz = xyz[semantic_labels == -1]
            # classified_xyz_ind = nn.kneighbors(unclassified_xyz)


            # Shifts to local and quantizes to float32 by default
            # xyz, shift_vector = self.shift_and_quantize(xyz, manual_shift=manual_shift)

            return (
                torch.from_numpy(xyz),
                torch.from_numpy(rgb),
                torch.from_numpy(semantic_labels),  # actual label
                torch.from_numpy(instance_labels),  # index of instance
                torch.from_numpy(room_label),
                shift_vector
            )

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

    def to_ply(self, pos, label, file):
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
        log.info(f"Calculate Meta: {self.split_areas_paths}")
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

    # INV_OBJECT_LABEL = INV_OBJECT_LABEL


    FORWARD_CLASS = "forward.nexplore.NexploreS3DISFusedForwardDataset"

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        sampling_format = dataset_opt.get("sampling_format", "sphere")
        dataset_cls = NexploreS3DISSphere

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
        log.info(f"INV_OBJECT_LABEL_MAP: {INV_OBJECT_LABEL_MAP}")

        global S3DIS_NUM_CLASSES
        S3DIS_NUM_CLASSES = len(INV_OBJECT_LABEL.keys())

        for key in INV_OBJECT_LABEL:
            log.info(f"key: {key}")

        self.train_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=self.dataset_opt.train.samples_per_epoch,
            radius=self.dataset_opt.train.radius,
            split="train",
            lowres_subsampling=self.dataset_opt.lowres_subsampling,
            train_areas=dataset_opt.train.areas,
            val_areas=dataset_opt.val.areas,
            test_areas=dataset_opt.test.areas,
            pre_collate_transform=self.pre_collate_transform,
            transform=self.train_transform,
        )

        self.val_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=self.dataset_opt.val.samples_per_epoch,
            radius=self.dataset_opt.val.radius,
            split="val",
            lowres_subsampling=self.dataset_opt.lowres_subsampling,
            train_areas=dataset_opt.train.areas,
            val_areas=dataset_opt.val.areas,
            test_areas=dataset_opt.test.areas,
            pre_collate_transform=self.pre_collate_transform,
            transform=self.val_transform,
        )

        # self.test_dataset = dataset_cls(
        #     self._data_path,
        #     sample_per_epoch=self.dataset_opt.test.samples_per_epoch,
        #     radius=self.dataset_opt.test.radius,
        #     split="test",
        #     lowres_subsampling=self.dataset_opt.lowres_subsampling,
        #     train_areas=dataset_opt.train.areas,
        #     val_areas=dataset_opt.val.areas,
        #     test_areas=dataset_opt.test.areas,
        #     pre_collate_transform=self.pre_collate_transform,
        #     transform=self.test_transform,
        # )

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
