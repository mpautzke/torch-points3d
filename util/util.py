from __future__ import print_function
import sys
import torch
import os
import numpy as np
from timeit import default_timer as timer
import laspy
from plyfile import PlyData, PlyElement
import random
import torch_points3d.core.data_transform as cT
from torch_geometric.transforms import FixedPoints
import glob


from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

OBJECTS = {
    0: "zero",
    1: "one"
}

def save_file(path, filename, results):
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

class Util():
    def __init__(self):
        pass

    def dir_pt_to_text(self, dir, process_full_res=False):
        files = glob.glob(os.path.join(dir, "*.pt"))
        for file in files:
            self.pt_to_text(file, process_full_res)

    def pt_to_text(self, path, process_full_res=False):
        b_path, f_name = os.path.split(path)

        d = torch.load(path)
        _data = d[0]
        _low_res = d[1]

        pos_out = np.array(_low_res.pos, dtype=np.float32)
        rgb_out = np.array(np.array(_low_res.rgb * 255, dtype=np.int32), dtype=np.int32)
        values_out = np.array(_low_res.y, dtype=np.int32).reshape((-1, 1))
        out = np.concatenate([pos_out, rgb_out, values_out], axis=1, dtype=object)
        save_file(b_path, f"{f_name}_lowres_debug", out)

        if process_full_res:
            pos_out = np.array(_data.pos, dtype=np.float32)
            rgb_out = np.array(np.array(_data.rgb * 255, dtype=np.int32), dtype=np.int32)
            values_out = np.array(_data.y, dtype=np.int32).reshape((-1, 1))
            out = np.concatenate([pos_out, rgb_out, values_out], axis=1, dtype=object)
            save_file(b_path, f"{f_name}_highres_debug", out)

    def pt_to_txt_grid(self, path):
        b_path, f_name = os.path.split(path)
        spheres = torch.load(path)
        size = len(spheres)
        _data = spheres[random.randint(0, len(spheres))]

        pos_out = np.array(_data.pos, dtype=np.str)
        rgb_out = np.array(np.array(_data.rgb * 255, dtype=np.int), dtype=np.str)
        values_out = np.array(_data.y, dtype=np.str).reshape((-1, 1))
        out = np.concatenate([pos_out, rgb_out, values_out], axis=1, dtype=object)
        save_file(b_path, f"{f_name}_sphere_debug", out)

    def convert_laz_to_txt(self, path):
        b_path, f_name = os.path.split(path)
        las = laspy.read(path)
        xyz = np.array(np.vstack((las.x, las.y, las.z)).T, dtype=np.float64)

        # check if it has rgb data
        if all([x in [x for x in las.point_format.dimension_names] for x in ("red", "green", "blue")]):
            mode = "rgb"
        elif all([x in [x for x in las.point_format.dimension_names] for x in ("intensity")]):
            mode = "intensity"

        if mode == "rgb":
            _max = max(np.max(las.red), np.max(las.green), np.max(las.blue))
            _min = min(np.min(las.red), np.min(las.green), np.min(las.blue))
            diff = _max - _min
            rgb = np.array(np.vstack((las.red, las.green, las.blue)).T, dtype=np.float32)
            rgb -= _min
            rgb /= diff
            rgb *= 255
            rgb = rgb.astype(dtype=np.int32)
        else:
            # convert greyscale to rgb
            intensity = las.intensity
            max = np.max(intensity)
            intensity = intensity / max
            intensity = intensity * 255
            rgb = np.array(np.vstack((intensity, intensity, intensity)).T, dtype=np.int32)

        pt = np.concatenate([xyz, rgb], axis=1, dtype=object)
        save_file(b_path, f"{f_name}_las_debug", pt)

        return xyz, rgb

    def convert_ply_to_txt(self, path):
        plydata = PlyData.read(path)
        print("hi")

    def test(self):
        global OBJECTS
        OBJECTS = {2: "two", 3: "three"}

class SphereTest():

    def __init__(self, path, radius=[10, 20, 30, 40], iterations=1, fixed_points=10000):
        self._b_path, self._f_name = os.path.split(path)
        self._f_name = self._f_name.rsplit( ".", 1 )[0]
        self._path = path

        self.fixed_points = fixed_points
        self.fp = FixedPoints(fixed_points)
        self._radius = radius
        self._iterations = iterations
        self._sample_data = self._load_sample_data(self._path)

    def output_spheres(self):
        for i in range(0, self._iterations):
            spheres = self._get_random()
            for index, sphere in enumerate(spheres):

                pos_out = np.array(sphere.pos, dtype=np.float32)
                rgb_out = np.array(np.array(sphere.rgb * 255, dtype=np.int32), dtype=np.int32)
                values_out = np.array(sphere.y, dtype=np.int32).reshape((-1, 1))
                out = np.concatenate([pos_out, rgb_out, values_out], axis=1, dtype=object)
                save_file(self._b_path, f"{self._f_name}_sphere_r{self._radius[index]}_{i}", out)

                sphere_fp = self.fp(sphere)
                pos_out = np.array(sphere_fp.pos, dtype=np.float32)
                rgb_out = np.array(np.array(sphere_fp.rgb * 255, dtype=np.int32), dtype=np.int32)
                values_out = np.array(sphere_fp.y, dtype=np.int32).reshape((-1, 1))
                out = np.concatenate([pos_out, rgb_out, values_out], axis=1, dtype=object)
                save_file(self._b_path, f"{self._f_name}_sphere_r{self._radius[index]}_fp{self.fixed_points}_{i}", out)

    def test_random(self):
        choices = []
        for i in range(0, self._iterations):
            chosen_label = np.random.choice(self._labels, p=self._label_counts)
            choices.append(chosen_label)

        choices, counts = np.unique(choices, return_counts=True)

        print(choices)
        print(counts)


    def total_size(self, o, handlers={}, verbose=False):
        """ Returns the approximate memory footprint an object and all of its contents.

        Automatically finds the contents of the following builtin containers and
        their subclasses:  tuple, list, deque, dict, set and frozenset.
        To search other containers, add handlers to iterate over their contents:

            handlers = {SomeContainerClass: iter,
                        OtherContainerClass: OtherContainerClass.get_elements}

        """
        dict_handler = lambda d: chain.from_iterable(d.items())
        all_handlers = {tuple: iter,
                        list: iter,
                        deque: iter,
                        dict: dict_handler,
                        set: iter,
                        frozenset: iter,
                        }
        all_handlers.update(handlers)  # user handlers take precedence
        seen = set()  # track which object id's have already been seen
        default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

        def sizeof(o):
            if id(o) in seen:  # do not double count the same object
                return 0
            seen.add(id(o))
            s = getsizeof(o, default_size)

            if verbose:
                print(s, type(o), repr(o), file=stderr)

            for typ, handler in all_handlers.items():
                if isinstance(o, typ):
                    s += sum(map(sizeof, handler(o)))
                    break
            return s

        return sizeof(o)

    def test_size(self):
        for i in range(0, self._iterations):
            spheres = self._get_random()
            for index, sphere in enumerate(spheres):

                # totalbytes = 0
                # for key, item in sphere:
                #     print("size")
                #     totalbytes += sys.getsizeof(sphere[key])

                totalbytes = self.total_size(sphere, verbose=True)

                print(f"{self._radius[index]} R before: {totalbytes}")

                sphere_fp = self.fp(sphere)

                # totalbytes = 0
                # for key, item in sphere_fp:
                #     totalbytes += sys.getsizeof(sphere_fp[key])
                totalbytes = self.total_size(sphere_fp, verbose=True)

                print(f"{self._radius[index]} R after: {totalbytes}")



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

    #Implement using get random
    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        sphere = []
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        #TODO fail after timeout
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        for radii in self._radius:
            sphere_sampler = cT.SphereSampling(radii, centre[:3], align_origin=False)
            sphere_test = sphere_sampler(self._sample_data)
            sphere.append(sphere_test)
        return sphere

if __name__ == "__main__":
    util = Util()
    util.convert_laz_to_txt("C:/Users/mpautzke/Data/las/tule.las")
    # util.pt_to_text("C:/Users/mpautzke/Data/points3d/nexplores3disfused/processed/train/toronto.pt", process_full_res=True)
    # util.dir_pt_to_text("C:/Users/mpautzke/Data/points3d/nexplores3disfused/processed/train/", process_full_res=False)
    # util.convert_laz_to_txt('C:/Users/mpautzke/Downloads/SEGMENT_3.2_Ave_144_Tule_River_20191202_LAZ_WGS_84_UTM_zone_11N_56_855_052_points.las')
    # util.convert_ply_to_txt("C:/Users/mpautzke/Downloads/cambridge_block_7.ply")
    # util.pt_to_txt_grid("E:/SensatUrbanDataset/nexplores3disfused/raw/sensat_birminghan_val_01/segment_12/processed/segment_12.txt.pt")
    #
    # st = SphereTest(path="C:/Users/mpautzke/Data/points3d/nexplores3disfused/processed/train/toronto.pt", radius=[20], iterations=1, fixed_points=50000)
    # st.output_spheres()
    # st.test_random()

