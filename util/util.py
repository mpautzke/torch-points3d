import torch
import os
import numpy as np
from timeit import default_timer as timer

OBJECTS = {
    0: "zero",
    1: "one"
}

class Util():
    def __init__(self):
        pass

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

    def pt_to_text(self, path):
        b_path, f_name = os.path.split(path)

        d = torch.load(path)
        _data = d[0]
        _low_res = d[1]

        pos_out = np.array(_data.pos, dtype=np.str)
        rgb_out = np.array(_data.rgb, dtype=np.str)
        values_out = np.array(_data.y, dtype=np.str).reshape((-1, 1))
        out = self.concatenate(pos_out, rgb_out, values_out)
        self.save_file(b_path, f"{f_name}_highres_debug", out)

        pos_out = np.array(_low_res.pos, dtype=np.str)
        rgb_out = np.array(_low_res.rgb, dtype=np.str)
        values_out = np.array(_low_res.y, dtype=np.str).reshape((-1, 1))
        out = self.concatenate(pos_out, rgb_out, values_out)
        self.save_file(b_path, f"{f_name}_lowres_debug", out)

    def test(self):
        global OBJECTS
        OBJECTS = {2: "two", 3: "three"}

if __name__ == "__main__":
    util = Util()
    # util.pt_to_text("D:/SensatUrbanDataset/nexplores3disfused/processed/train/sensat_cambridge_10.pt")
    for key in OBJECTS.keys():
        print (f"key: {key}")

    util.test()

    for key in OBJECTS.keys():
        print (f"key: {key}")
