import torch
import numpy as np
from timeit import default_timer as timer
import os


class Export:
    def __init__(self):
        pass

    def pt_to_ascii(self, in_path, out_path):
        d = torch.load(in_path)
        _data = d[0]

        if len(d) > 1:
            _low_res = d[1]
            res = self.concatenate(_low_res.pos, _low_res.rgb, _low_res.y)
            self.save(out_path, "export", res)

    def save(self, path, postfix, results):
        output_timer_start = timer()
        filename = "out"
        out_file = f"{filename}_{postfix}.txt"
        print(f"Writing {out_file}...")
        path = os.path.join(path, out_file)
        np.savetxt(path, results, fmt='%s')
        output_timer_end = timer()
        print(f"{out_file} elapsed time: {round(output_timer_end - output_timer_start, 2)} seconds")

    def concatenate(self, pos, rgb, preds):
        pos_rgb = np.concatenate((pos, rgb), axis=1)
        pos_rgb_preds = np.concatenate((pos_rgb, preds), axis=1)

        return pos_rgb_preds


if __name__ == "__main__":
    e = Export()
    e.pt_to_ascii("C:/Users/mpautzke/Data/points3d/nexplores3disfused/processed/train/a40.pt",
                  "C:/Users/mpautzke/Data/points3d/nexplores3disfused/processed/train/a40.txt")
