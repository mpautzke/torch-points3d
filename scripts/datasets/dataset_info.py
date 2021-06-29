import os
from glob import glob


class DatasetInfo():
    def __init__(self):
        pass

    def point_count(self, paths):
        total_points = 0
        for path in paths:
            count = self.line_count(path)
            print(f"{path} {count}")
            total_points += count
        return total_points

    def line_count(self, path):
        count = 0
        with open(path) as fp:
            for line in fp:
                if line.strip():
                    count += 1

        return count

    def s3dis_point_count(self, root):
        #get area folders

        total_count = 0
        area_dirs = os.listdir(root)
        for area_dir_name in area_dirs:
            paths = []
            area_dir_path = os.path.join(root, area_dir_name)
            if os.path.isdir(area_dir_path):
                segment_dirs = os.listdir(area_dir_path)
                for segment_dir_name in segment_dirs:
                    segment_dir_path = os.path.join(area_dir_path, segment_dir_name)
                    paths.append(os.path.join(segment_dir_path, f"{segment_dir_name}.txt"))

            count = self.point_count(paths)
            total_count += count
            print(f"area-'{area_dir_name}'-point_count {count}")

        print(f"total_point_count {total_count}")

if __name__ == "__main__":
    di = DatasetInfo()
    di.s3dis_point_count(r"C:\Users\mpautzke\Data\points3d\nexplores3disfused\raw")





