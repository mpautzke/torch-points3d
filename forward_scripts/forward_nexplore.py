import torch
import hydra
import logging
from omegaconf import OmegaConf
import os
import sys
import numpy as np
from typing import Dict
import pandas as pd
import math
import copy


DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

# Import building function for model and dataset
from torch_points3d.datasets.dataset_factory import instantiate_dataset, get_dataset_class
from torch_points3d.models.model_factory import instantiate_model

# Import BaseModel / BaseDataset for type checking
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

# Import from metrics
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

# Utils import
from torch_points3d.utils.colors import COLORS

from torch_geometric.nn import knn_interpolate
from timeit import default_timer as timer

from forward_scripts.faiss_knn import FaissKNeighbors

log = logging.getLogger(__name__)


def save(path, postfix, results):
    output_timer_start = timer()
    filename = "out"
    out_file = f"{filename}_{postfix}.txt"
    print(f"Writing {out_file}...")
    path = os.path.join(path, out_file)
    # np.save(path, results)  #These are faster
    np.savetxt(path, results, fmt='%s')
    # res_df = pd.DataFrame(results)
    # res_df.to_csv(path, sep=' ', index=False, header=False)
    output_timer_end = timer()
    print(f"{out_file} elapsed time: {round(output_timer_end - output_timer_start, 2)} seconds")


def run(model: BaseModel, dataset, device, output_path, process_full = True):
    loaders = dataset.test_dataloaders
    shifted_raw_data = dataset.test_dataset[0].shifted_test_data
    raw_data = dataset.test_dataset[0].raw_test_data
    results = {}
    for loader in loaders:
        loader.dataset.name
        with Ctq(loader) as tq_test_loader:
            for data in tq_test_loader:
                with torch.no_grad():
                    model.set_input(data, device)
                    model.forward()
                t_results = dataset.predict_original_samples(data, model.conv_type, model.get_output())
                for key, value in t_results.items():
                    results[key] = value

    print("Compiling subsampled points...")
    indices = list(results.keys())

    pos = np.array(raw_data.pos[indices], dtype=np.str)
    rgb = np.array(raw_data.rgb[indices], dtype=np.str)
    values = np.array(list(results.values()), dtype=np.str).reshape((-1, 1))

    subsampled = concatenate(pos, rgb, values)

    save(output_path,"subsampled", subsampled)
    del subsampled

    if not process_full:
        return

    fknn = FaissKNeighbors(k=5)  # use 1 to get exact or closest.  Use a higher number to remove noisy predictions
    # fknn.fit(np.array(sampled_pos), np.array(sampled_preds))
    fknn.fit(shifted_raw_data.pos[indices], np.array(list(results.values())))

    n = shifted_raw_data.pos.shape[0]
    batch_size = 65000
    batches = math.ceil(n / batch_size)

    raw_pos = np.array(shifted_raw_data.pos)
    prediction = np.array([], dtype=np.int)
    for a in Ctq(range(batches)):
        start = a * batch_size
        end = ((a+1) * batch_size)
        if end > n:
            end = n

        out = fknn.predict(raw_pos[start:end])
        prediction = np.concatenate((prediction, out))

    full_res = concatenate(raw_data.pos, raw_data.rgb.astype(dtype=np.str), prediction.reshape((-1, 1)).astype(dtype=np.str))

    save(output_path, "full", full_res)

def concatenate(pos, rgb, preds):
    pos_rgb = np.concatenate((pos, rgb), axis=1)
    pos_rgb_preds = np.concatenate((pos_rgb, preds), axis=1)

    return pos_rgb_preds

@hydra.main(config_path="conf/nexplore.yaml")
def main(cfg):
    compute_timer_start = timer()

    OmegaConf.set_struct(cfg, False)

    # Get device
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.cuda) else "cpu")
    log.info("DEVICE : {}".format(device))

    # Enable CUDNN BACKEND
    torch.backends.cudnn.enabled = cfg.enable_cudnn

    # Checkpoint
    checkpoint = ModelCheckpoint(cfg.checkpoint_dir, cfg.model_name, cfg.weight_name, strict=True)

    # Setup the dataset config
    # Generic config

    #checkpoint.data_config.test_transform[2].params.feat_names = "pos_z"
    #setattr(checkpoint.data_config.train_transform[7], "feat_names", "pos_z")
    #setattr(checkpoint.data_config.val_tranform[2], "feat_names", "pos_z")

    train_dataset_cls = get_dataset_class(checkpoint.data_config)
    # setattr(checkpoint.data_config, "class", "forward.nexplore.NexploreS3DISFusedForwardDataset")
    setattr(checkpoint.data_config, "class", train_dataset_cls.FORWARD_CLASS)
    setattr(checkpoint.data_config.test_transform[0], "lparams", [cfg.data.fixed_points])
    setattr(checkpoint.data_config, "first_subsampling", 0.08)
    setattr(checkpoint.data_config, "dataroot", cfg.input_path)
    setattr(checkpoint.data_config, "dataset_name", cfg.input_filename)


    # Datset specific configs
    if cfg.data:
        for key, value in cfg.data.items():
            checkpoint.data_config.update(key, value)
    if cfg.dataset_config:
        for key, value in cfg.dataset_config.items():
            checkpoint.dataset_properties.update(key, value)

    # Create dataset and model
    model = checkpoint.create_model(checkpoint.dataset_properties, weight_name=cfg.weight_name)
    log.info(model)
    log.info("Model size = %i", sum(param.numel() for param in model.parameters() if param.requires_grad))

    # Set dataloaders
    processing_timer_start = timer()
    dataset = instantiate_dataset(checkpoint.data_config)
    dataset.create_dataloaders(
        model, cfg.batch_size, cfg.shuffle, cfg.num_workers, False,
    )
    log.info(dataset)
    processin_timer_end = timer()

    print(f"Processing Time: {round(processin_timer_end - processing_timer_start, 2)} seconds")

    model.eval()
    if cfg.enable_dropout:
        model.enable_dropout_in_eval()
    model = model.to(device)

    # Run training / evaluation

    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path)

    run(model, dataset, device, cfg.output_path, cfg.process_full_resolution)

    compute_timer_end = timer()

    print(f"Elapsed Time: {round(compute_timer_end - compute_timer_start, 2)} seconds")


if __name__ == "__main__":
    main()
