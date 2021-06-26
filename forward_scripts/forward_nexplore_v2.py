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
ROOT = os.path.join(DIR, "../../../..")
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
    # np.savetxt(path, results) #this is faster
    res_df = pd.DataFrame(results)
    res_df.to_csv(path, sep=' ', index=False, header=False)
    output_timer_end = timer()
    print(f"{out_file} elapsed time: {round(output_timer_end - output_timer_start, 2)} seconds")


def run(model: BaseModel, dataset, device, output_path, confidence_threshold=0.0, include_labels = True):
    processing_start_end = timer()

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
                t_results = dataset.predict_original_samples(data, model.conv_type, model.get_output(), confidence_threshold)
                for key, value in t_results.items():
                    results[key] = value

    processing_timer_end = timer()

    print(f"Run Processing Time 1: {round(processing_timer_end - processing_start_end, 2)} seconds")

    all_points = np.where(shifted_raw_data.y != None)
    sampled_points = np.array(list(results.keys()))
    remainder_index = np.delete(all_points, sampled_points)

    AccuracyMatrix = np.zeros((3, 3), dtype=np.int32)

    results2 = list(results.items())
    npresults = np.array(results2)
    sampled_pos = raw_data.pos[np.array(sampled_points)]
    sampled_pos = sampled_pos.astype(np.float64)
    sampled_preds = npresults[:, 1]
    sampled_preds2 = sampled_preds.reshape((sampled_preds.shape[0], 1))
    subsampled = np.concatenate((sampled_pos, raw_data.rgb[np.array(sampled_points)], sampled_preds2), axis = 1)
    #subsampled = raw_data.pos[keyOriginal].tolist() + raw_data.rgb[keyOriginal].tolist() + [results[keyOriginal]])
    #AccuracyMatrix[int(results[keyOriginal]), int(raw_data.y[keyOriginal])] += 1

    #save(output_path, "subsampled_1", subsampled)

    processing_timer_end = timer()

    print(f"Run Processing Time 2: {round(processing_timer_end - processing_start_end, 2)} seconds")

    fknn = FaissKNeighbors(k=5)  # use 1 to get exact or closest.  Use a higher number to remove noisy predictions
    fknn.fit(np.array(sampled_pos), np.array(sampled_preds))

    n = len(remainder_index)
    batch_size = 65000
    batches = math.ceil(n / batch_size)

    raw_pos = raw_data.pos[np.array(remainder_index)]
    raw_rgb = raw_data.rgb[np.array(remainder_index)]
    # raw_y = raw_data.y[np.array(remainder_index)]
    prediction = np.array([])
    for a in Ctq(range(batches)):
        start = a * batch_size
        end = ((a+1) * batch_size)
        if (end > n):
            end = n

        out = fknn.predict(raw_pos[start:end])
        prediction = np.concatenate((prediction, out))

    #TODO make this more efficient.  O(N)
    # Probably could move to previous loop or use numpy instead of python list

    # KnnPoints = []

    print("adding full resolution points...")
    # for index, val in enumerate(Ctq(raw_pos)):
    #     KnnPoints.append(raw_data.pos[index].tolist() + raw_data.rgb[index].tolist() + [prediction[index]])
    #     AccuracyMatrix[int(prediction[index]), int(raw_data.y[index])] += 1

    # for index in range(raw_pos.shape[0]):
    #     KnnPoints.append(raw_pos[index].tolist() + raw_rgb[index].tolist() + [prediction[index]])
    #     AccuracyMatrix[int(prediction[index]), int(raw_y[index])] += 1

    prediction = prediction.reshape(prediction.shape[0], 1)
    subsampled2 = np.concatenate((raw_pos, raw_rgb, prediction), axis=1)
    subsampled = np.concatenate((subsampled, subsampled2))

    processing_timer_end = timer()

    print(f"Run Processing Time 3: {round(processing_timer_end - processing_start_end, 2)} seconds")

    fmt = '%s', '%s', '%s', '%d', '%d', '%d', '%d'

    np.savetxt(os.path.join(output_path, 'nptxt.txt'), subsampled, fmt=fmt)

    processing_timer_end = timer()

    print(f"Run Processing Time 4: {round(processing_timer_end - processing_start_end, 2)} seconds")

    if include_labels == True:
        remainder_index = remainder_index.reshape(remainder_index.shape[0], 1)

        remainder = np.concatenate((remainder_index, prediction), axis = 1)
        npresults = np.concatenate((npresults, remainder))

        dictnpresults = dict(npresults)
        dictnpresults = sorted(dictnpresults.items())
        dictnpresults = list(dictnpresults)
        dictnpresults = np.array(dictnpresults)
        raw_data_y = raw_data.y.reshape((raw_data.y.shape[0], 1))

        PredvsLabels = np.concatenate((dictnpresults, raw_data_y), axis=1)

        # printing accuracy:

        for a in range(3):
            if PredvsLabels[PredvsLabels[:,2]==a].shape[0] > 0:
                print(PredvsLabels[(PredvsLabels[:,2]==a) & (PredvsLabels[:,1]==a)].shape[0]
                  /PredvsLabels[PredvsLabels[:,2]==a].shape[0])
            else:
                print(0.0)

        processing_timer_end = timer()

    print(f"Run Processing Time Final: {round(processing_timer_end - processing_start_end, 2)} seconds")


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
    setattr(checkpoint.data_config, "class", train_dataset_cls.FORWARD_CLASS)
    setattr(checkpoint.data_config.test_transform[0], "lparams", [5000])
    # setattr(checkpoint.data_config, "first_subsampling", 1)
    setattr(checkpoint.data_config, "dataroot", cfg.input_path)
    setattr(checkpoint.data_config, "dataset_name", cfg.input_filename)
    setattr(checkpoint.data_config, "include_labels", cfg.include_labels)
    setattr(checkpoint.data_config, "confidence_threshold", cfg.confidence_threshold)


    # Datset specific configs
    if cfg.data:
        for key, value in cfg.data.items():
            checkpoint.data_config.update(key, value)
    if cfg.dataset_config:
        for key, value in cfg.dataset_config.items():
            checkpoint.dataset_properties.update(key, value)

    # Create dataset and mdoel
    model = checkpoint.create_model(checkpoint.dataset_properties, weight_name=cfg.weight_name)
    log.info(model)
    log.info("Model size = %i", sum(param.numel() for param in model.parameters() if param.requires_grad))

    processin_timer_end = timer()

    print(f"PreProcessing Time: {round(processin_timer_end - compute_timer_start, 2)} seconds")

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

    run(model, dataset, device, cfg.output_path, cfg.confidence_threshold, cfg.include_labels)

    compute_timer_end = timer()

    print(f"Elapsed Time: {round(compute_timer_end - compute_timer_start, 2)} seconds")


if __name__ == "__main__":
    main()
