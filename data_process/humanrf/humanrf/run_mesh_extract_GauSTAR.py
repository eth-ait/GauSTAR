#!/usr/bin/env python3
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import yaml

import actorshq.evaluation.presets as presets
from actorshq.dataset.data_loader import DataLoader
from actorshq.dataset.trajectory import (
    get_trajectory_dataloader_from_calibration,
    get_trajectory_dataloader_from_keycams,
)
from actorshq.dataset.volumetric_dataset import VolumetricDataset
from actorshq.evaluation.evaluate import evaluate
from humanrf.adaptive_temporal_partitioning import compute_adaptive_segment_sizes
from humanrf.args.run_args import parse_args
from humanrf.scene_representation.humanrf import HumanRF
from humanrf.trainer import Trainer
from humanrf.utils.memory import collect_and_free_memory

if __name__ == "__main__":
    config = parse_args()

    # Set the seed for each possible source of random numbers.
    random.seed(config.random_seed)
    os.environ["PYTHONHASHSEED"] = str(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)

    frame_numbers = config.dataset.frame_numbers
    frame_range = (min(frame_numbers), max(frame_numbers) + 1)

    workspace = config.workspace
    workspace.mkdir(parents=True, exist_ok=True)

    with open(workspace / "config.yaml", "w") as f:
        yaml.dump(config, f)

    data_folder = config.dataset.path / config.dataset.actor / config.dataset.sequence / f"{config.dataset.scale}x"

    if config.model.temporal_partitioning == "none":
        segment_sizes = [len(frame_numbers)]
    elif config.model.temporal_partitioning == "adaptive":
        segment_sizes = compute_adaptive_segment_sizes(
            dataset=VolumetricDataset(data_folder),
            sorted_frame_numbers=frame_numbers,
            expansion_factor_threshold=config.model.expansion_factor_threshold,
        )
    elif config.model.temporal_partitioning == "fixed":
        fixed_size = config.model.fixed_segment_size
        segment_sizes = [fixed_size for _ in range(int(np.ceil(len(frame_numbers) / fixed_size)))]
    else:
        raise NotImplementedError("Unknown temporal partitioning type!")

    inputs = {
        "sorted_frame_numbers": tuple(sorted(frame_numbers)),
        "segment_sizes": tuple(segment_sizes),
        **vars(config.model),
    }
    model = HumanRF(**inputs)

    results_folder = workspace / "results"

    if config.evaluate:
        if config.evaluation.frame_numbers is not None:
            frame_numbers = config.evaluation.frame_numbers

        render_sequence_evaluation = presets.get_render_sequence(
            coverage=config.evaluation.coverage,
            camera_preset=config.evaluation.camera_preset,
            frame_numbers=frame_numbers,
        )
        evaluation_data_loader = DataLoader(
            dataset=VolumetricDataset(data_folder, crop_center_square=False),
            device=config.device,
            mode=DataLoader.Mode.TEST,
            dataloader_output_mode=DataLoader.OutputMode.RAYS_AND_SAMPLES,
            space_pruning_mode=DataLoader.SpacePruningMode.OCCUPANCY_GRID,
            batch_size=config.test.rays_batch_size,
            camera_numbers=presets.camera_configs[config.evaluation.camera_preset],
            frame_numbers=frame_numbers,
            max_buffer_size=1,
            render_sequence=render_sequence_evaluation,
            # filter_light_bloom=config.dataset.filter_light_bloom,  # ZCW add
        )

        trainer = Trainer(
            config=config,
            workspace=workspace,
            checkpoint=config.test.checkpoint,
            model=model,
            optimizer=None,
            lr_scheduler=None,
        )

        trainer.extract_geometry(evaluation_data_loader, results_folder, evaluation_data_loader.frame_numbers, 512)


    print("== End ==")
