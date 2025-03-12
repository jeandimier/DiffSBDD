import argparse
import logging
import os
import pdb
import sys
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml

import mlflow_utils as utils
from lightning_modules import LigandPocketDDPM
from mlflow_utils import get_mlflow_logger

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def merge_args_and_yaml(args, config_dict):
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if key in arg_dict:
            warnings.warn(
                f"Command line argument '{key}' (value: "
                f"{arg_dict[key]}) will be overwritten with value "
                f"{value} provided in the config file."
            )
        if isinstance(value, dict):
            arg_dict[key] = Namespace(**value)
        else:
            arg_dict[key] = value

    return args


def merge_configs(config, resume_config):
    for key, value in resume_config.items():
        if isinstance(value, Namespace):
            value = value.__dict__
        if key in config and config[key] != value:
            warnings.warn(
                f"Config parameter '{key}' (value: "
                f"{config[key]}) will be overwritten with value "
                f"{value} from the checkpoint."
            )
        config[key] = value
    return config


# ------------------------------------------------------------------------------
# Training
# ______________________________________________________________________________
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)

    assert "resume" not in config

    # Get main config
    ckpt_path = None if args.resume is None else Path(args.resume)
    if args.resume is not None:
        resume_config = torch.load(ckpt_path, map_location=torch.device("cpu"))[
            "hyper_parameters"
        ]

        config = merge_configs(config, resume_config)

    args = merge_args_and_yaml(args, config)

    out_dir = Path(args.logdir, args.run_name)
    histogram_file = Path(args.datadir, "size_distribution.npy")
    histogram = np.load(histogram_file).tolist()
    pl_module = LigandPocketDDPM(
        outdir=out_dir,
        dataset=args.dataset,
        datadir=args.datadir,
        batch_size=args.batch_size,
        lr=args.lr,
        egnn_params=args.egnn_params,
        diffusion_params=args.diffusion_params,
        num_workers=args.num_workers,
        augment_noise=args.augment_noise,
        augment_rotation=args.augment_rotation,
        clip_grad=args.clip_grad,
        eval_epochs=args.eval_epochs,
        eval_params=args.eval_params,
        visualize_sample_epoch=args.visualize_sample_epoch,
        visualize_chain_epoch=args.visualize_chain_epoch,
        auxiliary_loss=args.auxiliary_loss,
        loss_params=args.loss_params,
        mode=args.mode,
        node_histogram=histogram,
        pocket_representation=args.pocket_representation,
        virtual_nodes=args.virtual_nodes,
        whole_dataset=args.whole_dataset,
    )

    mlflow_logger = get_mlflow_logger("see_loss", experiment_name="diffsbdd")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(out_dir, "checkpoints"),
        filename="best-model-epoch={epoch:02d}",
        monitor="loss/val",
        save_top_k=1,
        save_last=True,
        mode="min",
    )
    callbacks = [
        utils.MLFlowModelCheckpoint(
            mlflow_logger,
            filename="best_val_loss_pos",
            monitor="error_t_lig_pos/val",
            mode="min",
        ),
        utils.MLFlowModelCheckpoint(
            mlflow_logger, filename="last_epoch", monitor="epoch", mode="max"
        ),
    ]

    # Log important files
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, __file__)
    with open(os.path.join("configs", args.config), "r") as f:
        mlflow_logger.experiment.log_artifact(
            mlflow_logger.run_id, os.path.join("configs", args.config)
        )

    which_gpus = list(range(args.gpus)) if isinstance(args.gpus, int) else args.gpus

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        logger=mlflow_logger,
        callbacks=callbacks,
        enable_progress_bar=args.enable_progress_bar,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accelerator="gpu",
        devices=which_gpus,
        strategy=("ddp" if len(which_gpus) > 1 else None),
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    warningNum = len(w)
    "your code probably throw warning"
    trainer.fit(model=pl_module, ckpt_path=ckpt_path)
    if len(w) != warningNum:
        warningNum = len(w)  # set break point here


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="element")
    # trainer.fit(model=pl_module, ckpt_path=ckpt_path)
