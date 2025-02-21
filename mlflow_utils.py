from __future__ import annotations

import contextlib
import logging
import os
import pickle
from datetime import datetime
from typing import Any

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.utilities as pl_utilities
import tqdm
from mlflow import tracking as mlflow_tracking
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import loggers as pl_loggers
from torch import Tensor

LOGGER = logging.getLogger(__name__)


def deserialize_tensors(t_list: list[str], desc: str | None = None) -> list[Tensor]:
    return [pickle.loads(t) for t in tqdm.tqdm(t_list, desc=desc)]


@contextlib.contextmanager
def temporary_logging_level(tmp_level: logging._Level):
    logger = logging.getLogger()
    old_level = logger.getEffectiveLevel()
    logger.setLevel(tmp_level)
    try:
        yield
    finally:
        logger.setLevel(old_level)


def setup_python_logging(level: logging._Level = logging.INFO) -> None:
    """Configures the basic settings for Python's logging module.

    Args:
        level (logging._Level, optional): Sets the threshold for the logging system.
        Default is logging.INFO.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

def get_mlflow_tracking_uri() -> str:
    """Activate AWS SIGV4 and returns the mlflow server endpoint.

    Returns:
    -------
    str
        MLflow tracking uri
    """
    os.environ["MLFLOW_TRACKING_AWS_SIGV4"] = "True"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    return "https://apiml.overmind.aqemia.com/mlflow"


def get_mlflow_logger(
    run_name: str, experiment_name: str = "diffsbdd"
) -> pl_loggers.MLFlowLogger:
    """Instanciate and return an MLFlow logger

    Args:
        run_name (str): name to provide to the run in mlflow
        experiment_name (str, optional): experiment name. Defaults to "interactions-pred".

    Returns:
        pl_loggers.MLFlowLogger: corresponing logger for Trainer module of Lightning
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_run_name = f"{run_name} {current_time}"

    LOGGER.info("experiment name on mlflow: %s", experiment_name)
    LOGGER.info("run name on mlflow: %s", new_run_name)
    tracking_uri = get_mlflow_tracking_uri()
    mlflow_logger = pl_loggers.MLFlowLogger(experiment_name, new_run_name, tracking_uri)
    return mlflow_logger


class MLFlowModelCheckpoint(pl_callbacks.ModelCheckpoint):
    def __init__(
        self, mlflow_logger: pl_loggers.MLFlowLogger, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mlflow_logger = mlflow_logger

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Checkpoints model on epoch end. Function called by all processes in DDP case.

        Parameters
        ----------
        trainer : pl.Trainer
        pl_module : pl.LightningModule
        """
        super().on_validation_end(trainer, pl_module)
        self.upload_best_model(self.mlflow_logger.run_id)

    @pl_utilities.rank_zero_only
    def upload_best_model(self, mlflow_run_id: str) -> None:
        """Upload best model to mlflow server, under specified run_id.

        Only rank zero process uploads to mlflow.
        If the path is empty (no best model yet), skips.

        Parameters
        ----------
        mlflow_run_id : str
            run_id to log artifact to
        """
        if self.best_model_path:
            self.mlflow_logger.experiment.log_artifact(
                mlflow_run_id, self.best_model_path
            )


def load_from_ckpt_if_exists(
    model: pl.LightningModule,
    mlflow_run_id: str,
    checkpoint_fname: str,
) -> pl.LightningModule:
    """Load a checkpoint saved on MLFlow.

    Args:
        model (pl.LightningModule): object representing the model
        mlflow_run_id (str): mlflow run id of the trained model
        checkpoint_fname (str): name of the checkpoint to use

    Returns:
        pl.LightningModule: _description_
    """
    mlflow_uri = get_mlflow_tracking_uri()
    client = mlflow_tracking.MlflowClient(tracking_uri=mlflow_uri)

    artifacts = [a.path for a in client.list_artifacts(mlflow_run_id)]
    if checkpoint_fname not in artifacts:
        LOGGER.warning(
            "Checkpoint file not found for run_id %s. Artifacts are %s",
            mlflow_run_id,
            artifacts,
        )
    else:
        local_ckpt_path = client.download_artifacts(mlflow_run_id, checkpoint_fname)
        model = model.load_from_checkpoint(checkpoint_path=local_ckpt_path)  # type: ignore
        LOGGER.info("Restored %s from run %s", checkpoint_fname, mlflow_run_id)

    return model