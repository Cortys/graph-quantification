import json
import os
from typing import OrderedDict
import warnings

from sklearn.exceptions import UndefinedMetricWarning

os.environ["MKL_THREADING_LAYER"] = "GNU"

import logging
import torch
import pandas as pd
import numpy as np

from collections import OrderedDict
from sacred import Experiment
from gpn.utils import RunConfiguration, DataConfiguration
from gpn.utils import ModelConfiguration, TrainingConfiguration
from gpn.utils import results_dict_to_df
from gpn.experiments import MultipleRunExperiment


ex = Experiment()

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


@ex.config
def config():
    # pylint: disable=missing-function-docstring
    overwrite = None
    db_collection = None


@ex.automain
def run_experiment(
    run: dict, data: dict, model: dict, training: dict
) -> dict | list | pd.DataFrame | None:
    """main function to run experiment with sacred support

    Args:
        run (dict): configuration parameters of the job to run
        data (dict): configuration parameters of the data
        model (dict): configuration parameters of the model
        training (dict): configuration paramterers of the training

    Returns:
        dict: numerical results of the evaluation metrics for different splits
    """

    run_cfg = RunConfiguration(**run)
    data_cfg = DataConfiguration(**data)
    model_cfg = ModelConfiguration(**model)
    train_cfg = TrainingConfiguration(**training)

    if torch.cuda.device_count() <= 0:
        run_cfg.set_values(gpu=False)

    logging.info("Received the following configuration:")
    logging.info("RUN")
    logging.info(run_cfg.to_dict())
    logging.info("-----------------------------------------")
    logging.info("DATA")
    logging.info(data_cfg.to_dict())
    logging.info("-----------------------------------------")
    logging.info("MODEL")
    logging.info(model_cfg.to_dict())
    logging.info("-----------------------------------------")
    logging.info("TRAINING")
    logging.info(train_cfg.to_dict())
    logging.info("-----------------------------------------")

    experiment = MultipleRunExperiment(run_cfg, data_cfg, model_cfg, train_cfg, ex=ex)

    results = experiment.run()

    if run_cfg.delete_run:
        if run_cfg.results_path is not None and os.path.exists(run_cfg.results_path):
            os.remove(run_cfg.results_path)

        print("Deleted individual and aggregated runs.")
        return

    if run_cfg.job == "predict":
        return results

    assert isinstance(results, dict)

    if run_cfg.results_path is not None:
        df = results_dict_to_df(
            results, per_init_var=run_cfg.per_init_variance, num_inits=run_cfg.num_inits
        )

        # print()
        # print(df.to_markdown())
        # print()

        if run_cfg.results_path == "":
            return df

        df.to_json(path_or_buf=run_cfg.results_path, orient="columns", indent=4)
    else:
        return results
