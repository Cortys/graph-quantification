from collections.abc import Iterable
import json
import os
import time

import numpy as np
from torch import Tensor
import torch
from gq.models.model import Model

from gq.utils.prediction import Prediction
from gq.utils.storage import create_storage

os.environ["MKL_THREADING_LAYER"] = "GNU"

from typing import Callable, Optional, Dict, Any
from sacred import Experiment
import pyblaze.nn as xnn
from pyblaze.nn.engine._history import History
from pyblaze.utils.torch import gpu_device

import gq.nn as unn
from gq.models import create_model
from gq.models import EnergyScoring, DropoutEnsemble
from gq.utils import set_seed, ModelNotFoundError
from gq.nn import TransductiveGraphEngine
from gq.nn import get_callbacks_from_config
from gq.utils import RunConfiguration, DataConfiguration
from gq.utils import ModelConfiguration, TrainingConfiguration
from .dataset import ExperimentDataset


class TransductiveExperiment:
    """base experiment which works for default models and default GraphEngine"""

    def __init__(
        self,
        run_cfg: RunConfiguration,
        data_cfg: DataConfiguration,
        model_cfg: ModelConfiguration,
        train_cfg: TrainingConfiguration,
        ex: Optional[Experiment] = None,
        dataset: Optional[ExperimentDataset] = None,
        model: Optional[Model] = None,
    ):
        self.run_cfg = run_cfg
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.old_evaluation_results = None
        self.evaluation_results = None
        self.training_completed = False
        self.storage = None
        self.storage_params = None

        self.model = None
        self.ex = ex

        # metrics for evaluation of default graph
        # and id+ood splits combined for ood
        self.metrics = [
            "brier_score",
            "ece",
            "confidence_aleatoric_apr",
            "confidence_epistemic_apr",
            "confidence_structure_apr",
            "confidence_aleatoric_auroc",
            "confidence_epistemic_auroc",
            "confidence_structure_auroc",
            "ce",
            "avg_prediction_confidence_aleatoric",
            "avg_prediction_confidence_epistemic",
            "avg_sample_confidence_total",
            "avg_sample_confidence_total_entropy",
            "avg_sample_confidence_aleatoric",
            "avg_sample_confidence_aleatoric_entropy",
            "avg_sample_confidence_epistemic",
            "avg_sample_confidence_epistemic_entropy",
            "avg_sample_confidence_epistemic_entropy_diff",
            "avg_sample_confidence_features",
            "avg_sample_confidence_neighborhood",
            "average_entropy",
            # Learning to Quantify
            "quantification_cc",
            "quantification_pcc",
            "quantification_tuplets",
            # ARCs
            "accuracy_rejection_prediction_confidence_aleatoric",
            "accuracy_rejection_prediction_confidence_epistemic",
            "accuracy_rejection_sample_confidence_total",
            "accuracy_rejection_sample_confidence_total_entropy",
            "accuracy_rejection_sample_confidence_aleatoric",
            "accuracy_rejection_sample_confidence_aleatoric_entropy",
            "accuracy_rejection_sample_confidence_epistemic",
            "accuracy_rejection_sample_confidence_epistemic_entropy",
            "accuracy_rejection_sample_confidence_epistemic_entropy_diff",
        ]

        if self.run_cfg.no_quantification:
            self.metrics_with_val = []
            self.generic_metrics = []
        else:
            # Metrics for the test split that require, both, test and validation predictions/targets:
            self.metrics_with_val = [
                # Learning to Quantify
                "quantification_acc",
                "quantification_pacc",
                "quantification_dmy",
                "quantification_kdey",
            ]
            # Metrics which get the full (unmasked) predictions and data to compute metrics arbitrarily:

            self.generic_metrics = [
                "quantification_maj_neighbor",
                # "quantification_cluster_neighbor",
                # "quantification_edge",
                "quantification_weighted_dist",
                "quantification_ppr",
                "quantification_weighted_dist_neighbor",
                "quantification_ppr_neighbor",
                "quantification_kdey_ppr",
                "quantification_kdey_dist",
            ]

        if self.data_cfg.quantification_test_skew:
            self.generic_metrics = ["accuracy"] + self.generic_metrics
        else:
            self.metrics = ["accuracy"] + self.metrics

        if self.run_cfg.reduced_training_metrics:
            self.train_metrics = ["accuracy"]
            stopping_metric = self.train_cfg.stopping_metric
            if isinstance(stopping_metric, str):
                if stopping_metric.startswith("train_"):
                    stopping_metric = stopping_metric[6:]
                elif stopping_metric.startswith("val_"):
                    stopping_metric = stopping_metric[4:]
                elif stopping_metric.startswith("test_"):
                    stopping_metric = stopping_metric[5:]
            self.train_metrics.append(stopping_metric)
        else:
            # Leave out accuracy rejections curves for training
            self.train_metrics = self.metrics[:-9]

        self.ood_metrics = [
            # metrics for ood detection (id vs ood)
            "ood_detection_total_apr",
            "ood_detection_total_auroc",
            "ood_detection_total_entropy_apr",
            "ood_detection_total_entropy_auroc",
            "ood_detection_aleatoric_apr",
            "ood_detection_aleatoric_auroc",
            "ood_detection_aleatoric_entropy_apr",
            "ood_detection_aleatoric_entropy_auroc",
            "ood_detection_epistemic_apr",
            "ood_detection_epistemic_auroc",
            "ood_detection_epistemic_entropy_apr",
            "ood_detection_epistemic_entropy_auroc",
            "ood_detection_epistemic_entropy_diff_apr",
            "ood_detection_epistemic_entropy_diff_auroc",
            "ood_detection_features_apr",
            "ood_detection_features_auroc",
            "ood_detection_neighborhood_apr",
            "ood_detection_neighborhood_auroc",
            "ood_detection_structure_apr",
            "ood_detection_structure_auroc",
            # ood metrics
            "ood_accuracy",
            "ood_avg_prediction_confidence_aleatoric",
            "ood_avg_prediction_confidence_epistemic",
            "ood_avg_sample_confidence_total",
            "ood_avg_sample_confidence_total_entropy",
            "ood_avg_sample_confidence_aleatoric",
            "ood_avg_sample_confidence_aleatoric_entropy",
            "ood_avg_sample_confidence_epistemic",
            "ood_avg_sample_confidence_epistemic_entropy",
            "ood_avg_sample_confidence_epistemic_entropy_diff",
            "ood_avg_sample_confidence_neighborhood",
            "ood_avg_sample_confidence_features",
            "ood_average_entropy",
            # id metrics
            "id_accuracy",
            "id_avg_prediction_confidence_aleatoric",
            "id_avg_prediction_confidence_epistemic",
            "id_avg_sample_confidence_total",
            "id_avg_sample_confidence_total_entropy",
            "id_avg_sample_confidence_aleatoric",
            "id_avg_sample_confidence_aleatoric_entropy",
            "id_avg_sample_confidence_epistemic",
            "id_avg_sample_confidence_epistemic_entropy",
            "id_avg_sample_confidence_epistemic_entropy_diff",
            "id_avg_sample_confidence_features",
            "id_average_entropy",
        ]

        # base dataset
        set_seed(self.model_cfg.seed)

        self.setup_storage()

        if self.run_cfg.delete_run:
            return

        if self.evaluation_results is None and dataset is None:
            self.dataset = ExperimentDataset(
                self.data_cfg, to_sparse=self.data_cfg.to_sparse
            )
        else:
            self.dataset = dataset

        self.model = model

        if self.evaluation_results is None:
            assert self.dataset is not None
            self.model_cfg.set_values(
                dim_features=self.dataset.dim_features,
                num_classes=self.dataset.num_classes,
            )
            self.setup_model()
            self.setup_engine()

    def setup_storage(self):
        storage, storage_params = create_storage(
            self.run_cfg, self.data_cfg, self.model_cfg, self.train_cfg, ex=self.ex
        )
        self.storage = storage
        self.storage_params = storage_params

        if self.data_cfg.quantification_test_skew:
            quantification_test_skew_strategy = (
                self.data_cfg.quantification_test_skew_strategy
            )
            quantification_test_skew_repeats = (
                self.data_cfg.quantification_test_skew_repeats
            )
            quantification_test_skew_tuplet_size = (
                self.data_cfg.quantification_test_skew_tuplet_size
            )
            quantification_test_skew_depth_limit = (
                self.data_cfg.quantification_test_skew_depth_limit
            )
            quantification_test_skew_noise = (
                self.data_cfg.quantification_test_skew_noise
            )
        else:
            quantification_test_skew_strategy = None
            quantification_test_skew_repeats = None
            quantification_test_skew_tuplet_size = None
            quantification_test_skew_depth_limit = None
            quantification_test_skew_noise = None

        results_file_path = storage.create_results_file_path(
            self.model_cfg.model_name,
            storage_params,
            init_no=self.model_cfg.init_no,
            round_no=self.train_cfg.al_round,
            quantification_test_skew_strategy=quantification_test_skew_strategy,
            quantification_test_skew_repeats=quantification_test_skew_repeats,
            quantification_test_skew_tuplet_size=quantification_test_skew_tuplet_size,
            quantification_test_skew_depth_limit=quantification_test_skew_depth_limit,
            quantification_test_skew_noise=quantification_test_skew_noise,
            quantification_test_tuplet_oversampling_factor=self.data_cfg.quantification_test_tuplet_oversampling_factor,
            quantification_test_tuplet_size=self.data_cfg.quantification_test_tuplet_size,
        )
        self.results_file_path = results_file_path

        if self.train_cfg.al_round is not None:
            self.al_subset_file_path = storage.create_al_subset_file_path(
                self.model_cfg.model_name,
                storage_params,
                init_no=self.model_cfg.init_no,
                round_no=self.train_cfg.al_round,
            )
            self.al_next_subset_file_path = storage.create_al_subset_file_path(
                self.model_cfg.model_name,
                storage_params,
                init_no=self.model_cfg.init_no,
                round_no=self.train_cfg.al_round + 1,
            )
        else:
            self.al_subset_file_path = None
            self.al_next_subset_file_path = None

        if self.run_cfg.delete_run:
            try:
                model_file_path = storage.retrieve_model_file_path(
                    self.model_cfg.model_name,
                    storage_params,
                    init_no=self.model_cfg.init_no,
                    round_no=self.train_cfg.al_round,
                )
                print(f"Removing model: {model_file_path}")
                os.remove(model_file_path)
            except ModelNotFoundError:
                pass
            if os.path.exists(results_file_path):
                print(f"Removing results: {results_file_path}")
                os.remove(results_file_path)
            if self.al_subset_file_path is not None and os.path.exists(
                self.al_subset_file_path
            ):
                print(f"Removing AL subset: {self.al_subset_file_path}")
                os.remove(self.al_subset_file_path)
            return

        if (
            (not self.run_cfg.reeval or self.run_cfg.partial_reeval)
            and not self.run_cfg.retrain
            and not self.run_cfg.job == "predict"
            and os.path.exists(results_file_path)
        ):
            with open(results_file_path, "r") as f:
                self.old_evaluation_results: dict[str, Any] | None = json.load(f)
                if self.old_evaluation_results is not None:
                    # If cached eval results don't contain all metrics, re-evaluate
                    eval_res_keys = list(self.old_evaluation_results.keys())
                    if self.data_cfg.ood_flag:
                        metric_names = (
                            self.metrics
                            + self.metrics_with_val
                            + self.generic_metrics
                            + self.ood_metrics
                        )
                    else:
                        metric_names = (
                            self.metrics + self.metrics_with_val + self.generic_metrics
                        )
                    metrics = unn.get_metrics(metric_names).keys()
                    missing_metrics = set()
                    if self.run_cfg.reeval_metrics is not None:
                        if isinstance(self.run_cfg.reeval_metrics, str):
                            reeval_metrics = self.run_cfg.reeval_metrics.split(",")
                        elif isinstance(self.run_cfg.reeval_metrics, Iterable):
                            reeval_metrics = self.run_cfg.reeval_metrics
                        else:
                            raise ValueError(
                                f"Invalid reeval_metrics type: {type(self.run_cfg.reeval_metrics)}"
                            )

                        for metric in reeval_metrics:
                            missing_metrics.add(metric)

                    for metric in metrics:
                        found = False
                        for res_key in eval_res_keys:
                            if metric in res_key:
                                found = True
                                break
                        if not found:
                            missing_metrics.add(metric)

                    if len(missing_metrics) > 0:
                        mmstr = ", ".join(missing_metrics)
                        print(
                            f"Re-evaluating due to missing metrics {mmstr} in '{results_file_path}'."
                        )
                        if self.run_cfg.partial_reeval:
                            self.metrics = [
                                m for m in self.metrics if m in missing_metrics
                            ]
                            self.metrics_with_val = [
                                m for m in self.metrics_with_val if m in missing_metrics
                            ]
                            self.generic_metrics = [
                                m for m in self.generic_metrics if m in missing_metrics
                            ]
                            self.ood_metrics = [
                                m for m in self.ood_metrics if m in missing_metrics
                            ]
                        else:
                            self.old_evaluation_results = None
                    else:
                        self.evaluation_results = self.old_evaluation_results

    def setup_next_al_subset(
        self,
        compute_fn: Callable[[ExperimentDataset], Tensor],
    ):
        assert self.al_next_subset_file_path is not None
        assert self.dataset is not None

        al_mask = compute_fn(self.dataset)
        assert isinstance(al_mask, Tensor)
        with open(self.al_next_subset_file_path, "w") as f:
            json.dump(al_mask.tolist(), f)

    def setup_engine(self) -> None:
        assert self.dataset is not None
        self.engine = TransductiveGraphEngine(self.model, splits=self.dataset.splits)

    def setup_model(self) -> None:
        assert self.storage is not None
        assert self.storage_params is not None

        if self.run_cfg.eval_mode == "ensemble":
            self.run_cfg.set_values(save_model=False)

            # only allow creation of an ensemble when evaluating
            if self.run_cfg.job == "train":
                raise AssertionError

            if self.run_cfg.job in ("evaluate", "predict"):
                # model = Ensemble(self.model_cfg, models=None)
                raise NotImplementedError(
                    "Ensemble loading currently not supported (AL support missing)."
                )
            else:
                raise AssertionError

        else:
            if self.model is None:
                model = create_model(self.model_cfg)
            else:
                model = self.model

            if not self.run_cfg.retrain:
                try:
                    # if it is possible to load model: skip training
                    model_file_path = self.storage.retrieve_model_file_path(
                        self.model_cfg.model_name,
                        self.storage_params,
                        init_no=self.model_cfg.init_no,
                        round_no=self.train_cfg.al_round,
                    )
                    model.load_from_file(model_file_path)

                    if self.run_cfg.job == "train":
                        self.run_cfg.set_values(job="evaluate")
                    model.set_expects_training(False)
                    self.run_cfg.set_values(save_model=False)

                    if self.run_cfg.eval_mode == "dropout":
                        assert self.run_cfg.job in ("evaluate", "predict")
                        model = DropoutEnsemble(
                            model, num_samples=self.model_cfg.num_samples_dropout
                        )

                    elif self.run_cfg.eval_mode == "energy_scoring":
                        assert self.run_cfg.job in ("evaluate", "predict")
                        model = EnergyScoring(
                            model, temperature=self.model_cfg.temperature
                        )

                except ModelNotFoundError:
                    pass

        self.model = model

    def write_results(self, results):
        with open(self.results_file_path, "w") as f:
            json.dump(results, f)

    def evaluate(self) -> Dict[str, Any]:
        if self.evaluation_results is not None:
            return self.evaluation_results
        assert self.model is not None
        assert self.dataset is not None

        metrics = unn.get_metrics(self.metrics)
        metrics_with_val = unn.get_metrics(self.metrics_with_val)
        generic_metrics = unn.get_metrics(self.generic_metrics)
        eval_res = self.engine.evaluate(
            data=self.dataset.val_loader,
            metrics=metrics,
            metrics_with_val=metrics_with_val,
            generic_metrics=generic_metrics,
            gpu=self.run_cfg.gpu,
            old_results=self.old_evaluation_results,
            additional_params=dict(
                reduced_quantification=self.run_cfg.reduced_quantification
            ),
        )
        eval_val = eval_res["val"]
        eval_test = eval_res["test"]
        if self.old_evaluation_results is not None:
            results = self.old_evaluation_results
        else:
            results = {}
        results = results | {f"test_{k}": v for k, v in eval_test.items()}
        results = results | {f"val_{k}": v for k, v in eval_val.items()}

        if "all" in eval_res:
            eval_all = eval_res["all"]
            results = results | {f"all_{k}": v for k, v in eval_all.items()}

        self.write_results(results)

        return results

    def evaluate_ood(self) -> Dict[str, Any]:
        if self.evaluation_results is not None:
            return self.evaluation_results
        assert self.model is not None
        assert self.dataset is not None

        metrics = unn.get_metrics(self.metrics)
        metrics_with_val = unn.get_metrics(self.metrics_with_val)
        ood_metrics = unn.get_metrics(self.ood_metrics)

        # for isolated evaluation and poisoning experiments
        # target values are uses as ID values
        # for other cases, target usually represents both ID and OOD combined
        target_as_id = (self.data_cfg.ood_setting == "poisoning") or (
            self.data_cfg.ood_dataset_type == "isolated"
        )

        eval_res = self.engine.evaluate_target_and_ood(
            data=self.dataset.val_loader,
            data_ood=self.dataset.ood_loader,
            target_as_id=target_as_id,
            metrics=metrics,
            metrics_with_val=metrics_with_val,
            metrics_ood=ood_metrics,
            gpu=self.run_cfg.gpu,
        )

        eval_val = eval_res["val"]
        eval_test = eval_res["test"]
        if self.old_evaluation_results is not None:
            results = self.old_evaluation_results
        else:
            results = {}
        results = results | {f"test_{k}": v for k, v in eval_test.items()}
        results = results | {f"val_{k}": v for k, v in eval_val.items()}

        if "all" in eval_res:
            eval_all = eval_res["all"]
            results = results | {f"all_{k}": v for k, v in eval_all.items()}

        self.model.write_results(results)

        return results

    def predict(self):
        assert self.model is not None
        assert self.dataset is not None

        predictions = self.engine.predict(
            data=self.dataset.val_loader, gpu=self.run_cfg.gpu
        )

        return predictions

    def already_trained(self) -> bool:
        return (
            self.model is None
            or self.dataset is None
            or self.evaluation_results is not None
            or not self.model.expects_training()
        )

    def train(self) -> History | None:
        if self.already_trained():
            return None
        assert self.model is not None
        assert self.dataset is not None

        callbacks = []
        warmup_callbacks = []

        if self.run_cfg.log:
            batch_progress_logger = xnn.callbacks.BatchProgressLogger()
            callbacks.append(batch_progress_logger)
            warmup_callbacks.append(batch_progress_logger)

        metrics = unn.get_metrics(self.train_metrics)

        callbacks.extend(get_callbacks_from_config(self.train_cfg))

        # move training datasets to gpu before training
        gpu = self.engine._gpu_descriptor(self.run_cfg.gpu)
        device = gpu_device(gpu[0] if isinstance(gpu, list) else gpu)

        self.dataset.train_dataset.to(device)
        self.dataset.train_val_dataset.to(device)
        self.dataset.warmup_dataset.to(device)
        self.dataset.finetune_dataset.to(device)

        eval_every = self.train_cfg.eval_every

        if eval_every is None:
            eval_every = 1

        # ------------------------------------------------------------------------------------------------
        # warmup training
        warmup_epochs = (
            0 if self.train_cfg.warmup_epochs is None else self.train_cfg.warmup_epochs
        )
        if warmup_epochs > 0:
            # set-up optimizer
            optimizer = self.model.get_warmup_optimizer(
                self.train_cfg.lr, self.train_cfg.weight_decay
            )

            self.engine.model.set_warming_up(True)
            _ = self.engine.train(
                train_data=self.dataset.warmup_loader,
                val_data=self.dataset.train_val_loader,
                optimizer=optimizer,
                likelihood_optimizer=None,
                loss=self.model.warmup_loss,
                epochs=self.train_cfg.warmup_epochs,
                eval_every=eval_every,
                eval_train=True,
                callbacks=warmup_callbacks,
                metrics=metrics,
                gpu=self.run_cfg.gpu,
            )

            self.engine.model.set_warming_up(False)

        # ------------------------------------------------------------------------------------------------
        # main training loop training
        # set-up optimizer
        optimizer = self.model.get_optimizer(
            self.train_cfg.lr, self.train_cfg.weight_decay
        )
        likelihood_optimizer = None

        if isinstance(optimizer, (tuple, list)):
            likelihood_optimizer = optimizer[1]
            optimizer = optimizer[0]

        # default training
        history = self.engine.train(
            train_data=self.dataset.train_loader,
            val_data=self.dataset.train_val_loader,
            optimizer=optimizer,
            likelihood_optimizer=likelihood_optimizer,
            loss=self.model.loss,
            epochs=self.train_cfg.epochs,
            eval_every=eval_every,
            eval_train=True,
            callbacks=callbacks,
            metrics=metrics,
            gpu=self.run_cfg.gpu,
        )

        # ------------------------------------------------------------------------------------------------
        # finetuning
        finetune_epochs = (
            0
            if self.train_cfg.finetune_epochs is None
            else self.train_cfg.finetune_epochs
        )
        if finetune_epochs > 0:
            # set-up optimizer
            optimizer = self.model.get_finetune_optimizer(
                self.train_cfg.lr, self.train_cfg.weight_decay
            )
            likelihood_optimizer = None

            self.engine.model.set_finetuning(True)
            _ = self.engine.train(
                train_data=self.dataset.finetune_loader,
                val_data=self.dataset.train_val_loader,
                optimizer=optimizer,
                likelihood_optimizer=None,
                loss=self.model.finetune_loss,
                epochs=self.train_cfg.finetune_epochs,
                eval_every=eval_every,
                eval_train=True,
                callbacks=warmup_callbacks,
                metrics=metrics,
                gpu=self.run_cfg.gpu,
            )
            self.engine.model.set_finetuning(False)

        self.dataset.train_dataset.to("cpu")
        self.dataset.train_val_dataset.to("cpu")
        self.dataset.warmup_dataset.to("cpu")
        self.dataset.finetune_dataset.to("cpu")
        self.training_completed = True

        return history

    def run(
        self, after_train_callback: Callable | None = None
    ) -> Dict[str, Any] | list[Prediction]:
        assert not self.run_cfg.delete_run, "Cannot run experiments in delete mode."
        parts = [
            f"model={self.model_cfg.model_name}",
            f"dataset={self.data_cfg.dataset}",
            f"ood_type={self.data_cfg.ood_type}",
            f"split={self.data_cfg.split_no}",
            f"init={self.model_cfg.init_no}",
            f"results={self.results_file_path}",
            f"trained={self.already_trained()}",
            f"evaluated={self.evaluation_results is not None}",
            f"delete={self.run_cfg.delete_run}",
        ]

        exp_params = ", ".join(parts)
        print(f"Starting experiment ({exp_params}).")
        start_time = time.perf_counter()
        if self.run_cfg.job == "train":
            self.train()
            if after_train_callback is not None:
                after_train_callback()

        if self.data_cfg.ood_flag:
            assert self.run_cfg.job != "predict", "OOD not supported for predict job."
            results = self.evaluate_ood()
        else:
            if self.run_cfg.job == "predict":
                assert self.model is not None
                self.model.eval()
                results = self.predict()
            else:
                results = self.evaluate()

        # save trained model
        # or potential values to be cached
        # e.g. alpha_prior of y_soft
        if (
            self.model is not None
            and self.run_cfg.save_model
            and (
                self.training_completed
                or self.model_cfg.model_name in ("GDK", "GGP", "MaternGGP")
            )
        ):
            assert self.storage is not None
            assert self.storage_params is not None
            self.model.save_to_file(
                self.storage.create_model_file_path(
                    self.model_cfg.model_name,
                    self.storage_params,
                    init_no=self.model_cfg.init_no,
                    round_no=self.train_cfg.al_round,
                )
            )

        end_time = time.perf_counter()
        print(f"Completed experiment in {end_time - start_time:.2f}s ({exp_params}).")

        return results
