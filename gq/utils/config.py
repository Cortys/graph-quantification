from typing import Union, List, Tuple
import attr
from .object import HalfFrozenObject


@attr.s(frozen=True)
class RunConfiguration(HalfFrozenObject):
    """object specifying possible job configurations"""

    # experiment name (relevant for saving/loading trained models)
    experiment_name: str = attr.ib(default=None)

    # experiment-name for evaluation (relevant e.g. for evasion attacks)
    eval_experiment_name: str = attr.ib(default=None)

    # root directory to save/load models to/from
    experiment_directory: str = attr.ib(default=None)

    results_path: str = attr.ib(default=None)

    reduced_training_metrics: bool = attr.ib(default=False)

    # evaluation mode
    #   default: e.g. ood evasion or re-evaluation
    #   dropout: i.e. DropoutEnsemble evaluation
    #   ensemble: i.e. Ensemble evaluation (using model-init 1-10!!!)
    eval_mode: str = attr.ib(
        default=None,
        validator=lambda i, a, v: v
        in ("default",),
    )

    # flag whether to run a job as experiment ("train": training + evaluation)
    # or only in "evaluation" mode (e.g. re-evaluating model,
    # evlulating models on other datasets, or as dropout-models or ensembles)
    job: str = attr.ib(
        default=None, validator=lambda i, a, v: v in ("train", "evaluate", "predict")
    )

    # save-flag (e.g. for not saving GridSearch experiments)
    save_model: bool = attr.ib(default=None)
    delete_run: bool = attr.ib(default=None)
    retrain: bool = attr.ib(default=None)
    reeval: bool = attr.ib(default=None)
    partial_reeval: bool = attr.ib(default=None)
    reeval_metrics: list[str] | str = attr.ib(default=None)

    # gpu
    gpu: int = attr.ib(default=None, validator=lambda i, a, v: v in (0, False))

    # run multiple experiments at one
    num_inits: int = attr.ib(default=None)
    num_splits: int = attr.ib(default=None)
    per_init_variance: bool = attr.ib(default=False)

    # running experiment
    log: bool = attr.ib(default=True)  # flag for logging training progress and metrics
    debug: bool = attr.ib(default=True)  # flag for running code in a "DEBUG" mode
    ex_type: str = attr.ib(
        default="transductive",
        validator=lambda i, a, v: v in ("transductive",),  # type: ignore
    )  # type: ignore

    no_quantification: bool = attr.ib(default=False)
    # flag for evaluating quantification metrics
    reduced_quantification: bool = attr.ib(default=False)
    # whether to only evaluate quantification metrics with default params


@attr.s(frozen=True)
class DataConfiguration(HalfFrozenObject):
    """object specifying possible dataset configurations"""

    # sparseness
    to_sparse: bool = attr.ib(default=False)

    # randomness
    split_no: int = attr.ib(
        default=None, validator=lambda i, a, v: v is not None and v > 0
    )

    # dataset parameters
    dataset: str = attr.ib(default=None)
    root: str = attr.ib(default=None)
    split: str = attr.ib(
        default=None, validator=lambda i, a, v: v in ("public", "random")
    )
    # note that either the num-examples for the size values
    # must be specified, but not both at the same time!
    train_samples_per_class: Union[int, float] = attr.ib(default=None)
    val_samples_per_class: Union[int, float] = attr.ib(default=None)
    test_samples_per_class: Union[int, float] = attr.ib(default=None)
    train_size: float = attr.ib(default=None)
    val_size: float = attr.ib(default=None)
    test_size: float = attr.ib(default=None)

    # quantification tuplet preprocessing
    quantification_tuplets: bool = attr.ib(default=None)
    quantification_train_tuplet_strategy: str = attr.ib(
        default=None, validator=lambda i, a, v: v in ("random", "zipf")
    )
    quantification_train_tuplet_oversampling_factor: float = attr.ib(default=None)
    quantification_train_tuplet_size: int | float = attr.ib(default=None)
    quantification_val_tuplet_oversampling_factor: float = attr.ib(default=None)
    quantification_val_tuplet_size: int | float = attr.ib(default=None)
    quantification_test_tuplet_oversampling_factor: float = attr.ib(default=None)
    quantification_test_tuplet_size: int | float = attr.ib(default=None)

    # quantification skew preprocessing
    quantification_test_skew: bool = attr.ib(default=None)
    quantification_test_skew_strategy: str = attr.ib(
        default=None, validator=lambda i, a, v: v in ("zipf", "neighbor", "ppr")
    )
    quantification_test_skew_repeats: int = attr.ib(default=None)
    quantification_test_skew_tuplet_size: int = attr.ib(default=None)
    quantification_test_skew_depth_limit: int = attr.ib(default=None)
    quantification_test_skew_noise: float = attr.ib(default=None)

    @staticmethod
    def default_ignore() -> List[str]:
        """define default attributes to ignore when loading/storing models"""

        ignore = [
            "quantification_test_tuplet_oversampling_factor",
            "quantification_test_tuplet_size",
            "quantification_test_skew",
            "quantification_test_skew_strategy",
            "quantification_test_skew_repeats",
            "quantification_test_skew_tuplet_size",
            "quantification_test_skew_depth_limit",
            "quantification_test_skew_noise",
        ]

        return ignore


@attr.s(frozen=True)
class ModelConfiguration(HalfFrozenObject):
    """object specifying possible model configurations"""

    # model name
    model_name: str = attr.ib(
        default=None, validator=lambda i, a, v: v is not None and len(v) > 0
    )
    # randomness
    seed: int = attr.ib(default=None, validator=lambda i, a, v: v is not None and v > 0)
    init_no: int = attr.ib(
        default=None, validator=lambda i, a, v: v is not None and v > 0
    )

    # default parameters
    num_classes: int = attr.ib(default=None)
    dim_features: int = attr.ib(default=None)
    dim_hidden: Union[int, List[int]] = attr.ib(default=None)
    dropout_prob: float = attr.ib(default=None)
    dropout_prob_adj: float = attr.ib(default=0.0)
    # mainly relevant for ogbn-arxiv
    batch_norm: bool = attr.ib(default=None)
    entropy_num_samples: int | None = attr.ib(default=None)

    # for constrained linear layers
    k_lipschitz: float = attr.ib(default=None)

    # for deeper networks
    num_layers: int = attr.ib(default=None)

    # GAT
    heads_conv1: int = attr.ib(default=None)
    heads_conv2: int = attr.ib(default=None)
    negative_slope: float = attr.ib(default=None)
    coefficient_dropout_prob: float = attr.ib(default=None)

    # diffusion
    K: int = attr.ib(default=None)
    alpha_teleport: float = attr.ib(default=None)
    add_self_loops: bool = attr.ib(default=None)
    adj_normalization: str = attr.ib(default=None)
    sparse_propagation: bool = attr.ib(default=None)
    sparse_x_prune_threshold: float = attr.ib(default=None)

    # RGCN
    gamma: float = attr.ib(default=None)
    beta_kl: float = attr.ib(default=None)
    beta_reg: float = attr.ib(default=None)

    # DUN
    beta_dun: float = attr.ib(default=None)
    depth_in_message_passing: bool = attr.ib(default=None)

    # SGCN
    teacher_training: bool = attr.ib(default=None)
    teacher_params: dict = attr.ib(default=None)
    use_bayesian_dropout: bool = attr.ib(default=None)
    use_kernel: bool = attr.ib(default=None)
    lambda_1: float = attr.ib(default=None)
    sample_method: str = attr.ib(
        default=None,
        validator=lambda i, a, v: v in (None, "log_evidence", "alpha", "none"),
    )
    epochs: int = attr.ib(default=None)

    # dropout / ensemble
    num_samples_dropout: int = attr.ib(default=None)
    ensemble_min_init_no: int = attr.ib(default=None)
    ensemble_max_init_no: int = attr.ib(default=None)

    # scoring
    temperature: float = attr.ib(default=None)

    # quantification
    quantification_tuplet_aggregation: str = attr.ib(
        default=None, validator=lambda i, a, v: v in (None, "mean", "nn")
    )

    @staticmethod
    def default_ignore() -> List[str]:
        """define default attributes to ignore when loading/storing models"""

        ignore = [
            "temperature",
            "ensemble_max_init_no",
            "ensemble_min_init_no",
            "num_samples_dropout",
            "init_no",
            "dim_features",
            "num_classes",
            "entropy_num_samples",
        ]

        return ignore


@attr.s(frozen=True)
class TrainingConfiguration(HalfFrozenObject):
    """object specifying possible training configurations"""

    lr: float = attr.ib(default=None)
    weight_decay: float = attr.ib(default=None)
    epochs: int = attr.ib(default=None)
    warmup_epochs: int = attr.ib(default=None)
    finetune_epochs: int = attr.ib(default=None)
    stopping_mode: str = attr.ib(
        default=None,
        validator=lambda i, a, v: v in (None, "default", "average", "multiple"),
    )
    stopping_patience: int = attr.ib(default=None)
    stopping_restore_best: bool = attr.ib(default=None)
    stopping_metric: str = attr.ib(default=None)
    stopping_minimize: bool = attr.ib(default=None)
    eval_every: int = attr.ib(default=None)

    @staticmethod
    def default_ignore() -> List[str]:
        """define default attributes to ignore when loading/storing runs"""

        ignore = ["al_round"]

        return ignore


def configs_from_dict(
    d: dict,
) -> Tuple[
    RunConfiguration, DataConfiguration, ModelConfiguration, TrainingConfiguration
]:
    """utility function converting a dictionary (e.g. coming from a .yaml file) into the corresponding configuration objects

    Args:
        d (dict): dictionary containing all relevant configuration parameters

    Returns:
        Tuple[RunConfiguration, DataConfiguration, ModelConfiguration, TrainingConfiguration]: tuple of corresponding objects for run, data, model, and training configuration
    """
    run = RunConfiguration(**d["run"])
    data = DataConfiguration(**d["data"])
    model = ModelConfiguration(**d["model"])
    training = TrainingConfiguration(**d["training"])

    return run, data, model, training
