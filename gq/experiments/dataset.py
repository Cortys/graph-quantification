from torch import Tensor
from gq.data import DatasetManager
from gq.data import InMemoryDatasetProvider
from gq.data.quantification import add_tuplets_to_data, add_skewed_test_splits_to_data
from gq.utils import DataConfiguration


def set_num_left_out(data_cfg: DataConfiguration):
    """utility function setting the number-of-left-out classes for LOC experiments for each dataset accordingly

    Args:
        data_cfg (DataConfiguration): original data configuration

    Raises:
        ValueError: raised if unsupported dataset found
    """

    if data_cfg.dataset in ("Cora", "CoraML"):
        data_cfg.set_values(ood_num_left_out_classes=3)

    elif data_cfg.dataset == "CoraFull":
        data_cfg.set_values(ood_num_left_out_classes=30)

    elif "CiteSeer" in data_cfg.dataset:
        data_cfg.set_values(ood_num_left_out_classes=2)

    elif "PubMed" in data_cfg.dataset:
        data_cfg.set_values(ood_num_left_out_classes=1)

    elif data_cfg.dataset == "AmazonPhotos":
        data_cfg.set_values(ood_num_left_out_classes=3)

    elif data_cfg.dataset == "AmazonComputers":
        data_cfg.set_values(ood_num_left_out_classes=5)

    elif data_cfg.dataset == "ogbn-arxiv":
        data_cfg.set_values(ood_num_left_out_classes=15)

    elif data_cfg.dataset == "CoauthorPhysics":
        data_cfg.set_values(ood_num_left_out_classes=2)

    elif data_cfg.dataset == "CoauthorCS":
        data_cfg.set_values(ood_num_left_out_classes=4)

    else:
        raise ValueError(f"Dataset {data_cfg.dataset} not supported!")


class ExperimentDataset:
    """wrapper for dataset to be used in an experiment
    """

    def __init__(self, data_cfg: DataConfiguration, to_sparse: bool = False):
        self.data_cfg = data_cfg
        dataset = None

        for _ in range(data_cfg.split_no):
            dataset = DatasetManager(**data_cfg.to_dict())

        assert dataset is not None
        default_dataset = InMemoryDatasetProvider(dataset)

        self.dim_features = default_dataset.num_features
        self.num_classes = default_dataset.num_classes

        self.train_dataset = default_dataset
        self.train_val_dataset = default_dataset
        self.val_dataset = default_dataset
        self.ood_dataset = None

        self.to_sparse = to_sparse

        self.splits = ("train", "test", "val", "all")

        if to_sparse:
            self.train_dataset.to_sparse()

        if data_cfg.quantification_test_skew:
            self.setup_quantification_test_skew()

        if data_cfg.quantification_tuplets:
            self.setup_tuplets()

        # finally reset number of classes
        self.num_classes = self.train_dataset.num_classes

        # if nothing further specified: warmup/finetuning on training dataset
        self.warmup_dataset = self.train_dataset
        self.finetune_dataset = self.train_dataset

        self.train_loader = None
        self.train_val_loader = None
        self.val_loader = None
        self.ood_loader = None
        self.warmup_loader = None
        self.finetune_loader = None

        self.setup_loader()

    def setup_quantification_test_skew(self):
        strategy = self.data_cfg.quantification_test_skew_strategy
        repeats = self.data_cfg.quantification_test_skew_repeats
        tuplet_size = self.data_cfg.quantification_test_skew_tuplet_size
        depth_limit = self.data_cfg.quantification_test_skew_depth_limit
        noise = self.data_cfg.quantification_test_skew_noise
        assert repeats is not None and repeats > 0
        artifact_name = f"skewed_{strategy}_{self.data_cfg.split_no}_{repeats}_{tuplet_size}_{depth_limit}_{noise}"
        add_skewed_test_splits_to_data(
            self.val_dataset,
            strategy=strategy,
            repeats=repeats,
            tuplet_size=tuplet_size,
            depth_limit=depth_limit,
            noise=noise,
            artifact_name=artifact_name,
        )

    def setup_tuplets(self):
        train_oversampling_factor = (
            self.data_cfg.quantification_train_tuplet_oversampling_factor
        )
        if train_oversampling_factor is None:
            train_oversampling_factor = 1.0
        val_oversampling_factor = (
            self.data_cfg.quantification_val_tuplet_oversampling_factor
        )
        if val_oversampling_factor is None:
            val_oversampling_factor = 1.0
        test_oversampling_factor = (
            self.data_cfg.quantification_test_tuplet_oversampling_factor
        )
        if test_oversampling_factor is None:
            test_oversampling_factor = 1.0
        add_tuplets_to_data(
            self.train_dataset,
            train_tuplet_size=self.data_cfg.quantification_train_tuplet_size,
            val_tuplet_size=self.data_cfg.quantification_val_tuplet_size,
            test_tuplet_size=self.data_cfg.quantification_test_tuplet_size,
            train_oversampling_factor=train_oversampling_factor,
            val_oversampling_factor=val_oversampling_factor,
            test_oversampling_factor=test_oversampling_factor,
            train_strategy=self.data_cfg.quantification_train_tuplet_strategy,
        )

    def setup_loader(self):
        self.train_loader = self.train_dataset.loader()
        self.train_val_loader = self.train_val_dataset.loader()
        self.val_loader = self.val_dataset.loader()

        if self.ood_dataset is not None:
            self.ood_loader = self.ood_dataset.loader()
        else:
            self.ood_loader = None

        if self.warmup_dataset is not None:
            self.warmup_loader = self.warmup_dataset.loader()
        else:
            self.warmup_loader = None

        if self.finetune_dataset is not None:
            self.finetune_loader = self.finetune_dataset.loader()
        else:
            self.finetune_loader = None
