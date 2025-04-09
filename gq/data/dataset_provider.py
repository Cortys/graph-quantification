import copy
import os
from typing import Any, Callable
import torch
import torch_geometric.data as td
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToSparseTensor
import numpy as np
import gq.data as ud
import gq.data.data_artifacts as artifacts


class InMemoryDatasetProvider(td.InMemoryDataset):
    """InMemoryDatasetProvider

    Wrapper for a torch_geometric dataset which makes it compatible to our pipeline intended for usage with different OOD datasets.
    """

    def __init__(self, dataset: td.InMemoryDataset):
        super().__init__()

        self.data_list: list[td.Data] = list(dataset)  # type: ignore
        self._num_classes = dataset.num_classes
        self._num_features = dataset.num_features
        self._to_sparse = ToSparseTensor(remove_edge_index=True, fill_cache=True)
        self._processed_dir = dataset.processed_dir

    @property
    def num_classes(self):
        return self._num_classes

    def set_num_classes(self, n_c: int):
        self._num_classes = n_c

    @property
    def num_features(self):
        return self._num_features

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def loader(self, batch_size=1, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def clone(self, shallow=False):
        self_clone = copy.copy(self)
        if not shallow:
            self_clone.data_list = [d.clone() for d in self.data_list]

        return self_clone

    def to(self, device, **kwargs):
        for i, l in enumerate(self.data_list):
            self.data_list[i] = l.to(device, **kwargs)

        return self

    def to_sparse(self):
        for i, l in enumerate(self.data_list):
            self.data_list[i] = self._to_sparse(l)

        return self

    def get_artifact(
        self,
        name,
        compute_artifact: Callable[["InMemoryDatasetProvider"], Any] | None = None,
    ) -> Any:
        npy_artifact_path = os.path.join(self._processed_dir, name + ".npy")
        if os.path.exists(npy_artifact_path):
            return np.load(npy_artifact_path)

        artifact_path = os.path.join(self._processed_dir, name + ".pt")
        if os.path.exists(artifact_path):
            return torch.load(artifact_path, weights_only=False)

        if compute_artifact is None:
            if name == "apsp":
                compute_artifact = artifacts.compute_apsp

        assert (
            compute_artifact is not None
        ), f"Artifact {name} not found and no function to compute it provided."

        artifact = compute_artifact(self)

        if isinstance(artifact, np.ndarray):
            np.save(npy_artifact_path, artifact)
        else:
            torch.save(artifact, artifact_path)

        return artifact

    def get_apsp(self) -> np.ndarray:
        return self.get_artifact("apsp")
