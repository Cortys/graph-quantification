import cupy
from typing import Any
import cudf
import pylibcugraph as libcugraph
import numpy as np
import scipy.sparse as sp
import torch
from torch_sparse import SparseTensor
import joblib as jl

from gq.data.dataset_provider import InMemoryDatasetProvider
from gq.layers.appnp_propagation import APPNPPropagation
from gq.utils.graphs import edge_index_to_cugraph
from gq.utils.utils import bincount_last_axis


def zipf_frequencies(m: int, c: int, z: float | np.ndarray) -> np.ndarray:
    """Generates an array of frequencies that follow a Zipf distribution.

    Args:
        m (int): Size of Zipf samples.
        c (int): Number of classes.
        z (float): Zipf distribution parameter.

    Returns:
        ndarray: Array of frequencies that follow a Zipf distribution.
    """
    vals = np.reshape((np.arange(c) + 1), (1, -1)) ** np.reshape(z, (-1, 1))
    vals = m * vals / (vals.sum(axis=-1, keepdims=True))
    int_vals = np.floor(vals).astype(int)
    sum_diffs = m - np.sum(int_vals, axis=-1)
    diffs = vals - int_vals
    diff_idxs = np.argsort(diffs, axis=-1)[:, ::-1]

    for i in range(c):
        mask = sum_diffs > i
        if mask.sum() == 0:
            break
        int_vals[mask, diff_idxs[mask, i]] += 1

    return int_vals


def partition_labels(idxs, labels, num_classes):
    n = labels.shape[0]
    r = np.arange(n)
    S = sp.csr_matrix((idxs, [labels, r]), shape=(num_classes, n))
    class_partitions = np.split(S.data, S.indptr[1:-1])

    return class_partitions


def generate_zipf_tuplets(
    idxs: np.ndarray,
    labels: np.ndarray,
    tuplet_size: int,
    num_classes: int,
    oversampling_factor: float = 1.0,
    tuplet_count: int | None = None,
    seed=1337,
) -> tuple[np.ndarray, np.ndarray]:
    class_partitions = partition_labels(idxs, labels, num_classes)
    rnd = np.random.default_rng(seed)
    for part in class_partitions:
        rnd.shuffle(part)
    n = labels.shape[0]

    tuplet_count = (
        int(oversampling_factor * n / tuplet_size)
        if tuplet_count is None
        else tuplet_count
    )
    z = -np.log(1 - rnd.uniform(size=(tuplet_count,)))
    z[z == np.inf] = 0.0
    zipf_freqs = zipf_frequencies(tuplet_size, num_classes, z)
    rnd.permuted(zipf_freqs, axis=-1, out=zipf_freqs)
    zipf_row_offsets = np.cumsum(zipf_freqs, axis=-1)
    tuplets = np.empty(shape=(tuplet_count, tuplet_size), dtype=int)
    tuplet_dists = zipf_freqs / tuplet_size

    for j, part in enumerate(class_partitions):
        offset = 0
        for i in range(tuplet_count):
            next_offset = offset + zipf_freqs[i, j]
            tup_offset = zipf_row_offsets[i, j - 1] if j > 0 else 0
            tup_offset_end = zipf_row_offsets[i, j]
            tuplets[i, tup_offset:tup_offset_end] = np.take(
                part, range(offset, next_offset), mode="wrap"
            )
            offset = next_offset

    return tuplets, tuplet_dists


def generate_random_tuplets(
    idxs: np.ndarray,
    tuplet_size: int,
    oversampling_factor: float = 1.0,
    seed=1337,
):
    rnd = np.random.default_rng(seed)
    n = idxs.shape[0]
    tuplet_count = n // tuplet_size
    tuplets_list = []
    while oversampling_factor > 0:
        if oversampling_factor < 1:
            tuplet_count = int(oversampling_factor * n / tuplet_size)
        tuplets = rnd.choice(idxs, size=(tuplet_count, tuplet_size), replace=False)
        tuplets_list.append(tuplets)
        oversampling_factor -= 1
    tuplets = np.concatenate(tuplets_list, axis=0)

    return tuplets


def generate_neighborhood_tuplets(
    idxs: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    start_nodes_per_class: int,
    tuplet_size: int,
    num_classes: int,
    num_nodes: int,
    noise: float | int,
    depth_limit: int,
    seed=1337,
):
    class_partitions = partition_labels(idxs, labels[idxs], num_classes)
    rnd = np.random.default_rng(seed)
    for part in class_partitions:
        rnd.shuffle(part)
    start_nodes = np.concatenate(
        [part[: min(start_nodes_per_class, len(part))] for part in class_partitions],
        dtype=np.int32,
    )
    node_subset_mask = cupy.zeros(num_nodes, dtype=np.bool_)
    node_subset_mask[cupy.asarray(idxs.to(torch.int32))] = True
    edge_index = cupy.asarray(edge_index.to(torch.int32))
    edge_sub_mask = node_subset_mask[edge_index[0]] & node_subset_mask[edge_index[1]]
    edge_index_sub = edge_index[:, edge_sub_mask]
    non_isolated_node_mask = cupy.zeros(num_nodes, dtype=np.bool_)
    non_isolated_node_mask[edge_index_sub.reshape(-1)] = True

    g, handle = edge_index_to_cugraph(edge_index_sub)

    def compute_bfs(start_node):
        if not non_isolated_node_mask[start_node]:
            # If start node is isolated, return start node as all neighbors
            return np.full(tuplet_size, start_node)
        distances, _, _ = libcugraph.bfs(
            handle=handle,
            graph=g,
            sources=cudf.Series([start_node], dtype="int32"),
            direction_optimizing=True,
            depth_limit=depth_limit,
            compute_predecessors=False,
            do_expensive_check=False,
        )
        if noise > 0:
            rnd: cupy.random.Generator = cupy.random.default_rng(seed + start_node)
            distances = distances + noise * rnd.standard_normal(
                size=distances.size, dtype=cupy.float32
            )
        perm = cupy.argsort(distances)
        limit = cupy.searchsorted(
            distances, cupy.array(depth_limit), side="right", sorter=perm
        )
        perm = perm[: min(tuplet_size, limit)]
        neighbors = perm.get()  #  Sort idxs correspond to vertex idxs: Use directly!
        if neighbors.size >= tuplet_size:
            return neighbors[:tuplet_size]
        return np.concatenate(
            [neighbors, np.full(tuplet_size - neighbors.size, start_node)]
        )

    tuplets = np.asarray(
        # Dont use parallelism here, as it does not result in speedup
        jl.Parallel(n_jobs=1, backend="threading")(
            jl.delayed(compute_bfs)(start_node) for start_node in start_nodes
        )
    )

    return tuplets


def generate_full_neighborhood_tuplets(
    idxs: torch.Tensor,
    distances: np.ndarray,
    labels: torch.Tensor,
    start_nodes_per_class: int,
    tuplet_size: int,
    num_classes: int,
    num_nodes: int,
    noise: float | int,
    depth_limit: int,
    seed=1337,
):
    class_partitions = partition_labels(idxs, labels[idxs], num_classes)
    rnd = np.random.default_rng(seed)
    for part in class_partitions:
        rnd.shuffle(part)
    start_nodes = np.concatenate(
        [part[: min(start_nodes_per_class, len(part))] for part in class_partitions],
        dtype=np.int32,
    )
    num_start_nodes = len(start_nodes)
    rnd: np.random.Generator = np.random.default_rng(seed)
    filtered_distances = np.full(
        (num_start_nodes, distances.shape[-1]), np.iinfo(np.int32).max, dtype=np.int32
    )
    filtered_distances[:, idxs] = distances[start_nodes][:, idxs]

    if noise > 0:
        filtered_distances = filtered_distances + noise * rnd.standard_normal(
            size=filtered_distances.shape, dtype=cupy.float32
        )

    perms = np.argsort(filtered_distances, axis=-1)

    tuplets = []

    for i in range(num_start_nodes):
        dist = filtered_distances[i]
        perm = perms[i]
        limit = np.searchsorted(dist, depth_limit, side="right", sorter=perm)
        perm = perm[: min(tuplet_size, limit)]
        neighbors = perm  #  Sort idxs correspond to vertex idxs: Use directly!
        if neighbors.size >= tuplet_size:
            neighbors = neighbors[:tuplet_size]
        elif neighbors.size < tuplet_size:
            neighbors = np.concatenate(
                [neighbors, np.full(tuplet_size - neighbors.size, start_nodes[i])]
            )
        tuplets.append(neighbors)

    return np.vstack(tuplets, dtype=np.int32)


def generate_ppr_tuplets(
    idxs: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    start_nodes_per_class: int,
    tuplet_size: int,
    num_classes: int,
    num_nodes: int,
    noise: float | int,
    depth_limit: int,
    alpha: float = 0.1,
    sparse: bool = False,
    sparse_x_prune_threshold: float = 0.001,
    seed=1337,
):
    class_partitions = partition_labels(idxs, labels[idxs], num_classes)
    rnd = np.random.default_rng(seed)
    for part in class_partitions:
        rnd.shuffle(part)
    start_nodes = np.concatenate(
        [part[: min(start_nodes_per_class, len(part))] for part in class_partitions],
        dtype=np.int32,
    )
    adj_t = SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes), trust_data=True
    )

    propagate = APPNPPropagation(
        K=depth_limit,
        alpha=alpha,
        add_self_loops=True,
        dropout=0.0,
        normalization="out-degree",
        sparse_x_prune_threshold=sparse_x_prune_threshold,
    ).to(edge_index.device)

    if sparse:
        identity = SparseTensor.eye(num_nodes, dtype=torch.float32, device=edge_index.device)  # type: ignore
    else:
        identity = torch.eye(num_nodes, dtype=torch.float32, device=edge_index.device)

    probs = (1 - noise) * propagate(identity, adj_t) + noise
    probs = probs[start_nodes, :][:, idxs]
    probs /= probs.sum(-1, keepdim=True).numpy()
    tuplets = idxs[
        (
            np.expand_dims(probs.cumsum(-1), -1)
            >= rnd.uniform(size=(probs.shape[0], 1, tuplet_size))
        ).argmax(-2)
    ]

    return tuplets


def get_tuplet_dists(tuplets: np.ndarray, labels: np.ndarray, c: int) -> np.ndarray:
    tuplet_labels = labels[tuplets]
    counts = bincount_last_axis(tuplet_labels, c - 1)
    tuplet_dists = counts / tuplets.shape[-1]
    return tuplet_dists


def set_tuplets(
    dataset: InMemoryDatasetProvider,
    train_tuplets: np.ndarray | None = None,
    val_tuplets: np.ndarray | None = None,
    test_tuplets: np.ndarray | None = None,
    train_tuplet_dists: np.ndarray | None = None,
    val_tuplet_dists: np.ndarray | None = None,
    test_tuplet_dists: np.ndarray | None = None,
) -> InMemoryDatasetProvider:
    data: Any = dataset.data_list[0]
    data.train_tuplets = torch.tensor(train_tuplets)
    data.val_tuplets = torch.tensor(val_tuplets)
    data.test_tuplets = torch.tensor(test_tuplets)
    data.train_tuplet_dists = torch.tensor(train_tuplet_dists)
    data.val_tuplet_dists = torch.tensor(val_tuplet_dists)
    data.test_tuplet_dists = torch.tensor(test_tuplet_dists)
    return dataset


def get_tuplets(dataset: InMemoryDatasetProvider) -> dict[str, np.ndarray | None]:
    data: Any = dataset.data_list[0]
    return dict(
        train_tuplets=getattr(data, "train_tuplets", None),
        val_tuplets=getattr(data, "val_tuplets", None),
        test_tuplets=getattr(data, "test_tuplets", None),
    )


def add_skewed_test_splits_to_data(
    dataset: InMemoryDatasetProvider,
    strategy: str,
    repeats: int,
    tuplet_size: int | None,
    depth_limit: int | None,
    noise: float | None,
    seed: int = 1337,
    artifact_name: str | None = None,
) -> InMemoryDatasetProvider:
    data: Any = dataset.data_list[0]

    def compute_splits(*args):
        test_indices = data.test_mask.nonzero(as_tuple=False).flatten()
        test_size: int = test_indices.size(0)

        if strategy == "zipf":
            test_splits, test_split_dists = generate_zipf_tuplets(
                test_indices.numpy(),
                data.y[test_indices].numpy(),
                tuplet_size=tuplet_size or test_size,
                num_classes=dataset.num_classes,
                tuplet_count=repeats,
                seed=seed,
            )
        elif strategy == "neighbor":
            assert tuplet_size is not None, "BFS strategy requires tuplet size."
            assert depth_limit is not None, "BFS strategy requires depth limit."
            assert noise is not None, "BFS strategy requires noise."
            num_nodes = data.test_mask.size(0)
            test_splits = generate_neighborhood_tuplets(
                test_indices,
                data.edge_index,
                data.y,
                start_nodes_per_class=repeats,
                tuplet_size=tuplet_size,
                num_classes=dataset.num_classes,
                num_nodes=num_nodes,
                depth_limit=depth_limit,
                noise=noise,
                seed=seed,
            )
            test_split_dists = get_tuplet_dists(
                test_splits, data.y.numpy(), dataset.num_classes
            )
        elif strategy == "full_neighbor":
            assert tuplet_size is not None, "BFS strategy requires tuplet size."
            assert depth_limit is not None, "BFS strategy requires depth limit."
            assert noise is not None, "BFS strategy requires noise."
            num_nodes = data.test_mask.size(0)
            test_splits = generate_full_neighborhood_tuplets(
                test_indices,
                dataset.get_apsp(),
                data.y,
                start_nodes_per_class=repeats,
                tuplet_size=tuplet_size,
                num_classes=dataset.num_classes,
                num_nodes=num_nodes,
                depth_limit=depth_limit,
                noise=noise,
                seed=seed,
            )
            test_split_dists = get_tuplet_dists(
                test_splits, data.y.numpy(), dataset.num_classes
            )
        elif strategy == "ppr":
            assert tuplet_size is not None, "Random walk strategy requires tuplet size."
            assert depth_limit is not None, "Random walk strategy requires depth limit."
            assert noise is not None, "Random walk strategy requires noise."
            num_nodes = data.test_mask.size(0)
            test_splits = generate_ppr_tuplets(
                test_indices,
                data.edge_index,
                data.y,
                start_nodes_per_class=repeats,
                tuplet_size=tuplet_size,
                num_classes=dataset.num_classes,
                num_nodes=num_nodes,
                depth_limit=depth_limit,
                noise=noise,
                seed=seed,
            )
            test_split_dists = get_tuplet_dists(
                test_splits.numpy(), data.y.numpy(), dataset.num_classes
            )
        else:
            raise ValueError(f"Invalid test split generation strategy: {strategy}")

        return dict(
            test_splits=torch.tensor(test_splits),
            test_split_dists=torch.tensor(test_split_dists),
        )

    if artifact_name is None:
        splits = compute_splits()
    else:
        splits = dataset.get_artifact(artifact_name, compute_splits)

    test_splits = splits["test_splits"]
    test_split_dists = splits["test_split_dists"]

    data.skewed_test_splits = test_splits
    data.skewed_test_split_dists = test_split_dists

    return dataset


def add_tuplets_to_data(
    dataset: InMemoryDatasetProvider,
    train_tuplet_size: int | float,
    val_tuplet_size: int | float,
    test_tuplet_size: int | float,
    train_oversampling_factor: float = 1.0,
    val_oversampling_factor: float = 1.0,
    test_oversampling_factor: float = 1.0,
    train_strategy: str = "random",
    seed=1337,
) -> InMemoryDatasetProvider:
    data: Any = dataset.data_list[0]

    num_classes = dataset.num_classes
    train_indices = data.train_mask.nonzero(as_tuple=False).flatten()
    val_indices = data.val_mask.nonzero(as_tuple=False).flatten()
    test_indices = data.test_mask.nonzero(as_tuple=False).flatten()
    train_size: int = train_indices.size(0)
    val_size: int = val_indices.size(0)
    test_size: int = test_indices.size(0)

    train_tuplet_size = (
        train_tuplet_size
        if train_tuplet_size >= 1 and isinstance(train_tuplet_size, int)
        else int(train_size * train_tuplet_size)
    )
    y_np = data.y.numpy()
    if train_strategy == "zipf":
        train_tuplets, train_tuplet_dists = generate_zipf_tuplets(
            train_indices.numpy(),
            data.y[train_indices].numpy(),
            train_tuplet_size,
            num_classes,
            train_oversampling_factor,
            seed,
        )
    elif train_strategy == "random":
        if train_tuplet_size > train_size:
            train_tuplet_size = train_size
        train_tuplets = generate_random_tuplets(
            train_indices.numpy(),
            train_tuplet_size,
            train_oversampling_factor,
            seed,
        )
        train_tuplet_dists = get_tuplet_dists(train_tuplets, y_np, num_classes)
    else:
        raise ValueError(f"Invalid tuplet generation strategy: {train_strategy}")

    val_tuplet_size = (
        val_tuplet_size
        if val_tuplet_size > 1 and isinstance(val_tuplet_size, int)
        else int(val_size * val_tuplet_size)
    )
    val_tuplets = generate_random_tuplets(
        val_indices.numpy(),
        val_tuplet_size,
        val_oversampling_factor,
        seed + 1,
    )
    val_tuplet_dists = get_tuplet_dists(val_tuplets, y_np, num_classes)
    test_tuplet_size = (
        test_tuplet_size
        if test_tuplet_size > 1 and isinstance(test_tuplet_size, int)
        else int(test_size * test_tuplet_size)
    )

    if hasattr(data, "skewed_test_splits"):
        skewed_test_splits = data.skewed_test_splits.numpy()
        test_tuplets_selector = generate_random_tuplets(
            np.arange(skewed_test_splits.shape[-1]),
            test_tuplet_size,
            test_oversampling_factor,
            seed + 2,
        )
        test_tuplets = skewed_test_splits[:, test_tuplets_selector]

    else:
        test_tuplets = generate_random_tuplets(
            test_indices.numpy(),
            test_tuplet_size,
            test_oversampling_factor,
            seed + 2,
        )
    test_tuplet_dists = get_tuplet_dists(test_tuplets, y_np, num_classes)

    return set_tuplets(
        dataset,
        train_tuplets,
        val_tuplets,
        test_tuplets,
        train_tuplet_dists,
        val_tuplet_dists,
        test_tuplet_dists,
    )
