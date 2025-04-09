from typing import Callable, Literal, Optional
import numpy as np
import quapy
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_sparse
from torch_sparse import SparseTensor
import torch_scatter
import jax
import jax.numpy as jnp

from gq.data.dataset_provider import InMemoryDatasetProvider
from gq.layers.appnp_propagation import APPNPPropagation
from gq.utils.prediction import Prediction
from gq.nn.metrics import Split, get_mask
from gq.utils.utils import bincount_last_axis

from . import quantification_metrics as qm
from .quantification_metrics import quantification_cache

Mode = Literal["acc", "pacc"]
EPS = 1e-6


def sparse_kronecker_product(
    *idx_tensors: torch.Tensor, max_indices: list[int]
) -> torch.Tensor:
    dims = len(idx_tensors)
    idxs = idx_tensors[-1].to(torch.int64, copy=True)
    factor = 1
    for i in range(1, dims):
        factor *= max_indices[-i]
        idxs += idx_tensors[-i - 1] * factor
    return idxs


def sparse_kronecker_product_sum(
    *idx_tensors: torch.Tensor,
    max_indices: list[int],
    values: torch.Tensor | None = None,
) -> torch.Tensor:
    dims = len(idx_tensors)
    assert len(max_indices) == dims
    idxs = sparse_kronecker_product(*idx_tensors, max_indices=max_indices)
    max_index_prod = int(np.prod(max_indices))
    if values is None:
        values = torch.tensor(1, dtype=torch.int64)
    if values.dim() > idxs.dim():
        idxs = idxs.expand_as(values)
    result = torch.zeros(idxs.shape[:-1] + (max_index_prod,), dtype=values.dtype)
    result.scatter_add_(-1, idxs, values.expand_as(idxs))
    return result


def kronecker_product(*tensors: torch.Tensor) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]

    tensor_count = len(tensors)
    eq_in = []
    out = "..."
    for i in range(tensor_count):
        letter = chr(97 + i)
        eq_in.append(f"...{letter}")
        out += letter
    equation = ",".join(eq_in) + "->" + out
    result = torch.einsum(equation, *tensors)
    return result.reshape(result.shape[:-tensor_count] + (-1,))


def hard_multi_cond_prob_estimate(
    y_trues: list[torch.Tensor],
    y_preds: list[torch.Tensor],
    num_classes: int,
    add_missing: bool = True,
    use_missing_from_first: bool = True,
    y_true_weights: torch.Tensor | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Computes a confusion matrix for multi-dimensional predictions.
    """
    confusion = sparse_kronecker_product_sum(
        *y_trues,
        *y_preds,
        max_indices=[num_classes] * (len(y_trues) + len(y_preds)),
        values=y_true_weights,
    )
    confusion = confusion.view(confusion.shape[:-1] + (num_classes ** len(y_trues), -1))
    if normalize or add_missing:
        class_counts = confusion.sum(dim=-1)
        if add_missing:
            assert (
                confusion.dim() == 2
            ), "Batched confusion matrix computation not supported."
            missing_classes = torch.nonzero(class_counts == 0)
            if len(missing_classes) > 0:
                if use_missing_from_first and len(y_trues) > 1:
                    missing_primary_classes = missing_classes // num_classes
                    decomposed_confusion = confusion.view(
                        (num_classes, num_classes ** (len(y_trues) - 1), -1)
                    ).sum(dim=-2)
                    confusion[missing_classes] = decomposed_confusion[
                        missing_primary_classes
                    ]
                    class_counts = confusion.sum(dim=-1)
                    missing_classes = torch.nonzero(class_counts == 0)

                if len(missing_classes) > 0:
                    class_counts[missing_classes] = 1
                    if len(y_trues) > len(y_preds):
                        missing_classes_nested_idx = missing_classes // (
                            num_classes ** (len(y_trues) - len(y_preds))
                        )
                    elif len(y_trues) < len(y_preds):
                        missing_classes_nested_idx = missing_classes.clone()
                        factor = 1
                        for _ in range(len(y_preds) - len(y_trues)):
                            factor *= num_classes
                            missing_classes_nested_idx += missing_classes * factor
                    else:
                        missing_classes_nested_idx = missing_classes
                    confusion[missing_classes, missing_classes_nested_idx] = 1.0
        normalized_confusion = confusion / class_counts.unsqueeze(-1)
    else:
        normalized_confusion = confusion
    return normalized_confusion


def soft_multi_cond_prob_estimate(
    y_trues: list[torch.Tensor],
    y_preds_soft: list[torch.Tensor],
    num_classes: int,
    inner_num_classes: int | None = None,
    add_missing: bool = True,
    use_missing_from_first: bool = True,
    y_true_weights: torch.Tensor | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    if inner_num_classes is None:
        inner_num_classes = num_classes
    y_pred_product = kronecker_product(*y_preds_soft)
    y_pred_product = y_pred_product.to(torch.float32)
    if len(y_trues) == 1:
        y_true = y_trues[0].to(torch.int64)
    else:
        y_true = sparse_kronecker_product(
            *y_trues,
            max_indices=[num_classes] + [inner_num_classes] * (len(y_trues) - 1),
        )

    if y_true_weights is not None:
        weighted_y_pred_product = y_pred_product * y_true_weights.unsqueeze(-1)
        confusion = torch.zeros(
            weighted_y_pred_product.shape[:-2]
            + (
                num_classes * (inner_num_classes ** (len(y_trues) - 1)),
                weighted_y_pred_product.shape[-1],
            ),
            dtype=torch.float32,
        )
        torch_scatter.scatter_add(
            weighted_y_pred_product, y_true, dim=-2, out=confusion
        )
    else:
        confusion = torch.zeros(
            (
                num_classes * (inner_num_classes ** (len(y_trues) - 1)),
                y_pred_product.shape[-1],
            ),
            dtype=torch.float32,
        )
        torch_scatter.scatter_add(y_pred_product, y_true, dim=-2, out=confusion)

    if normalize or add_missing:
        class_counts = confusion.sum(dim=-1)
        if add_missing:
            missing_classes = torch.nonzero(class_counts < EPS)
            if len(missing_classes) > 0:
                if use_missing_from_first and len(y_trues) > 1:
                    missing_primary_classes = missing_classes // num_classes
                    decomposed_confusion = confusion.view(
                        (num_classes, num_classes ** (len(y_trues) - 1), -1)
                    ).mean(dim=-2)
                    confusion[missing_classes] = decomposed_confusion[
                        missing_primary_classes
                    ]
                    class_counts = confusion.sum(dim=-1)
                    missing_classes = torch.nonzero(class_counts < EPS)

                if len(missing_classes) > 0:
                    if len(y_trues) > len(y_preds_soft):
                        missing_classes_nested_idx = missing_classes // (
                            num_classes ** (len(y_trues) - len(y_preds_soft))
                        )
                    elif len(y_trues) < len(y_preds_soft):
                        missing_classes_nested_idx = missing_classes.clone()
                        factor = 1
                        for _ in range(len(y_preds_soft) - len(y_trues)):
                            factor *= num_classes
                            missing_classes_nested_idx += missing_classes * factor
                    else:
                        missing_classes_nested_idx = missing_classes
                    confusion[missing_classes, missing_classes_nested_idx] = 1.0
        confusion /= class_counts.unsqueeze(-1)

    return confusion


def hard_multi_prob_estimate(*y_preds: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Computes the probability distribution of multi-dimensional predictions.
    """
    occurrences = sparse_kronecker_product_sum(
        *y_preds, max_indices=[num_classes] * len(y_preds)
    )
    total_count = occurrences.sum(dim=-1, keepdim=True)
    multi_prob = occurrences / total_count
    return multi_prob


def soft_multi_prob_estimate(*y_preds: torch.Tensor) -> torch.Tensor:
    joint_pred_dists = kronecker_product(*y_preds)
    return joint_pred_dists.mean(dim=-2)


def compute_val_neighborhood_class_counts(data: Data, num_classes: int):
    if hasattr(data, "y_neigh_mat"):
        y_neigh_mat_val = data.y_neigh_mat
    else:
        edge_index: torch.Tensor = data.edge_index  # type: ignore
        num_nodes: int = data.num_nodes  # type: ignore
        val_mask = data.val_mask
        test_mask_inverse = ~data.test_mask

        non_test_edge_index_mask = (
            test_mask_inverse[edge_index[0]] & test_mask_inverse[edge_index[1]]
        )
        non_test_edge_index = edge_index[:, non_test_edge_index_mask]

        y_mat = F.one_hot(data.y, num_classes)  # type: ignore
        y_neigh_mat_val = torch_sparse.spmm(
            non_test_edge_index,
            torch.tensor([1], dtype=y_mat.dtype),
            num_nodes,
            num_nodes,
            y_mat,
        )[val_mask]
        isolated_nodes = y_neigh_mat_val.sum(dim=-1) == 0
        y_neigh_mat_val[isolated_nodes] = y_mat[val_mask][isolated_nodes]
        data.y_neigh_mat_val = y_neigh_mat_val
    return y_neigh_mat_val


def quantification_maj_neighbor(
    pred: Prediction,
    dataset: InMemoryDatasetProvider,
    splits: list[Split],
    err_fn: qm.ErrorFn = qm.quantification_errors,
    mode: Mode = "acc",
    true_neighbors: bool = False,
    pred_neighbors: bool = True,
    with_rounding: bool = True,
) -> dict[Split, qm.QuantificationResult]:
    global quantification_cache
    assert "val" in splits
    assert "test" in splits
    quantification_cache.set_key(pred)

    data: Data = dataset.data_list[0]
    result = {}
    num_nodes: int = data.num_nodes  # type: ignore
    num_classes: int = dataset.num_classes
    edge_index: torch.Tensor = data.edge_index  # type: ignore

    pred_tensor: torch.Tensor
    if mode == "acc":
        pred_tensor = pred.hard  # type: ignore
        pred_mat = F.one_hot(pred_tensor, num_classes)
    else:
        pred_tensor = pred.soft
        pred_mat = pred_tensor

    val_mask = data.val_mask

    y_val = data.y[val_mask]  # type: ignore
    y_hat_val = pred_tensor[val_mask]
    confusion_y_trues = [y_val]
    confusion_y_preds = [y_hat_val]

    if true_neighbors:
        true_neigh_mat_val = compute_val_neighborhood_class_counts(data, num_classes)
        true_neigh_val = true_neigh_mat_val.argmax(dim=-1)
        confusion_y_trues.append(true_neigh_val)

    if pred_neighbors:
        if (mode, "pred_neigh") in quantification_cache:
            pred_neigh = quantification_cache[mode, "pred_neigh"]
        else:
            pred_mat_neigh = torch_sparse.spmm(
                edge_index,
                torch.tensor([1], dtype=pred_mat.dtype),
                num_nodes,
                num_nodes,
                pred_mat,
            )

            isolated_nodes = pred_mat_neigh.sum(dim=-1) == 0
            pred_mat_neigh[isolated_nodes] = pred_mat[isolated_nodes]
            if mode == "acc":
                pred_neigh = pred_mat_neigh.argmax(dim=-1)
            else:
                pred_mat_neigh_sum = pred_mat_neigh.sum(-1, keepdim=True)
                pred_neigh = pred_mat_neigh / pred_mat_neigh_sum
            quantification_cache[mode, "pred_neigh"] = pred_neigh

        y_hat_neigh_val = pred_neigh[val_mask]
        confusion_y_preds.append(y_hat_neigh_val)

    confusion_matrix_fn: Callable[
        [list[torch.Tensor], list[torch.Tensor], int], torch.Tensor
    ] = (
        hard_multi_cond_prob_estimate
        if mode == "acc"
        else soft_multi_cond_prob_estimate
    )
    confusion = confusion_matrix_fn(
        confusion_y_trues, confusion_y_preds, num_classes=num_classes
    ).T  # Shape: (num_classes^len(y_trues), num_classes^len(y_preds))

    evaluated_splits: list[Split] = ["test"]
    for split in evaluated_splits:
        split_mask = get_mask(data, split)
        y_hat = pred_tensor[split_mask]

        batch_size = y_hat.shape[split_mask.ndim - 1]
        quapy.environ["SAMPLE_SIZE"] = batch_size

        y_hats = [y_hat]

        if pred_neighbors:
            y_hat_neigh = pred_neigh[split_mask]  # type: ignore
            y_hats.append(y_hat_neigh)

        if (mode, pred_neighbors, split, "multi_dim_quant") in quantification_cache:
            multi_dim_quant = quantification_cache[
                mode, pred_neighbors, split, "multi_dim_quant"
            ]
        else:
            if mode == "acc":
                multi_dim_quant = hard_multi_prob_estimate(
                    *y_hats, num_classes=num_classes
                )
            else:
                multi_dim_quant = soft_multi_prob_estimate(*y_hats)
            quantification_cache[mode, pred_neighbors, split, "multi_dim_quant"] = (
                multi_dim_quant
            )

        adjusted_quant = torch.Tensor(qm._solve_adjustment(confusion, multi_dim_quant))

        if true_neighbors:
            adjusted_quant = adjusted_quant.reshape(
                adjusted_quant.shape[:-1] + (num_classes, num_classes)
            )
            adjusted_quant = adjusted_quant.sum(dim=-1)

        true_quant: torch.Tensor
        if split == "test" and split_mask.dim() > 1:
            true_quant = data.skewed_test_split_dists
            tuple_size = split_mask.shape[-1]
        else:
            y = data.y[split_mask]  # type: ignore
            tuple_size = y.shape[0]
            true_quant = (
                torch.bincount(y.squeeze(), minlength=num_classes).float() / y.shape[0]
            )

        split_results = err_fn(adjusted_quant, true_quant)
        if with_rounding:
            rounded_adjusted_quant = qm.round_quant(adjusted_quant, tuple_size)
            split_results = {
                "": split_results,
                "round": err_fn(rounded_adjusted_quant, true_quant),
            }
        result[split] = split_results
    return result


def quantification_maj_meighbor_multi(
    pred: Prediction,
    dataset: InMemoryDatasetProvider,
    splits: list[Split],
    err_fn: qm.ErrorFn = qm.quantification_errors,
):
    if pred.hard is None:
        return {}

    def param_to_string(mode, true_neighbors, pred_neighbors):
        if true_neighbors and pred_neighbors:
            res = "full"
        elif true_neighbors:
            res = "true"
        elif pred_neighbors:
            res = "pred"
        else:
            raise ValueError("Invalid combination of true_neighbors and pred_neighbors")
        return f"{res}_{mode}"

    params = [
        dict(mode=mode, true_neighbors=true_neighbors, pred_neighbors=pred_neighbors)
        for mode in ["acc", "pacc"]
        for true_neighbors in [False, True]
        for pred_neighbors in [False, True]
        if true_neighbors or pred_neighbors
    ]

    results = quapy.util.parallel(
        func=lambda params: quantification_maj_neighbor(
            pred, dataset, splits, err_fn, **params
        ),
        args=params,
        n_jobs=None,
        backend="threading",
    )

    return {
        split: {
            param_to_string(**param): result[split]  # type: ignore
            for param, result in zip(params, results)
        }
        for split in ["test"]
    }


def quantification_cluster_neighbor(
    pred: Prediction,
    dataset: InMemoryDatasetProvider,
    splits: list[Split],
    err_fn: qm.ErrorFn = qm.quantification_errors,
    mode: Mode = "acc",
    num_clusters: int = 2,
    true_neighbors: bool = True,
    pred_neighbors: bool = False,
) -> dict[Split, qm.QuantificationResult]:
    global quantification_cache
    assert "val" in splits
    assert "test" in splits
    quantification_cache.set_key(pred)

    data: Data = dataset.data_list[0]
    result = {}
    num_nodes: int = data.num_nodes  # type: ignore
    num_classes: int = dataset.num_classes
    edge_index: torch.Tensor = data.edge_index  # type: ignore

    if mode == "acc":
        pred_mat = F.one_hot(pred.hard, num_classes)
    else:
        pred_mat = pred.soft

    val_mask = data.val_mask

    y_val = data.y[val_mask]  # type: ignore
    y_hat_val = pred_mat[val_mask]
    confusion_y_trues = [y_val]
    confusion_y_preds = [y_hat_val.to(torch.float32)]

    if true_neighbors:
        if not hasattr(data, "y_neigh_clusters_val"):
            data.y_neigh_clusters_val = dict()

        if num_clusters in data.y_neigh_clusters_val:
            true_neigh_clusters_val = data.y_neigh_clusters_val[num_clusters]
        else:
            true_neigh_mat_val = compute_val_neighborhood_class_counts(
                data, num_classes
            )
            true_neigh_mat_val = true_neigh_mat_val / true_neigh_mat_val.sum(
                dim=-1, keepdim=True
            )
            true_neigh_clusters_val = torch.zeros(len(y_val), dtype=torch.int32)
            for i in range(num_classes):
                cluster_mask = y_val == i
                km = KMeans(n_clusters=num_clusters)
                true_neigh_clusters_val[cluster_mask] = torch.tensor(
                    km.fit_predict(true_neigh_mat_val[cluster_mask]), dtype=torch.int32
                )
            data.y_neigh_clusters_val[num_clusters] = true_neigh_clusters_val
        confusion_y_trues.append(true_neigh_clusters_val)

    cache_mode = "acc_cluster" if mode == "acc" else "pacc"

    if pred_neighbors:
        if (cache_mode, "pred_neigh") in quantification_cache:
            pred_neigh = quantification_cache[cache_mode, "pred_neigh"]
        else:
            pred_mat_neigh = torch_sparse.spmm(
                edge_index,
                torch.tensor([1], dtype=pred_mat.dtype),
                num_nodes,
                num_nodes,
                pred_mat,
            )

            isolated_nodes = pred_mat_neigh.sum(dim=-1) == 0
            pred_mat_neigh[isolated_nodes] = pred_mat[isolated_nodes]
            pred_mat_neigh_sum = pred_mat_neigh.sum(-1, keepdim=True)
            pred_neigh = pred_mat_neigh / pred_mat_neigh_sum
            quantification_cache[cache_mode, "pred_neigh"] = pred_neigh

        y_hat_neigh_val = pred_neigh[val_mask]
        confusion_y_preds.append(y_hat_neigh_val)

    confusion = soft_multi_cond_prob_estimate(
        confusion_y_trues,
        confusion_y_preds,
        num_classes=num_classes,
        inner_num_classes=num_clusters,
    ).T  # Shape: (num_classes^len(y_trues), num_classes^len(y_preds))

    evaluated_splits: list[Split] = ["test"]
    for split in evaluated_splits:
        split_mask = get_mask(data, split)
        y_hat = pred_mat[split_mask].to(torch.float32)

        batch_size = y_hat.shape[split_mask.ndim - 1]
        quapy.environ["SAMPLE_SIZE"] = batch_size

        y_hats = [y_hat]

        if pred_neighbors:
            y_hat_neigh = pred_neigh[split_mask]  # type: ignore
            y_hats.append(y_hat_neigh)

        if (
            cache_mode,
            pred_neighbors,
            split,
            "multi_dim_quant",
        ) in quantification_cache:
            multi_dim_quant = quantification_cache[
                cache_mode, pred_neighbors, split, "multi_dim_quant"
            ]
        else:
            multi_dim_quant = soft_multi_prob_estimate(*y_hats)
            quantification_cache[
                cache_mode, pred_neighbors, split, "multi_dim_quant"
            ] = multi_dim_quant

        adjusted_quant = torch.Tensor(qm._solve_adjustment(confusion, multi_dim_quant))

        if true_neighbors:
            adjusted_quant = adjusted_quant.reshape(
                adjusted_quant.shape[:-1] + (num_classes, num_clusters)
            )
            adjusted_quant = adjusted_quant.sum(dim=-1)

        true_quant: torch.Tensor
        if split == "test" and split_mask.dim() > 1:
            true_quant = data.skewed_test_split_dists
        else:
            y = data.y[split_mask]  # type: ignore
            true_quant = (
                torch.bincount(y.squeeze(), minlength=num_classes).float() / y.shape[0]
            )

        split_results = err_fn(adjusted_quant, true_quant)
        result[split] = split_results
    return result


def quantification_cluster_neighbor_multi(
    pred: Prediction,
    dataset: InMemoryDatasetProvider,
    splits: list[Split],
    err_fn: qm.ErrorFn = qm.quantification_errors,
):
    if pred.hard is None:
        return {}

    def param_to_string(mode, true_neighbors, pred_neighbors):
        if true_neighbors and pred_neighbors:
            res = "full"
        elif true_neighbors:
            res = "true"
        elif pred_neighbors:
            res = "pred"
        else:
            raise ValueError("Invalid combination of true_neighbors and pred_neighbors")
        return f"{res}_{mode}"

    params = [
        dict(mode=mode, true_neighbors=True, pred_neighbors=pred_neighbors)
        for mode in ["acc", "pacc"]
        for pred_neighbors in [False, True]
    ]

    results = quapy.util.parallel(
        func=lambda params: quantification_cluster_neighbor(
            pred, dataset, splits, err_fn, **params
        ),
        args=params,
        n_jobs=None,
        backend="threading",
    )

    return {
        split: {
            param_to_string(**param): result[split]  # type: ignore
            for param, result in zip(params, results)
        }
        for split in ["test"]
    }


def class_pairs_to_idxs(class_pairs: torch.Tensor, num_classes: int) -> torch.Tensor:
    offsets = torch.cumsum(torch.arange(num_classes), dim=0)
    return offsets[class_pairs[0]] + class_pairs[1]


def quantification_edge(
    pred: Prediction,
    dataset: InMemoryDatasetProvider,
    splits: list[Split],
    err_fn: qm.ErrorFn = qm.quantification_errors,
    mode: Mode = "acc",
) -> dict[Split, qm.QuantificationResult]:
    global quantification_cache
    assert "val" in splits
    assert "test" in splits
    quantification_cache.set_key(pred)

    data: Data = dataset.data_list[0]
    result = {}
    num_nodes: int = data.num_nodes  # type: ignore
    num_classes: int = dataset.num_classes
    edge_index: torch.Tensor = data.edge_index  # type: ignore

    if mode == "acc":
        pred_mat = F.one_hot(pred.hard, num_classes)
    else:
        pred_mat = pred.soft

    val_mask = data.val_mask
    val_idxs = torch.nonzero(val_mask).squeeze()

    directed_edge_mask = edge_index[0] < edge_index[1]
    directed_edge_index = edge_index[:, directed_edge_mask]
    val_edge_mask = val_mask[directed_edge_index[0]] & val_mask[directed_edge_index[1]]
    val_edge_index = directed_edge_index[:, val_edge_mask]
    # val_edge_index = torch.cat([val_idxs.expand(2, -1), val_edge_index], dim=1)

    val_edge_true = torch.stack([data.y[val_edge_index[0]], data.y[val_edge_index[1]]])  # type: ignore
    val_edge_true = val_edge_true.sort(dim=0).values
    val_edge_idx_true = class_pairs_to_idxs(val_edge_true, num_classes)

    val_edge_pred = kronecker_product(
        pred_mat[val_edge_index[0]], pred_mat[val_edge_index[1]]
    ).reshape((-1, num_classes, num_classes))
    val_edge_pred += torch.triu(torch.transpose(val_edge_pred, -1, -2), diagonal=1)
    class_pairs = torch.triu_indices(num_classes, num_classes, offset=0)
    val_edge_pred = val_edge_pred[:, class_pairs[0], class_pairs[1]]

    confusion = soft_multi_cond_prob_estimate(
        [val_edge_idx_true],
        [val_edge_pred],
        num_classes=class_pairs.shape[1],
        add_missing=False,
    )
    missing_true_pairs = confusion.sum(dim=-1) > EPS
    confusion = confusion[missing_true_pairs].T
    class_pair_lut = class_pairs[:, missing_true_pairs]

    evaluated_splits: list[Split] = ["test"]
    for split in evaluated_splits:
        split_mask = get_mask(data, split)

        if split_mask.dtype == torch.int64:
            split_mask_bin = torch.zeros(
                split_mask.shape[:-1] + (num_nodes,), dtype=torch.bool
            )
            split_mask_bin.scatter_(
                -1, split_mask, torch.tensor(1, dtype=torch.bool).expand_as(split_mask)
            )
            split_mask = split_mask_bin

        split_edge_mask = (
            split_mask[..., directed_edge_index[0]]
            & split_mask[..., directed_edge_index[1]]
        )
        adjusted_quants = []
        for node_mask, edge_mask in zip(split_mask, split_edge_mask):
            split_edge_index = directed_edge_index[:, edge_mask]
            if split_edge_index.shape[1] == 0:
                node_idxs = torch.nonzero(node_mask).reshape(-1)
                split_edge_index = node_idxs.expand(2, -1)
            # node_idxs = torch.nonzero(node_mask).reshape(-1)
            # split_edge_index = torch.cat(
            #     [node_idxs.expand(2, -1), split_edge_index], dim=1
            # )

            split_edge_pred = kronecker_product(
                pred_mat[split_edge_index[0]], pred_mat[split_edge_index[1]]
            ).to(torch.float32)
            split_edge_pred = split_edge_pred.reshape(
                split_edge_pred.shape[:-1] + (num_classes, num_classes)
            )
            split_edge_pred += torch.triu(
                torch.transpose(split_edge_pred, -1, -2), diagonal=1
            )
            split_edge_pred = split_edge_pred[..., class_pairs[0], class_pairs[1]]
            split_edge_pred = split_edge_pred.mean(dim=-2)
            adjusted_edge_pred = torch.Tensor(
                qm._solve_adjustment(confusion, split_edge_pred)
            )
            adjusted_quant = torch.zeros(num_classes, dtype=torch.float32)
            adjusted_quant.scatter_add_(
                dim=0, index=class_pair_lut[0], src=adjusted_edge_pred
            )
            adjusted_quant.scatter_add_(
                dim=0, index=class_pair_lut[1], src=adjusted_edge_pred
            )
            adjusted_quant /= 2

            adjusted_quants.append(adjusted_quant)

        adjusted_quants = torch.stack(adjusted_quants)
        true_quant: torch.Tensor = data.skewed_test_split_dists

        split_results = err_fn(adjusted_quants, true_quant)
        result[split] = split_results

    return result


def quantification_edge_multi(
    pred: Prediction,
    dataset: InMemoryDatasetProvider,
    splits: list[Split],
    err_fn: qm.ErrorFn = qm.quantification_errors,
):
    if pred.hard is None:
        return {}

    def param_to_string(mode):
        return mode

    params = [dict(mode=mode) for mode in ["acc", "pacc"]]

    results = quapy.util.parallel(
        func=lambda params: quantification_edge(
            pred, dataset, splits, err_fn, **params
        ),
        args=params,
        n_jobs=None,
        backend="threading",
    )

    return {
        split: {
            param_to_string(**param): result[split]  # type: ignore
            for param, result in zip(params, results)
        }
        for split in ["test"]
    }


def quantification_weighted_dist(
    pred: Prediction,
    dataset: InMemoryDatasetProvider,
    splits: list[Split],
    err_fn: qm.ErrorFn = qm.quantification_errors,
    mode: Mode = "acc",
    depth_limit: int = 10,
    dist_aggregation: Literal["min", "avg"] = "min",
    weight_scale_method: Literal["exp", "inv"] = "exp",
    weight_scale: float | Literal["avg_deg", "dist_part"] = 1.0,
    pred_neighbors: bool = False,
    with_rounding: bool = True,
) -> dict[Split, qm.QuantificationResult]:
    global quantification_cache
    assert "val" in splits
    assert "test" in splits
    quantification_cache.set_key(pred)

    data: Data = dataset.data_list[0]
    result = {}
    num_nodes: int = data.num_nodes  # type: ignore
    num_classes: int = dataset.num_classes
    edge_index: torch.Tensor = data.edge_index  # type: ignore

    pred_tensor: torch.Tensor
    if mode == "acc":
        pred_tensor = pred.hard  # type: ignore
        pred_mat = F.one_hot(pred_tensor, num_classes)
    else:
        pred_tensor = pred.soft
        pred_mat = pred_tensor

    val_mask = data.val_mask
    y_val = data.y[val_mask]  # type: ignore
    y_hat_val = pred_tensor[val_mask]  # type: ignore

    confusion_y_preds = [y_hat_val]

    if pred_neighbors:
        if (mode, "pred_neigh") in quantification_cache:
            pred_neigh = quantification_cache[mode, "pred_neigh"]
        else:
            pred_mat_neigh = torch_sparse.spmm(
                edge_index,
                torch.tensor([1], dtype=pred_mat.dtype),
                num_nodes,
                num_nodes,
                pred_mat,
            )

            isolated_nodes = pred_mat_neigh.sum(dim=-1) == 0
            pred_mat_neigh[isolated_nodes] = pred_mat[isolated_nodes]
            if mode == "acc":
                pred_neigh = pred_mat_neigh.argmax(dim=-1)
            else:
                pred_mat_neigh_sum = pred_mat_neigh.sum(-1, keepdim=True)
                pred_neigh = pred_mat_neigh / pred_mat_neigh_sum
            quantification_cache[mode, "pred_neigh"] = pred_neigh

        y_hat_neigh_val = pred_neigh[val_mask]
        confusion_y_preds.append(y_hat_neigh_val)

    apsp_matrix = torch.tensor(dataset.get_artifact("apsp"))

    def _compute_val_dists(test_idxs: torch.Tensor) -> torch.Tensor:
        distances = apsp_matrix[test_idxs][:, val_mask]
        if dist_aggregation == "min":
            return distances.min(dim=0).values
        return distances

    evaluated_splits: list[Split] = ["test"]
    for split in evaluated_splits:
        split_mask = get_mask(data, split)
        y_hat = pred_tensor[split_mask]

        y_hats = [y_hat]

        if pred_neighbors:
            y_hat_neigh = pred_neigh[split_mask]  # type: ignore
            y_hats.append(y_hat_neigh)

        batch_size = y_hat.shape[split_mask.ndim - 1]
        quapy.environ["SAMPLE_SIZE"] = batch_size

        if ("val_dists", split, dist_aggregation, depth_limit) in quantification_cache:
            val_dists = quantification_cache[
                "val_dists", split, dist_aggregation, depth_limit
            ]
        else:
            if split_mask.dtype == torch.bool:
                node_idxs = torch.nonzero(split_mask).flatten()
                val_dists = _compute_val_dists(node_idxs)
            else:
                val_dists = []
                for node_idxs in split_mask:
                    val_dists.append(_compute_val_dists(node_idxs))
                val_dists = torch.stack(val_dists)
            quantification_cache["val_dists", split, dist_aggregation, depth_limit] = (
                val_dists
            )

        if weight_scale_method == "exp":
            assert isinstance(weight_scale, float)
            val_weights = torch.where(
                val_dists > depth_limit, 0, torch.exp(-weight_scale * val_dists)
            )
        else:
            if weight_scale == "avg_deg":
                num_nodes: int = data.num_nodes  # type: ignore
                num_edges = data.num_edges
                poly_weight_scale = max(1.0, num_edges / num_nodes)
                val_weights = torch.where(
                    val_dists > depth_limit, 0, 1.0 / (val_dists**poly_weight_scale)
                )
            elif weight_scale == "dist_part":
                val_dists = val_dists.detach().cpu().numpy()
                dist_freqs = bincount_last_axis(
                    val_dists,
                    maxbin=depth_limit,
                    counts_dtype=np.int32,
                )
                dist_freqs[..., 0] = 0
                with np.errstate(divide="ignore"):
                    val_weights = torch.tensor(
                        np.take_along_axis(
                            np.nan_to_num(
                                1 / dist_freqs,
                                nan=0,
                                posinf=0,
                                neginf=0,
                                copy=False,
                            ),
                            np.where(val_dists > depth_limit, 0, val_dists),
                            axis=-1,
                        )
                    )
            else:
                poly_weight_scale = weight_scale
                val_weights = torch.where(
                    val_dists > depth_limit, 0, 1.0 / (val_dists**poly_weight_scale)
                )

        if dist_aggregation == "avg":
            val_weights = val_weights.mean(dim=-2)

        val_weights = val_weights.to(device=y_val.device, dtype=torch.float32)

        compute_confusion = (
            hard_multi_cond_prob_estimate
            if mode == "acc"
            else soft_multi_cond_prob_estimate
        )

        confusion = compute_confusion(
            [y_val],
            confusion_y_preds,
            num_classes=num_classes,
            add_missing=False,
            y_true_weights=val_weights,
        )
        confusion = confusion.transpose(-1, -2)

        if (mode, pred_neighbors, split, "multi_dim_quant") in quantification_cache:
            multi_dim_quant = quantification_cache[
                mode, pred_neighbors, split, "multi_dim_quant"
            ]
        else:
            if mode == "acc":
                multi_dim_quant = hard_multi_prob_estimate(
                    *y_hats, num_classes=num_classes
                )
            else:
                multi_dim_quant = soft_multi_prob_estimate(*y_hats)
            quantification_cache[mode, pred_neighbors, split, "multi_dim_quant"] = (
                multi_dim_quant
            )

        adjusted_quant = torch.Tensor(qm._solve_adjustment(confusion, multi_dim_quant))

        if split == "test" and split_mask.dim() > 1:
            true_quant = data.skewed_test_split_dists
            tuple_size = split_mask.shape[-1]
        else:
            y = data.y[split_mask]  # type: ignore
            tuple_size = y.shape[0]
            true_quant = (
                torch.bincount(y.squeeze(), minlength=num_classes).float() / y.shape[0]
            )

        split_results = err_fn(adjusted_quant, true_quant)
        if with_rounding:
            rounded_adjusted_quant = qm.round_quant(adjusted_quant, tuple_size)
            split_results = {
                "": split_results,
                "round": err_fn(rounded_adjusted_quant, true_quant),
            }
        result[split] = split_results

    return result


def quantification_weighted_dist_multi(
    pred: Prediction,
    dataset: InMemoryDatasetProvider,
    splits: list[Split],
    err_fn: qm.ErrorFn = qm.quantification_errors,
    evaluated_submetrics: Optional[set[str]] = None,
    reduced_quantification: bool = False,
    **extra_params,
):
    if pred.hard is None:
        return {}

    def param_to_string(mode, dist_aggregation, weight_scale_method, weight_scale):
        if dist_aggregation == "avg":
            m = "avg_"
        else:
            m = ""  # Min was the default before, so it is not included in the name
        m += f"{weight_scale_method}_"
        if weight_scale_method == "exp":
            # 0.5 was the default before, so it is not included in the name
            if weight_scale != 0.5:
                m += f"{int(weight_scale * 10)}_"
        elif weight_scale_method == "inv":
            if isinstance(weight_scale, float):
                weight_scale = int(weight_scale * 10)
            m += f"{weight_scale}_"
        m += f"{mode}"
        return m

    if reduced_quantification:
        params = [
            dict(
                mode=mode,
                dist_aggregation=dist_aggregation,
                weight_scale_method=method,
                weight_scale=weight_scale,
            )
            for mode in ["acc", "pacc"]
            for dist_aggregation in ["avg"]
            for method in ["exp", "inv"]
            for weight_scale in ([1.0] if method == "inv" else [1.0])
        ]
    else:
        params = [
            dict(
                mode=mode,
                dist_aggregation=dist_aggregation,
                weight_scale_method=method,
                weight_scale=weight_scale,
            )
            for mode in ["acc", "pacc"]
            for dist_aggregation in ["min", "avg"]
            for method in ["exp", "inv"]
            for weight_scale in (
                ["dist_part", "avg_deg", 0.5, 1.0, 2.0]
                if method == "inv"
                else [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                    1.5,
                    2.0,
                    2.5,
                    3.0,
                ]
            )
        ]
    params = {param_to_string(**param): param for param in params}

    if evaluated_submetrics is not None:
        filtered_params = {
            k: v
            for k, v in params.items()
            if not any(e.startswith(k) for e in evaluated_submetrics)
        }
        if len(filtered_params) < len(params):
            # skipped = set(params.keys()) - set(filtered_params.keys())
            # print(f"Skipping the following params: {skipped}")
            print(f"Evaluating: {set(filtered_params.keys())}")
    else:
        filtered_params = params

    results = quapy.util.parallel(
        func=lambda params: quantification_weighted_dist(
            pred, dataset, splits, err_fn, **params, **extra_params
        ),
        args=filtered_params.values(),
        n_jobs=None,
        backend="threading",
    )

    return {
        split: {
            param_str: result[split]  # type: ignore
            for param_str, result in zip(filtered_params.keys(), results)
        }
        for split in ["test"]
    }


def quantification_weighted_dist_neighbor_multi(
    pred: Prediction,
    dataset: InMemoryDatasetProvider,
    splits: list[Split],
    err_fn: qm.ErrorFn = qm.quantification_errors,
    evaluated_submetrics: Optional[set[str]] = None,
    reduced_quantification: bool = False,
):
    return quantification_weighted_dist_multi(
        pred=pred,
        dataset=dataset,
        splits=splits,
        err_fn=err_fn,
        evaluated_submetrics=evaluated_submetrics,
        reduced_quantification=reduced_quantification,
        pred_neighbors=True,
    )


def _solve_ppr_adjustment(
    y_val,
    y_hat_val,
    y_hat,
    weights,
):
    y_val = jnp.asarray(y_val)  # V x K-idx
    y_hat_val = jnp.asarray(y_hat_val)  # V x K
    y_hat = jnp.asarray(y_hat)  # B x T x K
    weights = jnp.asarray(weights)  # B x V x T
    assert y_val.shape[0] == y_hat_val.shape[0]
    assert y_hat.shape[0] == weights.shape[0]
    assert y_hat_val.shape[-1] == y_hat.shape[-1]
    assert y_hat_val.shape[0] == weights.shape[1]
    assert y_hat.shape[1] == weights.shape[2]

    num_classes = y_hat.shape[-1]

    @jax.jit
    def compute_confusion():
        pairwise_hat_diffs = jnp.expand_dims(y_hat_val, (0, 2)) - jnp.expand_dims(
            y_hat, 1
        )  # B x V x T x K

        def class_conditional_weighted_diff_sum(j):
            j_mask = jnp.broadcast_to(
                jnp.reshape(y_val != j, (1, -1, 1)), weights.shape
            )
            masked_weights = jnp.place(weights, j_mask, 0, inplace=False)
            normalized_weights = jnp.nan_to_num(
                masked_weights / (jnp.sum(masked_weights, axis=-2, keepdims=True))
            )
            normalized_weights = normalized_weights / jnp.sum(
                normalized_weights, axis=(-2, -1), keepdims=True
            )
            weighted_diffs = pairwise_hat_diffs * jnp.expand_dims(
                normalized_weights, -1
            )
            diff_sum = jnp.sum(weighted_diffs, axis=(-3, -2))
            diff_sum = jnp.where(
                jnp.isnan(jnp.sum(diff_sum, axis=-1, keepdims=True)),
                0,
                diff_sum,
            )
            return diff_sum

        cc_diff_sums = jax.vmap(
            class_conditional_weighted_diff_sum, in_axes=0, out_axes=-1
        )(jnp.arange(num_classes))

        return cc_diff_sums

    unadjusted_pred = jnp.mean(y_hat, axis=-2)
    confusion = compute_confusion() + jnp.expand_dims(unadjusted_pred, -1)

    result = qm._solve_adjustment(np.array(confusion), np.array(unadjusted_pred))

    return torch.tensor(result)


def quantification_ppr(
    pred: Prediction,
    dataset: InMemoryDatasetProvider,
    splits: list[Split],
    err_fn: qm.ErrorFn = qm.quantification_errors,
    mode: Mode = "acc",
    depth_limit: int = 10,
    sparse: bool = False,
    alpha: float = 0.1,
    sparse_x_prune_threshold: float = 0.001,
    weight_scale_method: Literal[None, "exp", "interpolate"] = None,
    weight_scale: float = 1.0,
    grouped_weights: bool = True,
    pred_neighbors: bool = False,
    with_rounding: bool = True,
) -> dict[Split, qm.QuantificationResult]:
    global quantification_cache
    assert "val" in splits
    assert "test" in splits
    assert (
        not pred_neighbors or grouped_weights
    ), "pred_neighbors is currently only implemented for grouped_weights"
    quantification_cache.set_key(pred)

    data: Data = dataset.data_list[0]
    result = {}
    N: int = data.num_nodes  # type: ignore
    num_classes: int = dataset.num_classes
    edge_index: torch.Tensor = data.edge_index  # type: ignore

    pred_tensor: torch.Tensor
    if mode == "acc":
        pred_tensor = pred.hard  # type: ignore
        pred_mat = F.one_hot(pred_tensor, num_classes)
        if not grouped_weights:
            pred_tensor = pred_mat
    else:
        pred_tensor = pred.soft
        pred_mat = pred_tensor

    val_mask = data.val_mask
    y_val = data.y[val_mask]  # type: ignore
    y_hat_val = pred_tensor[val_mask]  # type: ignore

    confusion_y_preds = [y_hat_val]

    if pred_neighbors:
        if (mode, "pred_neigh") in quantification_cache:
            pred_neigh = quantification_cache[mode, "pred_neigh"]
        else:
            pred_mat_neigh = torch_sparse.spmm(
                edge_index,
                torch.tensor([1], dtype=pred_mat.dtype),
                N,
                N,
                pred_mat,
            )

            isolated_nodes = pred_mat_neigh.sum(dim=-1) == 0
            pred_mat_neigh[isolated_nodes] = pred_mat[isolated_nodes]
            if mode == "acc":
                pred_neigh = pred_mat_neigh.argmax(dim=-1)
            else:
                pred_mat_neigh_sum = pred_mat_neigh.sum(-1, keepdim=True)
                pred_neigh = pred_mat_neigh / pred_mat_neigh_sum
            quantification_cache[mode, "pred_neigh"] = pred_neigh

        y_hat_neigh_val = pred_neigh[val_mask]
        confusion_y_preds.append(y_hat_neigh_val)

    if ("ppr_val_weights", depth_limit, alpha) in quantification_cache:
        val_weights = quantification_cache["ppr_val_weights", depth_limit, alpha]
    else:

        def compute_weights(*args):
            edge_index: torch.Tensor = data.edge_index  # type: ignore
            adj_t = SparseTensor.from_edge_index(
                edge_index, sparse_sizes=(N, N), trust_data=True
            )
            propagate = APPNPPropagation(
                K=depth_limit,
                alpha=alpha,
                add_self_loops=True,
                dropout=0.0,
                normalization="in-degree",
                sparse_x_prune_threshold=sparse_x_prune_threshold,
            ).to(edge_index.device)

            if sparse:
                identity = SparseTensor.eye(N, dtype=torch.float32, device=edge_index.device)  # type: ignore
            else:
                identity = torch.eye(N, dtype=torch.float32, device=edge_index.device)

            weights = propagate(identity, adj_t)
            return weights

        weights = dataset.get_artifact(
            f"ppr_weights_{depth_limit}_{alpha}_{sparse}", compute_weights
        )
        val_weights = weights[val_mask]

        if sparse:
            val_weights = val_weights.to_dense()

        quantification_cache["ppr_val_weights", depth_limit, alpha] = val_weights

    evaluated_splits: list[Split] = ["test"]
    for split in evaluated_splits:
        split_mask = get_mask(data, split)
        y_hat = pred_tensor[split_mask]

        batch_size = y_hat.shape[split_mask.ndim - 1]
        quapy.environ["SAMPLE_SIZE"] = batch_size

        split_weights = val_weights[:, split_mask]
        split_weights = split_weights.transpose(0, -2)

        if grouped_weights:
            y_hats = [y_hat]

            if pred_neighbors:
                y_hat_neigh = pred_neigh[split_mask]  # type: ignore
                y_hats.append(y_hat_neigh)

            if (mode, pred_neighbors, split, "multi_dim_quant") in quantification_cache:
                multi_dim_quant = quantification_cache[
                    mode, pred_neighbors, split, "multi_dim_quant"
                ]
            else:
                if mode == "acc":
                    multi_dim_quant = hard_multi_prob_estimate(
                        *y_hats, num_classes=num_classes
                    )
                else:
                    multi_dim_quant = soft_multi_prob_estimate(*y_hats)
                quantification_cache[mode, pred_neighbors, split, "multi_dim_quant"] = (
                    multi_dim_quant
                )

            split_weights = split_weights.sum(-1)
            if weight_scale_method == "exp":
                split_weights = torch.exp(weight_scale * split_weights)
            elif weight_scale_method == "interpolate":
                split_weights = weight_scale * split_weights + (1 - weight_scale)

            compute_confusion = (
                hard_multi_cond_prob_estimate
                if mode == "acc"
                else soft_multi_cond_prob_estimate
            )

            confusion = compute_confusion(
                [y_val],
                confusion_y_preds,
                num_classes=num_classes,
                add_missing=False,
                y_true_weights=split_weights,
            )
            confusion = confusion.transpose(-1, -2)
            adjusted_quant = torch.Tensor(
                qm._solve_adjustment(confusion, multi_dim_quant)
            )
        else:
            if weight_scale_method == "exp":
                split_weights = torch.exp(weight_scale * split_weights)
            elif weight_scale_method == "interpolate":
                split_weights = weight_scale * split_weights + (1 - weight_scale)

            # split_weights = torch.ones_like(split_weights)

            if split_mask.dim() == 1:
                y_hat = y_hat.unsqueeze(0)
                split_weights = split_weights.unsqueeze(0)
            adjusted_quant = _solve_ppr_adjustment(
                y_val, y_hat_val, y_hat, split_weights
            )
            if split_mask.dim() == 1:
                adjusted_quant = adjusted_quant[0]

        if split == "test" and split_mask.dim() > 1:
            true_quant = data.skewed_test_split_dists
            tuple_size = split_mask.shape[-1]
        else:
            y = data.y[split_mask]  # type: ignore
            tuple_size = y.shape[0]
            true_quant = (
                torch.bincount(y.squeeze(), minlength=num_classes).float() / y.shape[0]
            )

        split_results = err_fn(adjusted_quant, true_quant)
        if with_rounding:
            rounded_adjusted_quant = qm.round_quant(adjusted_quant, tuple_size)
            split_results = {
                "": split_results,
                "round": err_fn(rounded_adjusted_quant, true_quant),
            }
        result[split] = split_results

    return result


def quantification_ppr_multi(
    pred: Prediction,
    dataset: InMemoryDatasetProvider,
    splits: list[Split],
    err_fn: qm.ErrorFn = qm.quantification_errors,
    pred_neighbors: bool = False,
    evaluated_submetrics: Optional[set[str]] = None,
    reduced_quantification: bool = False,
):
    if pred.hard is None:
        return {}

    def param_to_string(mode, grouped_weights, weight_scale_method, weight_scale):
        g = "group" if grouped_weights else "pairs"
        if weight_scale_method is None:
            m = ""
        else:
            m = weight_scale_method[:3] + "_"

            if (weight_scale_method == "interpolate" and weight_scale != 0.5) or (
                weight_scale_method == "exp" and weight_scale != 1.0
            ):
                m += f"{int(weight_scale * 10)}_"

        return f"{g}_{m}{mode}"

    if reduced_quantification:
        params = [
            dict(
                mode=mode,
                grouped_weights=gw,
                weight_scale_method=method,
                weight_scale=weight_scale,
            )
            for gw in [True]  # [True, False]
            if not pred_neighbors
            or gw  # pred_neighbors only implemented for grouped_weights
            for mode in ["acc", "pacc"]
            for method in [None, "interpolate"]  # [None, "exp", "interpolate"]
            for weight_scale in ([0.5] if method == "interpolate" else [1.0])
        ]
    else:
        params = [
            dict(
                mode=mode,
                grouped_weights=gw,
                weight_scale_method=method,
                weight_scale=weight_scale,
            )
            for gw in [True]  # [True, False]
            if not pred_neighbors
            or gw  # pred_neighbors only implemented for grouped_weights
            for mode in ["acc", "pacc"]
            for method in [None, "interpolate"]  # [None, "exp", "interpolate"]
            for weight_scale in (
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                if method == "interpolate"
                else [1.0]
            )
        ]
    params = {param_to_string(**param): param for param in params}

    if evaluated_submetrics is not None:
        filtered_params = {
            k: v
            for k, v in params.items()
            if not any(e.startswith(k) for e in evaluated_submetrics)
        }
        if len(filtered_params) < len(params):
            skipped = set(params.keys()) - set(filtered_params.keys())
            print(f"Skipping the following params: {skipped}")
            # print(f"Evaluating: {set(filtered_params.keys())}")
    else:
        filtered_params = params

    results = quapy.util.parallel(
        func=lambda params: quantification_ppr(
            pred, dataset, splits, err_fn, pred_neighbors=pred_neighbors, **params
        ),
        args=filtered_params.values(),
        n_jobs=None,
        backend="threading",
    )

    return {
        split: {
            param: result[split]  # type: ignore
            for param, result in zip(filtered_params.keys(), results)
        }
        for split in ["test"]
    }


def quantification_ppr_neighbor_multi(
    pred: Prediction,
    dataset: InMemoryDatasetProvider,
    splits: list[Split],
    err_fn: qm.ErrorFn = qm.quantification_errors,
    evaluated_submetrics: Optional[set[str]] = None,
    reduced_quantification: bool = False,
):
    return quantification_ppr_multi(
        pred=pred,
        dataset=dataset,
        splits=splits,
        err_fn=err_fn,
        evaluated_submetrics=evaluated_submetrics,
        reduced_quantification=reduced_quantification,
        pred_neighbors=True,
    )
