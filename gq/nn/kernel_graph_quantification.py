from functools import reduce
from typing import Any, Callable, Literal, Optional
import numpy as np
import quapy
import quapy.method.aggregative as qma
import quapy.functional as F
from sklearn.neighbors import KernelDensity
import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from gq.data.dataset_provider import InMemoryDatasetProvider
from gq.layers.appnp_propagation import APPNPPropagation
from gq.nn.metrics import Split, get_mask
from gq.utils.prediction import Prediction
from gq.utils.utils import bincount_last_axis

from . import quantification_metrics as qm
from .quantification_metrics import quantification_cache

QuantifierFn = Callable[[torch.Tensor, str], torch.Tensor]
KernelFn = Callable[[torch.Tensor, str], torch.Tensor]
ArtifactFn = Callable[..., Any]

QuantifierGeneratorFn = Callable[
    [Prediction, Data, torch.Tensor, ArtifactFn], QuantifierFn
]
KernelGeneratorFn = Callable[[Prediction, Data, torch.Tensor, ArtifactFn], KernelFn]
Kernel = Callable[..., KernelGeneratorFn]
KernelQuantifier = Callable[[KernelGeneratorFn], QuantifierGeneratorFn]

## Factories


def _quantification(
    pred: Prediction,
    dataset: InMemoryDatasetProvider,
    splits: list[Split],
    quantifier_gen: QuantifierGeneratorFn,
    err_fn: qm.ErrorFn = qm.quantification_errors,
    with_rounding: bool = True,
) -> dict[Split, qm.QuantificationResult]:
    assert "val" in splits
    assert "test" in splits
    global quantification_cache
    quantification_cache.set_key(pred)
    data: Data = dataset.data_list[0]
    result = {}
    num_classes: int = dataset.num_classes
    val_mask = data.val_mask
    quantifier = quantifier_gen(pred, data, val_mask, dataset.get_artifact)
    evaluated_splits: list[Split] = ["test"]

    for split in evaluated_splits:
        split_mask = get_mask(data, split)
        adjusted_quant = quantifier(split_mask, split)
        if split == "test" and split_mask.dim() > 1:
            true_quant = data.skewed_test_split_dists
            tuple_size = split_mask.shape[-1]
        else:
            y = data.y[split_mask]  # type: ignore
            tuple_size = y.shape[0]
            true_quant = (
                torch.bincount(y.squeeze(), minlength=num_classes).float() / tuple_size
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


def _kernel_quantification_multi(
    kernel_quantifier: KernelQuantifier,
    kernel: Kernel,
    kernel_params_fn: Callable[[bool], dict[str, dict[str, Any]]],
):
    def quantification_multi(
        pred: Prediction,
        dataset: InMemoryDatasetProvider,
        splits: list[Split],
        err_fn: qm.ErrorFn = qm.quantification_errors,
        evaluated_submetrics: Optional[set[str]] = None,
        reduced_quantification: bool = False,
    ) -> dict[Split, dict]:

        if pred.hard is None:
            return {}

        kernel_params = kernel_params_fn(reduced_quantification)
        if evaluated_submetrics is not None:
            filtered_kernel_params = {
                k: v
                for k, v in kernel_params.items()
                if not any(e.startswith(k) for e in evaluated_submetrics)
            }
            if len(filtered_kernel_params) < len(kernel_params):
                skipped = set(kernel_params.keys()) - set(filtered_kernel_params.keys())
                print(f"Skipping the following kernel params: {skipped}")
        else:
            filtered_kernel_params = kernel_params

        if len(filtered_kernel_params) == 0:
            return {}

        results = quapy.util.parallel(
            func=lambda params: _quantification(
                pred,
                dataset,
                splits,
                quantifier_gen=kernel_quantifier(kernel(**params)),
                err_fn=err_fn,
            ),
            args=filtered_kernel_params.values(),
            n_jobs=None,
            backend="threading",
        )

        evaluated_splits: list[Split] = ["test"]
        return {
            split: {
                param_str: result[split]  # type: ignore
                for param_str, result in zip(filtered_kernel_params.keys(), results)
            }
            for split in evaluated_splits
        }

    return quantification_multi


## Kernel functions


def _ppr_kernel(
    depth_limit: int = 10,
    alpha: float = 0.1,
    sparse: bool = False,
    sparse_x_prune_threshold: float = 0.001,
    weight_scale_method: Literal[None, "exp", "interpolate"] = None,
    weight_scale: float = 1.0,
) -> KernelGeneratorFn:
    global quantification_cache

    def kernel_generator(
        pred: Prediction, data: Data, val_mask: torch.Tensor, get_artifact: ArtifactFn
    ) -> KernelFn:
        N: int = data.num_nodes  # type: ignore

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
                    identity = torch.eye(
                        N, dtype=torch.float32, device=edge_index.device
                    )

                weights = propagate(identity, adj_t)
                return weights

            weights = get_artifact(
                f"ppr_weights_{depth_limit}_{alpha}_{sparse}", compute_weights
            )
            val_weights = weights[val_mask]

            if sparse:
                val_weights = val_weights.to_dense()

            quantification_cache["ppr_val_weights", depth_limit, alpha] = val_weights

        def kernel(split_mask: torch.Tensor, split: str) -> torch.Tensor:
            split_weights = val_weights[:, split_mask]
            split_weights = split_weights.transpose(0, -2)
            split_weights = split_weights.sum(-1)
            if weight_scale_method == "exp":
                split_weights = torch.exp(weight_scale * split_weights)
            elif weight_scale_method == "interpolate":
                split_weights = weight_scale * split_weights + (1 - weight_scale)
            return split_weights

        return kernel

    return kernel_generator


def _dist_kernel(
    mode: Literal["min", "avg"] = "min",
    depth_limit: int = 10,
    weight_scale_method: Literal["exp", "inv"] = "exp",
    weight_scale: float | Literal["avg_deg", "dist_part"] = 1.0,
) -> KernelGeneratorFn:
    global quantification_cache

    def kernel_generator(
        pred: Prediction, data: Data, val_mask: torch.Tensor, get_artifact: ArtifactFn
    ) -> KernelFn:
        y: torch.Tensor = data.y  # type: ignore

        apsp_matrix = torch.tensor(get_artifact("apsp"))

        def _compute_val_dists(test_idxs: torch.Tensor) -> torch.Tensor:
            distances = apsp_matrix[test_idxs][:, val_mask]
            if mode == "min":
                return distances.min(dim=0).values
            return distances

        def kernel(split_mask: torch.Tensor, split: str) -> torch.Tensor:
            if ("val_dists", split, mode, depth_limit) in quantification_cache:
                val_dists = quantification_cache["val_dists", split, mode, depth_limit]
            else:
                if split_mask.dtype == torch.bool:
                    node_idxs = torch.nonzero(split_mask).flatten()
                    val_dists = _compute_val_dists(node_idxs)
                else:
                    val_dists = []
                    for node_idxs in split_mask:
                        val_dists.append(_compute_val_dists(node_idxs))
                    val_dists = torch.stack(val_dists)
                quantification_cache["val_dists", split, mode, depth_limit] = val_dists

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

            if mode == "avg":
                val_weights = val_weights.mean(dim=-2)

            val_weights = val_weights.to(y.device)
            return val_weights

        return kernel

    return kernel_generator


def _ppr_kernel_params(reduced_quantification: bool = False):
    def param_to_string(depth_limit, alpha, weight_scale_method, weight_scale):
        if weight_scale_method is None:
            m = ""
        else:
            m = f"{weight_scale_method[:3]}_{int(weight_scale * 10)}_"
        m += f"{depth_limit}_{int(alpha * 100)}"
        return m

    int_vals = (
        [0.5]
        if reduced_quantification
        else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    params = [
        dict(
            depth_limit=depth_limit,
            alpha=alpha,
            weight_scale_method=method,
            weight_scale=weight_scale,
        )
        for depth_limit in [10]  # [5, 10]
        for alpha in [0.1]  # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        for method in [None, "interpolate"]  # [None, "exp", "interpolate"]
        for weight_scale in (int_vals if method == "interpolate" else [1.0])
    ]
    return {param_to_string(**param): param for param in params}


def _dist_kernel_params(reduced_quantification: bool = False):
    def param_to_string(mode, weight_scale_method, weight_scale):
        m = f"{mode}_{weight_scale_method}"
        if weight_scale_method != "inv":
            m += f"_{int(weight_scale * 10)}"
        else:
            m += f"_{weight_scale}"
        return m

    params = [
        dict(
            mode=mode,
            weight_scale_method=method,
            weight_scale=weight_scale,
        )
        for mode in ["avg"]  # ["min", "avg"]
        for method in ["inv", "exp"]
        for weight_scale in (
            ["dist_part", "avg_deg", 1] if method == "inv" else [0.2, 0.5]
        )
    ]
    return {param_to_string(**param): param for param in params}


## Quantification methods


def _kdey(kernel_generator: KernelGeneratorFn | None) -> QuantifierGeneratorFn:
    def quantifier_gen(
        pred: Prediction, data: Data, val_mask: torch.Tensor, get_artifact: ArtifactFn
    ) -> QuantifierFn:
        num_classes = pred.soft.size(-1)
        classes = np.arange(num_classes)
        y_val = data.y[val_mask]  # type: ignore
        true_labels = y_val.numpy().squeeze()
        y_hat_val = pred.soft[val_mask]
        pred_dists = y_hat_val.numpy()

        kdey = qma.KDEyML(True, random_state=1337)  # type: ignore
        classes_cond_dists = []
        uniform_dist = F.uniform_prevalence(num_classes)
        for cat in classes:
            class_cond_dists = pred_dists[true_labels == cat]
            if class_cond_dists.size == 0:
                class_cond_dists = [uniform_dist]
            classes_cond_dists.append(class_cond_dists)

        if kernel_generator is not None:
            kernel = kernel_generator(pred, data, val_mask, get_artifact)
        else:
            kernel = None
            kdey.mix_densities = [
                KernelDensity(bandwidth=kdey.bandwidth).fit(class_cond_dists)  # type: ignore
                for class_cond_dists in classes_cond_dists
            ]

        def quantifier(split_mask: torch.Tensor, split: str):
            y_hat = pred.soft[split_mask].numpy()

            batch_size = y_hat.shape[split_mask.ndim - 1]
            quapy.environ["SAMPLE_SIZE"] = batch_size

            if kernel is not None:
                split_weights = kernel(split_mask, split).numpy()
            else:
                split_weights = None

            def aggregate(post, weights=None):
                if weights is not None:
                    classes_cond_weights = []
                    filtered_classes_cond_dists = []
                    for cat in classes:
                        class_cond_weights = weights[true_labels == cat]
                        class_cond_dists = classes_cond_dists[cat]
                        weight_filter = class_cond_weights > 0
                        class_cond_dists = class_cond_dists[weight_filter]
                        class_cond_weights = class_cond_weights[weight_filter]
                        if class_cond_weights.size == 0:
                            class_cond_dists = [uniform_dist]
                            class_cond_weights = [1.0]
                        filtered_classes_cond_dists.append(class_cond_dists)
                        classes_cond_weights.append(class_cond_weights)

                    kdey.mix_densities = [
                        KernelDensity(bandwidth=kdey.bandwidth).fit(class_cond_dists, sample_weight=class_cond_weights)  # type: ignore
                        for class_cond_dists, class_cond_weights in zip(
                            filtered_classes_cond_dists, classes_cond_weights
                        )
                    ]
                return kdey.aggregate(post)

            if y_hat.ndim > 2:
                quant = quapy.util.parallel(
                    func=lambda kwargs: aggregate(**kwargs),
                    args=[
                        dict(
                            post=y_hat[i],
                            weights=None if split_weights is None else split_weights[i],
                        )
                        for i in range(y_hat.shape[0])
                    ],
                    n_jobs=None,
                    backend="threading",
                )  # type: ignore
            else:
                quant = aggregate(post=y_hat, weights=split_weights)

            return torch.Tensor(quant)

        return quantifier

    return quantifier_gen


## Combined quantification metrics

quantification_kdey_ppr_multi = _kernel_quantification_multi(
    kernel_quantifier=_kdey,
    kernel=_ppr_kernel,
    kernel_params_fn=_ppr_kernel_params,
)
quantification_kdey_dist_multi = _kernel_quantification_multi(
    kernel_quantifier=_kdey,
    kernel=_dist_kernel,
    kernel_params_fn=_dist_kernel_params,
)
