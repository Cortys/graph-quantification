from typing import Callable
import optax
import torch
from torch import Tensor
import numpy as np
import jax
import jax.numpy as jnp
import mdmm_jax as mdmm
import quapy
import quapy.method.aggregative as qma
import quapy.functional as qf
import quapy.error as qe
import scipy.stats as ss
from gq.nn.loss import JSD_loss
from gq.utils import Prediction
from gq.utils.utils import InvalidatableCache, bincount_last_axis

QuantificationResult = Tensor | dict[str, "QuantificationResult"]
ErrorFn = Callable[[Tensor, Tensor], QuantificationResult]

quantification_cache = InvalidatableCache()

## General Quantification Method and Metric Wrappers


def quantification_errors(
    pred_quant: Tensor | None = None, true_quant: Tensor | None = None
) -> QuantificationResult:
    if pred_quant is None:
        return dict(
            ae=torch.as_tensor(float("nan")),
            rae=torch.as_tensor(float("nan")),
            kld=torch.as_tensor(float("nan")),
            jsd=torch.as_tensor(float("nan")),
        )
    assert true_quant is not None
    return dict(
        ae=qe.mae(true_quant, pred_quant),
        rae=qe.mrae(true_quant, pred_quant),
        kld=qe.mkld(true_quant, pred_quant),
        jsd=JSD_loss(pred_quant, true_quant),
    )


def to_cpu(res) -> QuantificationResult:
    if isinstance(res, dict):
        return {k: to_cpu(v) for k, v in res.items()}

    return res.cpu().detach()


def round_quant(quant: Tensor, tuple_size: int) -> Tensor:
    rescaled_quant = (quant * tuple_size).numpy()
    rounded_quant = np.floor(rescaled_quant)
    missing = tuple_size - rounded_quant.sum(axis=-1)
    diff_quant = rescaled_quant - rounded_quant
    diff_ranks = ss.rankdata(-diff_quant, method="ordinal", axis=-1).astype(int) - 1
    if missing.ndim == 0:
        rounded_quant += (np.arange(tuple_size) < missing)[diff_ranks]
    else:
        rounded_quant += np.take_along_axis(
            np.arange(tuple_size) < missing[:, None], diff_ranks, axis=-1
        )
    rounded_quant = rounded_quant / tuple_size

    return torch.tensor(rounded_quant)


def _aggregative_quantification_metric(
    y_hat: Prediction,
    y: Tensor,
    quant_fn: Callable[[Prediction, Prediction | None, Tensor | None], Tensor],
    err_fn: ErrorFn,
    y_hat_val: Prediction | None = None,
    y_val: Tensor | None = None,
    with_rounding: bool = True,
) -> QuantificationResult:
    """calculates the quantization absolute error

    Args:
        y_hat (Prediction): model predictions
        y (Tensor): ground-truth labels

    Returns:
        Tensor: quantization absolute error
    """

    if (y_hat.soft is None) or (y_hat.hard is None):
        return (
            quantification_errors()
            if err_fn == quantification_errors
            else torch.as_tensor(float("nan"))
        )

    batch_size = y_hat.soft.size(-2)
    if batch_size == 0:
        return (
            quantification_errors()
            if err_fn == quantification_errors
            else torch.as_tensor(float("nan"))
        )

    num_classes = y_hat.soft.size(-1)
    quant = quant_fn(y_hat, y_hat_val, y_val)
    if y.shape[-1] == num_classes:
        true_quant = y
    else:
        true_quant = (
            torch.bincount(y.squeeze(), minlength=num_classes).float() / batch_size
        )

    quapy.environ["SAMPLE_SIZE"] = batch_size

    err = err_fn(quant, true_quant)

    if with_rounding:
        rounded_quant = round_quant(quant, batch_size)
        err = {"": err, "round": err_fn(rounded_quant, true_quant)}

    return to_cpu(err)


## Quantification Methods


def _classify_and_count(y_hat: Prediction, *args) -> Tensor:
    batch_size = y_hat.soft.size(-2)
    num_classes = y_hat.soft.size(-1)
    if y_hat.soft.dim() > 2:
        quant = (
            torch.tensor(
                bincount_last_axis(y_hat.hard, num_classes - 1, counts_dtype=np.int32)
            ).float()
            / batch_size
        )
    else:
        quant = (
            torch.bincount(y_hat.hard.squeeze(), minlength=num_classes).float()
            / batch_size
        )
    return quant


def _prob_classify_and_count(y_hat: Prediction, *args) -> Tensor:
    return y_hat.soft.mean(dim=-2)


def _solve_adjustment_jax(pteCondEstim, quant) -> np.ndarray:
    pteCondEstim = jnp.asarray(pteCondEstim)
    quant = jnp.asarray(quant)
    num_batches = quant.shape[0]
    num_classes = pteCondEstim.shape[1]

    def jax_loss(p, quant) -> jnp.ndarray:
        return jnp.sum(
            jax.vmap(lambda p, q: jnp.linalg.norm(pteCondEstim @ p - q), (0, 0))(
                p, quant
            )
        )

    def jax_constraint(p) -> jnp.ndarray:
        return jnp.sum(p, axis=-1) - 1

    params = jnp.full(fill_value=1 / num_classes, shape=(num_batches, num_classes))
    constraint = mdmm.eq(jax_constraint)
    mdmm_params = constraint.init(params)

    gradient_transform = optax.chain(optax.sgd(0.001), mdmm.optax_prepare_update())
    opt_state = gradient_transform.init((params, mdmm_params))

    def jax_constrained_loss(params, quant, mdmm_params):
        loss = jax_loss(params, quant)
        mdmm_loss, inf = constraint.loss(mdmm_params, params)
        return loss + mdmm_loss

    @jax.jit
    def opt_step(params, mdmm_params, opt_state):
        grads = jax.grad(jax_constrained_loss)(params, quant, mdmm_params)
        updates, opt_state = gradient_transform.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        params = jax.vmap(optax.projections.projection_simplex, 0)(params)
        return params, mdmm_params, opt_state

    for _ in range(5000):
        params, mdmm_params, opt_state = opt_step(params, mdmm_params, opt_state)

    return np.array(params)  # type: ignore


def _solve_adjustment(pteCondEstim, quant, backend="scipy") -> np.ndarray:
    def solve(pteCondEstim, quant):
        def loss(prev):
            return np.linalg.norm(pteCondEstim @ prev - quant)

        return qf.optim_minimize(loss, n_classes=pteCondEstim.shape[1])

    if quant.ndim > 1:
        if backend == "scipy":
            if pteCondEstim.ndim == 2:
                args = [(pteCondEstim, quant[i]) for i in range(quant.shape[0])]
            else:
                assert pteCondEstim.shape[0] == quant.shape[0]
                args = [(pteCondEstim[i], quant[i]) for i in range(quant.shape[0])]
            result = quapy.util.parallel(
                func=lambda args: solve(*args),
                args=args,
                n_jobs=None,
                backend="threading",
            )
        elif backend == "jax":
            result = _solve_adjustment_jax(pteCondEstim, quant)

        return result  # type: ignore
    return solve(pteCondEstim, quant)


def _adjusted_classify_and_count(
    y_hat: Prediction, y_hat_val: Prediction | None, y_val: Tensor | None
) -> Tensor:
    assert y_hat_val is not None
    assert y_val is not None
    num_classes = y_hat.soft.size(-1)
    classes = np.arange(num_classes)
    quant: np.ndarray = _classify_and_count(y_hat).numpy()
    pteCondEstim = qma.ACC.getPteCondEstim(
        classes, y_val.squeeze(), y_hat_val.hard.squeeze()
    )
    adjusted_quant = _solve_adjustment(pteCondEstim, quant)

    return torch.Tensor(qf.normalize_prevalence(adjusted_quant, method="clip"))


def _adjusted_prob_classify_and_count(
    y_hat: Prediction, y_hat_val: Prediction | None, y_val: Tensor | None
) -> Tensor:
    assert y_hat_val is not None
    assert y_val is not None
    num_classes = y_hat.soft.size(-1)
    classes = np.arange(num_classes)
    quant = _prob_classify_and_count(y_hat).numpy()
    pteCondEstim = qma.PACC.getPteCondEstim(classes, y_val.squeeze(), y_hat_val.soft)
    adjusted_quant = _solve_adjustment(pteCondEstim, quant)

    return torch.Tensor(qf.normalize_prevalence(adjusted_quant, method="clip"))


def _dmy_distributions(posteriors, nbins=8, cdf=False) -> np.ndarray:
    if posteriors.ndim == 2:
        if posteriors.shape[-1] == 2:
            posteriors = posteriors[:, np.newaxis, 0]
        posteriors = posteriors.T
    elif posteriors.ndim == 3:
        if posteriors.shape[-1] == 2:
            posteriors = posteriors[:, :, np.newaxis, 0]
        posteriors = posteriors.transpose(0, 2, 1)
    else:
        raise ValueError("posteriors must have 2 or 3 dimensions")

    posteriors_int = (posteriors * nbins).astype(int)
    posteriors_int = np.clip(posteriors_int, 0, nbins - 1)
    histograms = bincount_last_axis(posteriors_int, nbins - 1, counts_dtype=np.int32)
    distributions = histograms / histograms.sum(axis=-1, keepdims=True)
    if cdf:
        distributions = np.cumsum(distributions, axis=-1)
    return distributions


def _dmy_solve(val_distributions, test_distribution, divergence="HD"):
    divergence = qf.get_divergence(divergence)
    num_classes, channels, _ = val_distributions.shape
    val_distributions = val_distributions.reshape(num_classes, -1)

    def solve(test_distribution):
        def loss(prev):
            prev = np.expand_dims(prev, axis=0)
            mixture_distribution = (prev @ val_distributions).reshape(channels, -1)
            divs = [
                divergence(test_distribution[ch], mixture_distribution[ch])
                for ch in range(channels)
            ]
            return np.mean(divs)  # type: ignore

        return qf.argmin_prevalence(loss, num_classes, method="optim_minimize")

    if test_distribution.ndim > 2:
        num_target_dists, _ = test_distribution.shape[:2]

        return quapy.util.parallel(
            func=solve,
            args=[test_distribution[i] for i in range(num_target_dists)],
            n_jobs=None,
            backend="threading",
        )
    return solve(test_distribution)


def _dmy(
    y_hat: Prediction, y_hat_val: Prediction | None, y_val: Tensor | None
) -> Tensor:
    assert y_hat_val is not None
    assert y_val is not None
    num_classes = y_hat.soft.size(-1)

    pred_dists = y_hat_val.soft.numpy()
    true_labels = y_val.numpy().squeeze()
    val_distributions: np.ndarray = quapy.util.parallel(
        func=_dmy_distributions,
        args=[pred_dists[true_labels == c] for c in range(num_classes)],
        n_jobs=None,
        backend="threading",
    )  # type: ignore
    num_classes, channels, bins = val_distributions.shape
    test_distribution = _dmy_distributions(y_hat.soft.numpy())
    quant = _dmy_solve(val_distributions, test_distribution)

    return torch.Tensor(quant)


def _kdey(
    y_hat: Prediction, y_hat_val: Prediction | None, y_val: Tensor | None
) -> Tensor:
    assert y_hat_val is not None
    assert y_val is not None
    num_classes = y_hat.soft.size(-1)
    classes = np.arange(num_classes)

    pred_dists = y_hat_val.soft.numpy()
    true_labels = y_val.numpy().squeeze()
    kdey = qma.KDEyML(True, random_state=1337)  # type: ignore
    kdey.mix_densities = kdey.get_mixture_components(
        pred_dists, true_labels, classes, kdey.bandwidth
    )
    if y_hat.soft.ndim > 2:
        # quants = []
        # for i in range(y_hat.soft.size(0)):
        #     quants.append(kdey.aggregate(y_hat.soft[i].numpy()))
        # quant = np.stack(quants)

        quant = quapy.util.parallel(
            func=lambda post: kdey.aggregate(post),
            args=[y_hat.soft[i].numpy() for i in range(y_hat.soft.size(0))],
            n_jobs=None,
            backend="threading",
        )  # type: ignore
    else:
        quant = kdey.aggregate(y_hat.soft.numpy())

    return torch.Tensor(quant)


## Metrics: Quantification Method + Error Function

# Direct Count Methods


def quantification_cc(
    y_hat: Prediction,
    y: Tensor,
    err_fn: ErrorFn = quantification_errors,
):
    return _aggregative_quantification_metric(y_hat, y, _classify_and_count, err_fn)


def quantification_pcc(
    y_hat: Prediction,
    y: Tensor,
    err_fn: ErrorFn = quantification_errors,
):
    return _aggregative_quantification_metric(
        y_hat, y, _prob_classify_and_count, err_fn
    )


# Adjusted Count Methods


def quantification_acc(
    y_hat: Prediction,
    y: Tensor,
    y_hat_val: Prediction,
    y_val: Tensor,
    err_fn: ErrorFn = quantification_errors,
):
    return _aggregative_quantification_metric(
        y_hat, y, _adjusted_classify_and_count, err_fn, y_hat_val, y_val
    )


def quantification_pacc(
    y_hat: Prediction,
    y: Tensor,
    y_hat_val: Prediction,
    y_val: Tensor,
    err_fn: ErrorFn = quantification_errors,
):
    return _aggregative_quantification_metric(
        y_hat, y, _adjusted_prob_classify_and_count, err_fn, y_hat_val, y_val
    )


# Distribution Matching


def quantification_dmy(
    y_hat: Prediction,
    y: Tensor,
    y_hat_val: Prediction,
    y_val: Tensor,
    err_fn: ErrorFn = quantification_errors,
):
    return _aggregative_quantification_metric(y_hat, y, _dmy, err_fn, y_hat_val, y_val)


def quantification_kdey(
    y_hat: Prediction,
    y: Tensor,
    y_hat_val: Prediction,
    y_val: Tensor,
    err_fn: ErrorFn = quantification_errors,
):
    return _aggregative_quantification_metric(y_hat, y, _kdey, err_fn, y_hat_val, y_val)


# Direct Quantification


def quantification_tuplets(
    y_hat: Prediction,
    y: Tensor,
    err_fn: ErrorFn = quantification_errors,
    with_rounding: bool = True,
) -> QuantificationResult:
    split = y_hat.selected_split

    if (
        split is None
        or (y_hat.tuplets_soft is None)
        or y_hat.tuplets_soft[split] is None
    ):
        return (
            quantification_errors()
            if err_fn == quantification_errors
            else torch.as_tensor(float("nan"))
        )

    tuplets_soft = y_hat.tuplets_soft[split]
    batch_size = y.size(0)
    if batch_size == 0:
        return (
            quantification_errors()
            if err_fn == quantification_errors
            else torch.as_tensor(float("nan"))
        )

    num_classes = tuplets_soft.size(-1)
    quant = tuplets_soft.mean(dim=-2)
    if y.shape[-1] == num_classes:
        true_quant = y
    else:
        true_quant = (
            torch.bincount(y.squeeze(), minlength=num_classes).float() / batch_size
        )
    quapy.environ["SAMPLE_SIZE"] = batch_size

    quant = quant.expand(*true_quant.shape)

    err = err_fn(quant, true_quant)
    if with_rounding:
        rounded_quant = round_quant(quant, batch_size)
        err = {
            "": err,
            "round": err_fn(rounded_quant, true_quant),
        }

    return to_cpu(err)
