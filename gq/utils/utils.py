import inspect
from typing import Callable, Union, Any, Tuple
import attr
import pandas as pd
import funcy as fy
import random
import collections.abc
import torch
from torch import Tensor
from torch_geometric.data import Data
import numpy as np
from .prediction import Prediction


def map_tensor(tensor: Tensor, mapping: dict):
    """map elements of a tensor according to a specified mapping

    Args:
        tensor (Tensor): input tensor
        mapping (dict): dictionary specifying the mapping

    Returns:
        Tensor: mapped tensor
    """

    tensor = tensor.clone()

    if tensor.dim() == 1:
        for i in range(tensor.size(0)):
            tensor[i] = mapping[tensor[i].item()]

    else:
        for i in range(tensor.size(0)):
            tensor[i, :] = map_tensor(tensor[i, :], mapping)

    return tensor


def __apply(v: Tensor, m: Tensor) -> Tensor:
    """internal function to apply a mask to a tensor or value"""

    if v.dim() == 0:
        return v

    if v.size(0) == m.size(0) or m.dtype != torch.bool:
        # apply boolean masks only if the first dimensions match
        # index masks are always applied (at the risk of out-of-bounds errors)
        return v[m]

    return v


def strip_prefix(string: str, prefix: str) -> str:
    """strips prefix from a string

    Args:
        string (str): input string
        prefix (str): prefix to strip

    Returns:
        str: stripped string
    """

    if string.startswith(prefix):
        return string[len(prefix) :]

    return string


def _apply_mask(
    y_hat: Union[dict, Tensor, Prediction], mask: Tensor
) -> Union[dict, Tensor, Prediction]:
    """applies a mask to a representation of a model's predictions

    Args:
        y_hat (Union[dict, Tensor, Prediction]): model's predictions
        mask (Tensor): mask, e.g. mask for a validation split

    Raises:
        AssertionError: raised if predictions are of an unsupported type

    Returns:
        Union[dict, Tensor, Prediction]: returns predictions selected by mask
    """

    if isinstance(y_hat, dict):
        _y_hat = {k: __apply(v, mask) for k, v in y_hat.items()}

    elif isinstance(y_hat, torch.Tensor):
        _y_hat = __apply(y_hat, mask)

    elif isinstance(y_hat, Prediction):
        y_hat_dict_full: dict[str, Any] = y_hat.to_dict()
        y_hat_dict_masked = {
            k: __apply(v, mask)
            for k, v in y_hat_dict_full.items()
            if not k.startswith("tuplets_")
        }
        y_hat_dict_rest = {
            k: v for k, v in y_hat_dict_full.items() if k.startswith("tuplets_")
        }
        _y_hat = Prediction(**y_hat_dict_masked, **y_hat_dict_rest)

    else:
        raise AssertionError

    return _y_hat


def apply_mask(
    data: Data,
    y_hat: Union[dict, Tensor, Prediction],
    split: str,
    return_target: bool = True,
) -> Union[
    Union[dict, Tensor, Prediction], Tuple[Union[dict, Tensor, Prediction], Tensor]
]:
    """applies a specified split/mask to model's predictions

    Args:
        data (Data): data representation
        y_hat (Union[dict, Tensor, Prediction]): model's predictions
        split (str): specified split
        return_target (bool, optional): whether or whether not to return ground-truth labels of desired split in addition to masked predictions. Defaults to True.

    Raises:
        NotImplementedError: raised if passed split is not supported

    Returns:
        Union[Union[dict, Tensor, Prediction], Tuple[Union[dict, Tensor, Prediction], Tensor]]: predictions (and ground-truth labels) after applying mask
    """

    target = None

    if split == "train":
        # If there is an active learning mask, use it instead of the entire training mask:
        if hasattr(data, "al_mask"):
            mask = data.al_mask
        else:
            mask = data.train_mask

    elif split == "val":
        mask = data.val_mask

    elif split == "test":
        if hasattr(data, "skewed_test_splits"):
            mask = data.skewed_test_splits  # index-based mask
            target = data.skewed_test_split_dists  # skewed target distributions
        else:
            mask = data.test_mask  # boolean mask

    elif split == "ood":
        mask = data.ood_mask

    elif split == "id":
        mask = data.id_mask

    elif split == "ood_val":
        mask = data.ood_val_mask

    elif split == "ood_test":
        mask = data.ood_test_mask

    elif split == "ood_train":
        # not intended as mask
        # for not breaking pipeline: empty mask
        mask = torch.zeros_like(data.y, dtype=torch.bool)  # type: ignore

    elif split == "id_val":
        mask = data.id_val_mask

    elif split == "id_test":
        mask = data.id_test_mask

    elif split == "id_train":
        # not intended as mask
        # for not breaking pipeline: empty mask
        mask = torch.zeros_like(data.y, dtype=torch.bool)  # type: ignore

    else:
        raise NotImplementedError(f"split {split} is not implemented!")

    _y_hat = _apply_mask(y_hat, mask)

    if isinstance(_y_hat, Prediction):
        _y_hat = attr.evolve(_y_hat, selected_split=split)

    if return_target:
        if target is None:
            target = data.y[mask]  # type: ignore
        return _y_hat, target
    return _y_hat


def to_one_hot(targets: Tensor, num_classes: int) -> Tensor:
    """maps hard-coded ground-truth targets to one-hot representation of those

    Args:
        targets (Tensor): ground-truth labels
        num_classes (int): number of classes

    Returns:
        Tensor: one-hot encoding
    """

    if len(targets.shape) == 1:
        targets = targets.unsqueeze(dim=-1)

    soft_output = torch.zeros((targets.size(0), num_classes), device=targets.device)
    soft_output.scatter_(1, targets, 1)

    return soft_output


def recursive_update(d: dict, u: collections.abc.Mapping):
    """recursively update a dictionary d with might contain nested sub-dictionarieswith values from u"""

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d[k], v)
        else:
            d[k] = v

    return d


def recursive_delete(d: dict, k: dict) -> dict:
    """delete a key k from a dict d which might be contained in nested sub-dictionaries"""

    for key, v in d.items():
        if key == k:
            del d[k]
            return d

        if isinstance(v, collections.abc.Mapping):
            d[key] = recursive_delete(d.get(key, {}), k)

    return d


def recursive_clean(d: dict) -> Union[dict, None]:
    """recursively clean a dictionary d which might contain nested sub-dictionaries, i.e. remove None-entries"""

    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = recursive_clean(v)
        if v is not None:
            new_dict[k] = v
    return new_dict or None


def recursive_get(d: dict, k: Any) -> Any:
    """recursively get a value specified by a key k from a dictionary d which might contain nested sub-dictionaries"""

    for key, v in d.items():
        if key == k:
            return v

        if isinstance(v, collections.abc.Mapping):
            _v = recursive_get(d.get(key, {}), k)
            if _v is not None:
                return _v

    return None


def set_seed(seed: int) -> None:
    """set seeds for controlled randomness"""

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def results_dict_to_df(
    results: dict, per_init_var=False, num_inits: int | None = None
) -> pd.DataFrame:
    metrics = [m[5:] for m in results.keys() if m.startswith("test_")]
    metrics_names = [(m[:-4] + "_var" if m.endswith("_val") else m) for m in metrics]
    result_values = {"val": [], "test": []}

    for s in ("val", "test"):
        for m in metrics:
            key = f"{s}_{m}"
            if key in results:
                val = results[key]
                if isinstance(val, list) and m.endswith("_val"):
                    val = np.array(val)
                    if per_init_var:
                        assert isinstance(num_inits, int)
                        shape_rest = val.shape[1:]
                        val = val.reshape((-1, num_inits) + shape_rest)
                        val = np.var(val, axis=1, ddof=1)
                        val = np.mean(val, axis=0)
                    else:
                        val = np.var(val, axis=0, ddof=1)
                if isinstance(val, np.ndarray):
                    val = list(val)
                result_values[s].append(val)
            else:
                result_values[s].append(None)

    if "al_train_sizes" in results:
        al_train_sizes = results["al_train_sizes"]
        result_values["val"].append(al_train_sizes)
        result_values["test"].append(al_train_sizes)
        metrics_names.append("al_train_sizes")

    return pd.DataFrame(data=result_values, index=metrics_names)


def bincount_last_axis(a, maxbin=None, counts_dtype=None):
    """
    Like np.bincount, but works for ND arrays.
    The bincount is performed along the last axis of the input.

    Args:
        a:
            ndarray, with unsigned integer contents
        maxbin:
            The highest bin value to return.
            Determines the length of the last axis in the results.
            If the input array contains higher values than this,
            those values will not be counted in the results.
    Returns:
        ndarray

        The dtype of the output will be the minimum unsigned type that is
        guaranteed to accommodate the counts, based on the length of the input.

        If the input shape is:

            (S0, S1, S2, ..., S(N-1), SN),

        then the result shape will be:

            (S0, S1, S2, ..., S(N-1), maxbin+1)

    Author: https://github.com/stuarteberg
    """
    a = np.asarray(a)
    assert maxbin is None or maxbin >= -1
    assert np.issubdtype(a.dtype, np.integer)
    if np.issubdtype(a.dtype, np.signedinteger):
        assert a.min() >= 0, "Can't operate on negative values"

    maxval = a.max()
    if maxbin is None:
        num_bins = maxval + 1
    elif maxval <= maxbin:
        num_bins = maxbin + 1
    else:
        # Leave one extra bin for all values above maxbin,
        # and force all high input values into that bin,
        # which we will discard before returning.
        num_bins = maxbin + 2
        a = np.clip(a, 0, maxbin + 1)

    # Flat counts
    counts_dtype = (
        np.min_scalar_type(a.shape[-1]) if counts_dtype is None else counts_dtype
    )
    counts = np.zeros((a.size // a.shape[-1]) * num_bins, dtype=counts_dtype)

    # Calculate flat indexes into the 'counts' array
    index_dtype = np.min_scalar_type(a.size)
    i = np.arange(a.size, dtype=index_dtype) // a.shape[-1]
    i *= num_bins
    i += a.reshape(-1).astype(index_dtype)

    # Perform the bincounts
    np.add.at(counts, i, 1)

    # Reshape back to ND
    counts = counts.reshape((*a.shape[:-1], num_bins))

    if maxbin is None or maxval <= maxbin:
        return counts

    # Discard the extra bin we used for above-max values.
    return counts[..., :-1]


class InvalidatableCache:
    def __init__(self):
        self.key = None
        self.cache: dict[Any, Any] = {}

    def set_key(self, key):
        if key is not self.key:
            self.cache = {}
            self.key = key

    def __getitem__(self, key):
        return self.cache[key]

    def __setitem__(self, key, value) -> None:
        self.cache[key] = value

    def __delitem__(self, key) -> None:
        del self.cache[key]

    def __contains__(self, key):
        return key in self.cache


def tolerant(
    f: Callable | None = None, only_named=True, ignore_varkwargs=False
) -> Callable:
    if f is None:
        return lambda f: tolerant(f, only_named, ignore_varkwargs)

    if hasattr(f, "__tolerant__"):
        return f

    spec = inspect.getfullargspec(f.__init__ if inspect.isclass(f) else f)
    f_varargs = spec.varargs is not None
    f_varkws = not ignore_varkwargs and spec.varkw is not None

    if (only_named or f_varargs) and f_varkws:
        return f

    f_args = spec.args
    f_kwonlyargs = spec.kwonlyargs

    @fy.wraps(f)
    def wrapper(*args, **kwargs):
        if not (only_named or f_varargs):
            args = args[: len(f_args)]
        if not f_varkws:
            kwargs = fy.project(kwargs, f_args[len(args) :] + f_kwonlyargs)

        return f(*args, **kwargs)  # type: ignore

    wrapper.__tolerant__ = True  # type: ignore

    return wrapper
