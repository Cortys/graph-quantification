from typing import Optional, Tuple
from networkx import edge_expansion
from numpy import isin

from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv import MessagePassing
from .utils import mat_norm


class APPNPPropagation(MessagePassing):
    """APPNP-like propagation (approximate personalized page-rank)
    code taken from the torch_geometric repository on GitHub (https://github.com/rusty1s/pytorch_geometric)
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        K: int,
        alpha: float,
        dropout: float = 0.0,
        cached: bool = False,
        add_self_loops: bool = True,
        normalization: str | None = "sym",
        sparse_x_prune_threshold: float | None = None,
        stochastic_x=True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.K = K
        self.alpha = alpha
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        assert normalization in (
            "sym",
            "rw",
            "in-degree",
            "out-degree",
            "in-degree-sym",
            "sym-var",
            "sum",
            None,
        )
        self.normalization = normalization
        self.sparse_x_prune_threshold = sparse_x_prune_threshold
        self.stochastic_x = stochastic_x

        self._cached_edge_index = None
        self._cached_adj_t = None

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(
        self, x: Tensor | SparseTensor, edge_index: Adj, edge_weight: OptTensor = None
    ) -> Tensor | SparseTensor:
        """"""
        if self.normalization is not None:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    x_dtype = x.dtype() if isinstance(x, SparseTensor) else x.dtype
                    edge_index, edge_weight = mat_norm(
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        improved=False,
                        add_self_loops=self.add_self_loops,
                        dtype=x_dtype,
                        normalization=self.normalization,
                    )  # type: ignore

                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)

                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    x_dtype = x.dtype() if isinstance(x, SparseTensor) else x.dtype
                    edge_index = mat_norm(
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        improved=False,
                        add_self_loops=self.add_self_loops,
                        dtype=x_dtype,
                        normalization=self.normalization,
                    )  # type: ignore

                    if self.cached:
                        self._cached_adj_t = edge_index  # type: ignore
                else:
                    edge_index = cache  # type: ignore

        if isinstance(x, SparseTensor):
            h = x.set_value(x.storage.value() * self.alpha, "coo")  # type: ignore
        else:
            h = self.alpha * x

        for _ in range(self.K):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    assert edge_weight is not None
                    edge_weight = F.dropout(edge_weight, p=self.dropout)
                elif isinstance(edge_index, SparseTensor):
                    value = edge_index.storage.value()  # type: ignore
                    assert value is not None
                    value = F.dropout(value, p=self.dropout)
                    edge_index = edge_index.set_value(value, layout="coo")  # type: ignore
                else:
                    raise TypeError(f"Invalid edge_index type {type(edge_index)}.")

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(
                edge_index, x=x, edge_weight=edge_weight, size=None
            )

            if isinstance(x, SparseTensor):
                x = x.set_value(x.storage.value() * (1 - self.alpha), "coo") + h  # type: ignore
                t = self.sparse_x_prune_threshold

                if t is not None:
                    x_vals: Tensor | None = x.storage.value()
                    if x_vals is not None:
                        x_sum = 1 if self.stochastic_x else x.sum(-1)  # type: ignore
                        x_mask = x_vals >= t
                        x = x.masked_select_nnz(x_mask, layout="coo")  # type: ignore
                        x_sum_masked = x.sum(-1)  # type: ignore
                        x_sum_diff = x_sum - x_sum_masked
                        x = x.set_diag(x.get_diag() + x_sum_diff)  # type: ignore

            else:
                x = x * (1 - self.alpha)
                x += h

        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)  # type: ignore

    def __repr__(self):
        return "{}(K={}, alpha={})".format(self.__class__.__name__, self.K, self.alpha)
