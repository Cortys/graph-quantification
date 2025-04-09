from typing import Tuple
import torch
import torch.nn.functional as F
import torch_geometric.nn as tnn
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj
from gq.layers import LinearSequentialLayer
from gq.nn.loss import categorical_entropy_reg
from gq.utils import Prediction, ModelConfiguration
from .model import Model


class EgoNet(Model):
    """Classify according to the majority of the ego network around each node."""

    def __init__(self, params: ModelConfiguration):
        super().__init__(params)

        self.propagation = tnn.APPNP(
            K=params.K,
            alpha=params.alpha_teleport,
            add_self_loops=params.add_self_loops,
            normalize=False,
        )

    def forward(self, data: Data) -> Prediction:
        x = self.forward_impl(data)

        soft = x
        log_soft = torch.log(soft + 1e-8)
        max_soft, hard = soft.max(dim=-1)
        neg_entropy = categorical_entropy_reg(soft, 1, reduction="none")

        # ---------------------------------------------------------------------------------
        pred = Prediction(
            # prediction and intermediary scores
            soft=soft,
            log_soft=log_soft,
            hard=hard,
            # confidence of prediction
            prediction_confidence_aleatoric=max_soft,
            prediction_confidence_epistemic=None,
            prediction_confidence_structure=None,
            # confidence of sample
            sample_confidence_total=max_soft,
            sample_confidence_total_entropy=neg_entropy,
            sample_confidence_aleatoric=max_soft,
            sample_confidence_aleatoric_entropy=neg_entropy,
            sample_confidence_epistemic=None,
            sample_confidence_epistemic_entropy=None,
            sample_confidence_epistemic_entropy_diff=None,
            sample_confidence_features=None,
            sample_confidence_structure=None,
        )
        # ---------------------------------------------------------------------------------

        return pred

    def forward_impl(self, data: Data) -> torch.Tensor:
        if data.edge_index is not None:
            edge_index = data.edge_index
        else:
            edge_index = data.adj_t

        h = F.one_hot(data.y, num_classes=self.params.num_classes).float()  # type: ignore

        class_hist = h.sum(dim=0)
        class_hist = class_hist

        if hasattr(data, "train_mask"):
            h[~data.train_mask] = 0

        x = self.propagation(h, edge_index)

        if hasattr(data, "train_mask"):
            x[data.train_mask] = h[data.train_mask]

        x[x.sum(dim=-1) == 0] = class_hist

        x = x / x.sum(dim=-1, keepdim=True)

        return x

    def expects_training(self) -> bool:
        return False

    def loss(self, prediction: Prediction, data: Data) -> dict[str, torch.Tensor]:
        raise NotImplementedError
