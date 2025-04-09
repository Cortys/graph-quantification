import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from gq.layers.appnp_propagation import APPNPPropagation
from gq.models.gat import GAT
from gq.models.gcn import GCN
from gq.nn.loss import JSD_loss
from gq.utils.prediction import Prediction
from gq.utils.utils import apply_mask
from ..model import Model


class MemoryQuantifier(Model):
    """MemoryQuantifier model
    This model is a simple baseline that memorizes the training class distribution
    """

    def __init__(self, params):
        super().__init__(params)

    def forward(self, data: Data, *_, **__) -> Prediction:
        _, y = apply_mask(data, dict(), "train")  # type: ignore
        y_dist: torch.Tensor = (
            torch.bincount(y.squeeze(), minlength=self.params.num_classes).float()
            / y.shape[0]
        ).reshape(1, -1)
        y_dist_log: torch.Tensor = torch.log(y_dist + 1e-8)
        return Prediction(
            tuplets_log_soft=dict(
                train=y_dist_log,
                val=y_dist_log,
                test=y_dist_log,
            ),
            tuplets_soft=dict(
                train=y_dist,
                val=y_dist,
                test=y_dist,
            ),
        )

    def expects_training(self) -> bool:
        return False

    def loss(self, prediction: Prediction, data: Data) -> dict[str, torch.Tensor]:
        raise NotImplementedError
