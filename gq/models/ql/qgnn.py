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
from ..model import Model


class QGNN(Model):
    """QGNN model"""

    default_normalization = "sym"

    def __init__(self, params):
        super().__init__(params)
        self.init_layers()

    def init_layers(self):
        assert isinstance(self.params.dim_hidden, int)

        # Input Encoder

        self.input_encoder = nn.Sequential(
            nn.Linear(self.params.dim_features, self.params.dim_hidden),
            nn.ReLU(),
            nn.Dropout(p=self.params.dropout_prob),
        )

        # Convolution

        params = self.params.clone()
        params.set_values(
            dim_features=self.params.dim_hidden,
            num_classes=self.params.dim_hidden,
        )
        if self.params.convolution_name == "gcn":
            self.gnn = GCN(params)
            self.gnn_fn = lambda data: self.gnn.forward_impl(data)
        elif self.params.convolution_name == "gat":
            self.gnn = GAT(params)
            self.gnn_fn = lambda data: self.gnn.forward_impl(data)
        elif self.params.convolution_name == "appnp":
            normalization = self.params.adj_normalization
            if normalization is None:
                normalization = self.default_normalization
            self.gnn = APPNPPropagation(
                K=self.params.K,
                alpha=self.params.alpha_teleport,
                add_self_loops=self.params.add_self_loops,
                cached=False,
                normalization=normalization,
            )
            self.gnn_fn = lambda data: self.gnn(
                data.x, data.edge_index if data.edge_index is not None else data.adj_t
            )
        else:
            raise ValueError("Invalid convolution_name.")

        # Final Aggregation

        agg_type = self.params.quantification_tuplet_aggregation
        conv_out_dim = (
            self.params.dim_hidden if agg_type == "nn" else self.params.num_classes
        )
        self.post_conv_transformation = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.params.dim_hidden, conv_out_dim),
        )

        if agg_type == "mean":
            self.aggregation_fn = lambda x, tuplets: self.tuplet_aggregation(
                self.post_conv_transformation(x), tuplets
            )
        elif agg_type == "nn":
            self.aggregation_nn = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.params.dim_hidden, self.params.dim_hidden),
                nn.ReLU(),
                nn.Linear(self.params.dim_hidden, self.params.num_classes),
            )

            def aggregation_fn(x, tuplets):
                x = self.post_conv_transformation(x)
                x = self.tuplet_aggregation(x, tuplets)
                x = self.aggregation_nn(x)

                return x

            self.aggregation_fn = aggregation_fn

    @staticmethod
    def tuplet_aggregation(x, tuplets):
        return x[tuplets].mean(dim=-2)

    def forward(self, data: Data, *_, **__) -> Prediction:
        tuplets = dict(
            train=getattr(data, "train_tuplets", None),
            val=getattr(data, "val_tuplets", None),
            test=getattr(data, "test_tuplets", None),
        )
        assert tuplets is not None
        h = self.input_encoder(data.x)
        new_data = data.clone()
        new_data["x"] = h
        h = self.gnn_fn(new_data)
        splits = ["train"] if self.training else ["train", "val", "test"]
        tuplets_logits_dict = dict()
        tuplets_log_soft_dict = dict()
        tuplets_soft_dict = dict()

        for split in splits:
            tuplets_logits = self.aggregation_fn(h, tuplets[split])
            tuplets_log_soft = F.log_softmax(tuplets_logits, dim=-1)
            tuplets_soft = torch.exp(tuplets_log_soft)
            tuplets_logits_dict[split] = tuplets_logits
            tuplets_log_soft_dict[split] = tuplets_log_soft
            tuplets_soft_dict[split] = tuplets_soft

        pred = Prediction(
            tuplets_logits=tuplets_logits_dict,
            tuplets_log_soft=tuplets_log_soft_dict,
            tuplets_soft=tuplets_soft_dict,
        )

        return pred

    def loss(self, prediction: Prediction, data: Data) -> dict[str, torch.Tensor]:
        train_tuplet_dists = getattr(data, "train_tuplet_dists", None)
        assert train_tuplet_dists is not None
        log_y_hat = prediction.tuplets_log_soft["train"]
        y_hat: torch.Tensor = prediction.tuplets_soft["train"]
        y = train_tuplet_dists

        return {
            "JSD": JSD_loss(
                y_hat, y, log_y_hat, reduction=self.params.loss_reduction or "sum"
            )
        }
