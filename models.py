import turtle
import torch.nn as nn
from torch.nn import Linear, ReLU
from torch_geometric.nn import GCNConv, InnerProductDecoder, Sequential, global_add_pool, NNConv, BatchNorm, global_mean_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_dense_batch
from torch_geometric.data import Data, Batch
import torch
import torch.nn.functional as F
from math import ceil

class GcnEncoderGraph(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 embedding_dim,
                 label_dim,
                 num_layers,
                 pred_hidden_dims=[],
                 bn=True,
                 concat=False,
                 dropout=0.0,
                 add_self=True,
                 args=None) -> None:
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        self.add_self = add_self
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1
        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim,
            hidden_dim,
            embedding_dim,
            num_layers,
            add_self,
            normalize=True,
            dropout=dropout,
        )
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if True:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        self.pred_model = self.build_pred_layers(self.pred_input_dim,
                                                 pred_hidden_dims,
                                                 label_dim,
                                                 num_aggs=self.num_aggs)

    def build_conv_layers(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        num_layers,
        add_self,
        normalize=True,
        dropout=0.0,
    ):
        conv_first = GCNConv(in_channels=input_dim,
                             out_channels=hidden_dim,
                             add_self_loops=add_self,
                             normalize=normalize,
                             bias=self.bias)

        conv_block = nn.ModuleList([
            GCNConv(in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    add_self_loops=add_self,
                    normalize=normalize,
                    bias=self.bias) for i in range(num_layers - 2)
        ])

        conv_last = GCNConv(in_channels=hidden_dim,
                            out_channels=embedding_dim,
                            add_self_loops=add_self,
                            normalize=normalize,
                            bias=self.bias)

        return conv_first, conv_block, conv_last

    def build_pred_layers(self,
                          pred_input_dim,
                          pred_hidden_dims,
                          label_dim,
                          num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_hidden_dims[-1], label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def forward(self, x, adj, batch=None, **kwargs):
        out_all = []
        x = self.conv_first(x, adj)
        x = self.act(x)
        out_all.append(x)
        for layer in self.conv_block:
            x = layer(x, adj)
            x = self.act(x)
            out_all.append(x)
            # batch norm if need
        x = self.conv_last(x, adj)
        out_all.append(x)
        out = torch.cat(out_all, dim=1)
        out = global_mean_pool(out, batch)
        ypred = self.pred_model(out)
        return ypred


if __name__ == "__main__":
    pass