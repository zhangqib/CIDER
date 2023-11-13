import torch.nn as nn
from torch.nn import Linear, ReLU
import torch_geometric
from torch_geometric.nn import (
    GCNConv,
    InnerProductDecoder,
    Sequential,
    global_add_pool,
    NNConv,
    BatchNorm,
    global_mean_pool,
)
# from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_dense_batch
from torch_geometric.data import Data, Batch
import torch
import torch.nn.functional as F
from math import ceil


class CIDER(nn.Module):
    def __init__(
        self,
        in_channels,
        task_model,
        hidden_channels1=32,
        hidden_channels2=64,
        hidden_channels3=10,
        decoder_act=torch.relu,
    ) -> None:
        super(CIDER, self).__init__()

        self.gcn_shared = Sequential(
            "x, edge_index",
            [
                (GCNConv(in_channels, hidden_channels1), "x, edge_index -> x"),
            ],
        )
        self.gcn_mu_causal = GCNConv(hidden_channels1, hidden_channels2)
        self.gcn_mu_non_causal = GCNConv(hidden_channels1, hidden_channels2)
        self.gcn_logvar_causal = GCNConv(hidden_channels1, hidden_channels2)
        self.gcn_logvar_non_causal = GCNConv(hidden_channels1, hidden_channels2)

        self.decoder_causal = InnerProductDecoderMLP(hidden_dims=None, act=decoder_act)
        self.decoder_non_causal = InnerProductDecoderMLP(hidden_dims=None, act=decoder_act)
        self.task_model = task_model
        self.relu = ReLU()
        self.hidden_channels2 = hidden_channels2

    def encode(self, x, edge_index, edge_attr=None):
        x = self.relu(self.gcn_shared(x, edge_index))
        mu_causal = self.gcn_mu_causal(x, edge_index)
        mu_non_causal = self.gcn_mu_non_causal(x, edge_index)
        logvar_causal = self.gcn_logvar_causal(x, edge_index)
        logvar_non_causal = self.gcn_logvar_non_causal(x, edge_index)
        return mu_causal, mu_non_causal, logvar_causal, logvar_non_causal

    def decode(self, z_causal, z_non_causal, edge_index):
        return self.decoder_causal(z_causal, edge_index), self.decoder_non_causal(
            z_non_causal, edge_index
        )

    def _sample_encode(
        self,
        x,
        edge_index,
        edge_attr=None,
        num_sample=5,
        batch=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Sample from the latent space and encode the sampled latent variables in Input Conditioned X, P(Z|X)
        Args:
            x (torch.Tensor): Node feature of input graph.
            edge_index (torch.nn.Module): Edge index of input graph(adjacency list, pairs of nodes).
            edge_weight (torch.Tensor or None, optional): Edge feature of input graph.
            num_sample (int, optional): Number of couterfactual samples to take from the latent space, default is 5.
            batch (torch.Tensor or None, optional): Optional batch information.
        """

        ## mu_causal.shape: [#nodes, #hiddeen_channels2]
        ## mu_non_causal.shape: [#nodes, #hiddeen_channels2]

        mu_causal, mu_non_causal, logvar_causal, logvar_non_causal = self.encode(
            x, edge_index, edge_attr
        )

        sampled_z_causal = self.reparameterize(mu_causal, logvar_causal, device)
        sampled_z_non_causal = self.reparameterize(
            mu_non_causal.repeat(num_sample, 1),
            logvar_non_causal.repeat(num_sample, 1),
            device,
        )

        ## return sampled_z_causal.shape: [#nodes, #hiddeen_channels2]
        ##        sampled_z_non_causal.shape: [#nodes*#samples, #hiddeen_channels2]
        return sampled_z_causal, sampled_z_non_causal

    def forward(
        self,
        x,
        edge_index,
        edge_attr=None,
        batch=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Performs a forward pass through the model for a causal graph prediction task.

        Args:
            x (torch.Tensor): Node feature of input graph.
            edge_index (torch.nn.Module): Edge index of input graph(adjacency list, pairs of nodes).
            edge_attr (torch.Tensor or None, optional): Edge feature of input graph.
            batch (torch.Tensor or None, optional): Optional batch information.
            device (torch.device, optional): Device to use for computation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]: Tuple of edge weights for causal and non-causal edges, and the mean and logvar of the latent variables for causal and non-causal edges.
        """
        mu_causal, mu_non_causal, logvar_causal, logvar_non_causal = self.encode(
            x, edge_index, edge_attr
        )
        z_causal = self.reparameterize(mu_causal, logvar_causal, device)

        z_non_causal = self.reparameterize(mu_non_causal, logvar_non_causal, device)

        edge_weight_causal, edge_weight_non_causal = self.decode(
            z_causal, z_non_causal, edge_index
        )
        return (
            edge_weight_causal,
            edge_weight_non_causal,
            mu_causal,
            mu_non_causal,
            logvar_causal,
            logvar_non_causal,
        )

    def CF_forward(
        self,
        data: torch_geometric.data.Data,
        causal_criterion,
        num_sample=5,
        sparsity=0.8,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Performs a forward pass through the model for counterfactual samples

        Args:
            data (torch_geometric.data.Data): Input graph data.
            causal_criterion (torch.nn.Module): Criterion used to calculate the causal loss.
            num_sample (int, optional): Number of counterfactual samples.
            sparsity (float, optional): Sparsity level for enforcing edge sparsity.
            device (torch.device, optional): Device to use for computation.

        Returns:
            Tuple[torch.Tensor, float]: Loss_causal and accuracy (correct/num_sample).
        """

        # Encode the input data and sample causal and non-causal latent variables
        ## x.shape: [#node, #feature]
        ## edge_index.shape: [2, #edge]
        ## y.shape: [#node, #class]
        ## sampled_z_causal.shape: [#node, #hidden_channels2]
        ## sampled_z_non_causal.shape: [#node*#num_sample, #hidden_channels2]

        x = data.x
        edge_index = data.edge_index
        # y = data.y
        edge_attr = data.edge_attr
        batch = data.batch

        sampled_z_causal, sampled_z_non_causal = self._sample_encode(
            x,
            edge_index,
            num_sample=num_sample,
            edge_attr=edge_attr,
            batch=batch,
            device=device,
        )

        # Decode the latent variables to obtain sampled causal adjacency list
        ## edge_weight_causal.shape: [#edge]
        edge_weight_causal = self.decoder_causal(sampled_z_causal, edge_index)

        # merge counterfactual samples into one batch data
        data_list = [data for i in range(num_sample)]
        data_batch = (Batch.from_data_list(data_list)).to(device)


        # TODO: ERROR: the `sampled_z_non_causal` always 1. 
        # Decode the latent variables to obtain sampled non-causal adjacency list
        ## edge_weight_non_causal.shape: [#edge*#num_sample]
        edge_weight_non_causal = self.decoder_non_causal(
            sampled_z_non_causal, data_batch.edge_index
        )

        # compute uniform loss for non-causal graph
        edge_index_non_causal = data_batch.edge_index.T[edge_weight_non_causal >= 0].T
        sampled_y_non_cuasal = self.task_model(
            data_batch.x, edge_index_non_causal, batch=data_batch.batch
        )
        uniform_target = (
            torch.ones_like(sampled_y_non_cuasal) / self.task_model.label_dim
        ).to(device)
        loss_uniform = F.kl_div(
            F.softmax(sampled_y_non_cuasal), uniform_target, reduction="batchmean"
        )


        # TODO: There are some error, the threshold is not correct, it should be computed by edge_weight_causal only
        # repeat causal edge weights for each counterfactual sample and add then to edge_weight
        # edge_weight = edge_weight_causal.repeat(num_sample) + edge_weight_non_causal
        edge_weight = edge_weight_causal.repeat(num_sample)

        # Select top-k edges based on the weight threshold to enforce sparsity
        topk = min(
            ceil(edge_weight_causal.shape[0] * sparsity),
            edge_weight_causal.shape[0] - 1,
        )

        # reshape the edge_weight to [#num_sample, #edge] to select top-k edges for each sample
        edge_weight_reshape = edge_weight.reshape(num_sample, -1)

        # sort the edge weights in descending order in first dim and select the top-k edges for each sample(every row represents a sample)
        threshold = (
            edge_weight_reshape.sort(descending=True, dim=1)
            .values.topk(topk)
            .values[:, -1]
        )

        # expand the threshold to the same shape as edge_weight_reshape
        ## threshold.shape: [#num_sample, #edge]
        threshold = threshold.unsqueeze(1).expand_as(edge_weight_reshape)

        # reshape the threshold to verctor and calculate the edge mask
        ## edge_mask.shape: [#num_sample*#edge]
        # warning: the > is require to avoid the all edge_weight are 0
        edge_mask = (edge_weight_reshape > threshold).reshape(-1) | (edge_weight_non_causal>0.5)

        # select the top-k edges for each sample
        data_batch.edge_index = data_batch.edge_index.T[edge_mask].T

        # Pass the sampled input and  adjacency list to the task model
        sampled_y = self.task_model(
            data_batch.x, data_batch.edge_index, batch=data_batch.batch
        )

        # Calculate the causal loss using the criterion and repeat the target labels
        loss_causal = causal_criterion(sampled_y, data_batch.y)

        # Calculate the accuracy by comparing the predicted and target labels
        correct = float(sampled_y.argmax(dim=1).eq(data_batch.y).sum().item())

        return loss_uniform, loss_causal, correct / data_batch.num_graphs

    @torch.no_grad()
    def get_explainations(
        self,
        x,
        edge_index,
        edge_attr=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Returns the explainations for the input data
        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Input edge indices.
            edge_attr (torch.Tensor, optional): Input edge features.
            device (torch.device, optional): Device to use for computation
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the explainations of different sparsity.
        """
        self.eval()
        sparsities = [round(i / (i + 1), 2) for i in range(1, 10)][::-1]
        true_sparsities = [round(0.1 * i, 2) for i in range(9, 0, -1)]
        explainations = {}
        for sparsity, true_sparsity in zip(sparsities, true_sparsities):
            # mu_causal, _, logvar_causal, _ = self.encode(
            # x, edge_index, edge_attr)
            mu_causal, mu_non_causal, logvar_causal, logvar_non_causal = self.encode(
                x, edge_index, edge_attr
            )
            z_causal = self.reparameterize(mu_causal, logvar_causal, device)
            z_non_causal = self.reparameterize(mu_non_causal, logvar_non_causal, device)
            edge_weight_causal = self.decoder_causal(z_causal, edge_index)
            # edge_weight_non_causal = self.decoder_non_causal(z_non_causal, edge_index)
            topk = max(ceil(edge_index.shape[1] * sparsity), 1)
            threshold = (
                edge_weight_causal.sort(descending=True).values.topk(topk).values[-1]
            )
            # print(str(true_sparsity), ' causal ', edge_weight_causal)
            # print(str(true_sparsity), ' non ausal ', edge_weight_non_causal)
            noise = (torch.randn(edge_weight_causal.shape[0]//2)*1e-4).repeat_interleave(2)
            noise = noise.to(device)
            edge_index = edge_index.T[edge_weight_causal+noise > threshold].T
            explainations[str(true_sparsity)] = Data(x=x, edge_index=edge_index)

        return explainations

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor, device
    ) -> torch.Tensor:
        """
        return the sampling from the latent Gaussian distribution with reparaterization trick
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) 2*log(Standard deviation) of the latent Gaussian
        :return : (Tensor) Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)
        return eps 
        std + mu


# class InnerProductDecoderMLP(nn.Module):
#     """Decoder for using inner product for prediction."""

#     def __init__(
#         self, input_dim, hidden_dim1, hidden_dim2, dropout=0.1, act=torch.sigmoid
#     ):
#         super(InnerProductDecoderMLP, self).__init__()

#         # Fully connected layers
#         self.fc = nn.Linear(input_dim, hidden_dim1)
#         self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)

#         self.dropout = dropout
#         self.act = act

#         # Initialize the parameters
#         self._reset_parameters()

#     def _reset_parameters(self):
#         """
#         Reset model parameters using Xavier initialization.
#         """
#         torch.nn.init.xavier_uniform_(self.fc.weight)
#         torch.nn.init.zeros_(self.fc.bias)
#         torch.nn.init.xavier_uniform_(self.fc2.weight)
#         torch.nn.init.zeros_(self.fc2.bias)

#     def forward_all(self, z):
#         """
#         Compute the forward pass for the entire graph.

#         Args:
#             z (torch.Tensor): The latent space Z.

#         Returns:
#             torch.Tensor: The adjacency matrix of the graph.
#         """
#         z = self._forward_fc(z)
#         adj = self.act(torch.matmul(z, z.t()))
#         return adj

#     def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
#         """
#         Compute the forward pass for the given node-pairs.

#         Args:
#             z (torch.Tensor): The latent space Z.
#             edge_index (torch.Tensor): Index tensor representing node-pairs in the graph.

#         Returns:
#             torch.Tensor: The predicted values for the node-pairs.
#         """
#         z = self._forward_fc(z)

#         edge_weight = self.act((z[edge_index[0]] * z[edge_index[1]]).sum(dim=1))

#         return edge_weight

#     def _forward_fc(self, z):
#         """
#         Compute the forward pass through the fully connected layers.

#         Args:
#             z (torch.Tensor): The latent space Z.

#         Returns:
#             torch.Tensor: The output after passing through the fully connected layers.
#         """
#         z1 = torch.relu(self.fc(z))
#         z2 = torch.sigmoid(self.fc2(z1))
#         z3 = F.dropout(z2, self.dropout, training=self.training)
#         return z3

class InnerProductDecoderMLP(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, hidden_dims, dropout=0.1, act=torch.sigmoid):
        super(InnerProductDecoderMLP, self).__init__()
        self.dropout = dropout
        self.act = act

        # Initialize hidden_dims as an empty list if None is provided
        hidden_dims = hidden_dims or []

        # Create the layers based on hidden_dims
        self.fc_layers = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.fc_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))

        # Output layer is only added if there are hidden layers
        if hidden_dims:
            self.output_layer = nn.Linear(hidden_dims[-1], hidden_dims[0])
            self._reset_parameters()

    def _reset_parameters(self):
        """
        Reset model parameters using Xavier initialization for all layers.
        """
        for layer in self.fc_layers:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        torch.nn.init.zeros_(self.output_layer.bias)

    def _forward_fc(self, z):
        """
        Compute the forward pass through the fully connected layers.
        """
        for layer in self.fc_layers:
            z = F.relu(layer(z))
            z = F.dropout(z, self.dropout, training=self.training)
        if self.fc_layers:
            z = self.output_layer(z)
        return z

    def forward_all(self, z):
        """
        Compute the forward pass for the entire graph.
        """
        if self.fc_layers:
            z = self._forward_fc(z)
        adj = self.act(torch.matmul(z, z.t()))
        return adj

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass for the given node-pairs.
        """
        if self.fc_layers:
            z = self._forward_fc(z)
        edge_weight = self.act((z[edge_index[0]] * z[edge_index[1]]).sum(dim=1))
        return edge_weight


if __name__ == "__main__":
    pass
