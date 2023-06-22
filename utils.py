# import math
from typing import List
import torch
import numpy as np
from torch_geometric.utils import to_dense_adj, dense_to_sparse, add_self_loops
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss, L1Loss
from torch_geometric.data import Data
from math import ceil
import random
import numpy as np
import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix


def dpp_diversity(samples: List[torch.Tensor],
                  distance_criterion=torch.nn.CosineSimilarity(),
                  submethod="inverse_dist") -> torch.Tensor:
    sample_nums = len(samples)
    det_entries = torch.zeros([sample_nums, sample_nums])
    for i in range(sample_nums):
        for j in range(sample_nums):
            if submethod == "inverse_dist":
                det_entries[(i, j)] = 1.0 / (
                    1.0 + distance_criterion(samples[i], samples[j]))
            else:
                det_entries[(i, j)] = 1.0 / (torch.exp(
                    distance_criterion(samples[i], samples[j])))
            if i == j:
                det_entries[(i, j)] += 0.0001
    diversity_loss = torch.det(det_entries)
    return diversity_loss


def train_one_epoch(train_loader, explainer_model, causal_criterion,
                    reconstruction_criterion, optimizer, task_model, epoch,
                    args):
    # loss clear
    total_loss_r = 0  # reconstruction loss
    total_loss_c = 0  # causal loss
    total_loss_kld_causal = 0  # kl-divergence loss
    total_loss_kld_non_causal = 0
    # loss_d = 0
    total_loss_diff = 0
    total_loss_reg = 0
    total_loss = 0

    acc = 0

    diff_criterion = MSELoss()
    reg_criterion = L1Loss()

    sparsity = random.uniform(0.5, 0.9) if args.random_sparsity else 0.8

    explainer_model.train()
    task_model.train()
    optimizer.zero_grad()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for data in train_loader:
        data = data.to(device)
        mu_causal, mu_non_causal, logvar_causal, logvar_non_causal = explainer_model.encode(
            data.x, data.edge_index, data.edge_attr)

        edge_weight_causal, edge_weight_non_causal, mu_causal, mu_non_causal, logvar_causal, logvar_non_causal = explainer_model(
            data.x, data.edge_index)

        mu_causal, mu_non_causal, logvar_causal, logvar_non_causal = explainer_model.encode(
            data.x, data.edge_index, data.edge_attr)

        edge_weight_reconstruction = edge_weight_causal + edge_weight_non_causal

        loss_r = reconstruction_criterion(
            edge_weight_reconstruction,
            torch.ones_like(edge_weight_reconstruction).to(device))
        loss_diff = -diff_criterion(edge_weight_causal, edge_weight_non_causal)
        loss_kld_causal = -0.5 * torch.sum(1. + logvar_causal - mu_causal**2 -
                                           logvar_causal.exp())
        loss_kld_non_causal = -0.5 * torch.sum(1. + logvar_non_causal -
                                               mu_non_causal**2 -
                                               logvar_non_causal.exp())
        loss_reg = reg_criterion(
            edge_weight_causal,
            torch.zeros_like(edge_weight_causal).to(device))

        # compute causal loss
        loss_c, correct_batch = explainer_model.CF_forward(data,
                                                           causal_criterion,
                                                           num_sample=args.N,
                                                           sparsity=sparsity)

        # Backpropagate the total loss and update the model parameters using the optimizer
        loss = args.alpha_r * loss_r + (args.alpha_c * loss_c) + (
            args.alpha_kld * (loss_kld_causal + loss_kld_non_causal)
        ) + args.alpha_reg * loss_reg + args.alpha_diff * loss_diff
        loss.backward()
        optimizer.step()

        total_loss_r += loss_r.item()
        total_loss_c += loss_c.item()
        total_loss_kld_causal -= loss_kld_causal.item()
        total_loss_kld_non_causal -= loss_kld_non_causal.item()
        total_loss_reg += loss_reg.item()
        total_loss_diff -= loss_diff.item()
        total_loss += loss.item()

        torch.cuda.empty_cache()
        acc += correct_batch
    acc /= len(train_loader)

    return {
        "total loss": total_loss,
        "causal loss": total_loss_c,
        "regular loss": total_loss_reg,
        "reconstruction loss": total_loss_r,
        "causal diversity loss": total_loss_kld_causal,
        "non causal diversity": total_loss_kld_non_causal,
        "diff loss": total_loss_diff,
        "trian acc": acc
    }


@torch.no_grad()
def validate(
        val_set,
        explainer_model,
        task_model,
        batch_size,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        require_cm=False):

    explainer_model.eval()
    task_model.eval()
    spa_set = val_set

    sparsities = [round(i / (i + 1), 2) for i in range(1, 10)][::-1]
    true_sparsities = [round(0.1 * i, 2) for i in range(9, 0, -1)]
    results = {}
    if require_cm:
        cms = {}

    for sparsity, true_sparsity in zip(sparsities, true_sparsities):
        tmp_set = []
        for spa_data in spa_set:
            spa_data = spa_data.cuda()
            edge_weight_causal, _, _, _, _, _ = explainer_model(
                spa_data.x, spa_data.edge_index)

            topk = max(ceil(spa_data.edge_index.shape[1] * sparsity), 1)
            threshold = edge_weight_causal.sort(
                descending=True).values.topk(topk).values[-1]
            tmp_edge = spa_data.edge_index.T[edge_weight_causal >= threshold].T
            tmp_set.append(
                Data(x=spa_data.x, edge_index=tmp_edge, y=spa_data.y))
        spa_set = tmp_set
        val_loader = DataLoader(tmp_set, batch_size=batch_size)
        if require_cm:
            results[str(true_sparsity)], cms[str(
                true_sparsity)] = evaluate_graphs_accuracy(val_loader,
                                                           task_model,
                                                           require_cm=True)
        else:
            results[str(true_sparsity)] = evaluate_graphs_accuracy(
                val_loader, task_model)

    spa_edge_count = 0
    edge_count = 0
    for spa_data, data in zip(spa_set, val_set):
        spa_edge_count += spa_data.edge_index.shape[1]
        edge_count += data.edge_index.shape[1]
    print('sparsitiy', spa_edge_count / edge_count)

    if require_cm:
        return results, cms
    else:
        return results


@torch.no_grad()
def evaluate_graphs_accuracy(
        test_loader: torch_geometric.loader.DataLoader,
        model: torch.nn.Module,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        require_cm=False):
    """Evaluates the accuracy of a graph neural network model on a test dataset.

    Args:
        test_loader (torch_geometric.loader.DataLoader): A data loader for the test dataset.
        model (torch.nn.Module): The graph neural network model to evaluate.
        device (torch.device or str, optional): The device to use for computation. If None, the
            default device will be used. Defaults to None.
        require_cm (bool, optional): Whether to return the confusion matrix along with accuracy.
            Defaults to False.

    Returns:
        float or Tuple[float, np.ndarray]: The accuracy of the model on the test dataset. If
            `require_cm` is True, the confusion matrix is also returned as a numpy array.

    Raises:
        ValueError: If `test_loader` or `model` is not provided.
    """

    model.to(device)
    model.eval()
    correct = 0
    all_labels = []
    if require_cm:
        all_predictions = []

    for data in test_loader:
        output = model(
            data.x.to(device),
            data.edge_index.to(device),
            data.batch.to(device),
        )
        predictions = output.argmax(dim=1).cpu().numpy().reshape(-1)
        labels = data.y.cpu().numpy().reshape(-1)
        all_labels.extend(labels)
        correct += float((predictions == labels).sum())
        if require_cm:
            all_predictions.extend(predictions)
    accuracy = correct / len(test_loader.dataset)
    if require_cm:
        cm = confusion_matrix(all_labels, all_predictions)
        return accuracy, cm
    else:
        return accuracy


@torch.no_grad()
def random_test(test_loader, model):
    model.eval()
    correct = 0

    for data in test_loader:
        adj = to_dense_adj(data.edge_index)
        random_adj = torch.randn_like(adj) > 0
        random_edge, _ = dense_to_sparse(random_adj)
        # data = data.cuda()
        output = model(
            data.x.cuda(),
            random_edge.cuda(),
            data.batch.cuda(),
        )
        correct += float(output.argmax(dim=1).eq(data.y.cuda()).sum().item())
    return correct / (len(test_loader.dataset))


if __name__ == '__main__':
    pass
