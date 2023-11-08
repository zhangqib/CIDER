from typing import List
from torch_geometric.data import InMemoryDataset, Data, Batch
import pandas as pd
import os.path as osp
import torch
import random

class Immune(InMemoryDataset):

    splits = ['training', 'evaluation', 'testing']
    gene_name_list = None

    def __init__(self, root: str = 'data/immune', mode: str = 'testing'):
        assert mode in self.splits
        self.mode = mode

        super().__init__(root)

        idx = self.processed_file_names.index('{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self) -> List[str]:
        return ['SKCM_Immune.txt', 'SKCM_Survival.txt', 'Tcell.CSN.txt']

    @property
    def processed_file_names(self) -> List[str]:
        return ['training.pt', 'evaluation.pt', 'testing.pt']

    @property
    def gene_names(self) -> List[str]:
        if self.gene_name_list is not None:
            return self.gene_name_list
        else:
            node_features = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[0]), sep='\t')
            self.gene_name_list = node_features.index.values
            return self.gene_name_list


    def process(self):
        # more data detail in `readMe.txt`
        # node_features: all gene expressions of the sample
        node_features = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[0]), sep='\t')
        # graph_label: the survival ['Low', 'High', 'Unknown']
        graph_labels = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[1]), sep='\t')
        # edge_indexes: the edge between genes
        edge_indexes = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[2]), sep='\t')

        # first, we should select samples with labels != Unknown
        ## create mask for data, select data which graph_label != Unknown
        mask = graph_labels['group'] != 'Unknown'
        # mask = mask.to_list()

        ## use mask to select sample id
        sample_ids = node_features.columns[mask]

        ## use mask to select node_features
        node_features = node_features[node_features.columns[mask]]

        ## use mask to select edge_index
        edge_indexes = edge_indexes[edge_indexes['sampleID'].isin(sample_ids)]

        ### map edge_indexes's genes to the index(int) according the node_features.index(gene's name)
        edge_index = [[], []]
        edge_index[0] = edge_indexes['gene1'].map(lambda x: node_features.index.get_loc(x)).values
        edge_index[1] = edge_indexes['gene2'].map(lambda x: node_features.index.get_loc(x)).values

        ### get Batch info
        batch = edge_indexes['sampleID'].map(lambda x: sample_ids.get_loc(x)).values

        ## use mask to select graph labels, and map 'Low' to 0, 'High' to 1
        graph_labels = graph_labels[mask]['group'].replace({'Low': 0, 'High': 1})


        # convert data to Tensor
        ## x.shape: [#samples, #genes]
        x = torch.tensor(node_features.values, dtype=torch.float).T
        
        ## edge_index.shape [2, #all_edges]
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        ## y.shape [#samples]
        y = torch.tensor(graph_labels.values, dtype=torch.long)

        data_list = []
        for i in range(len(sample_ids)):
            data = Data(x=x[i].unsqueeze(1), edge_index=edge_index.T[batch == i].T, y=y[i], sample_id=sample_ids[i])
            data_list.append(data)

        assert len(data_list) == 230

        random.shuffle(data_list)

        torch.save(self.collate(data_list[46:]), self.processed_paths[0])
        torch.save(self.collate(data_list[23:46]), self.processed_paths[1])
        torch.save(self.collate(data_list[:23]), self.processed_paths[2])

if __name__ == "__main__":
    from models import GcnEncoderGraph
    from torch_geometric.loader import DataLoader
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
    from utils import evaluate_graphs_accuracy

    # training setting
    batch_size = 128
    lr = 1e-1
    epochs = 10000
    num_workers = 16

    train_set = Immune(mode='training')
    test_set = Immune(mode="testing")
    val_set = Immune(mode='evaluation')

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    criterion = CrossEntropyLoss()

    model = GcnEncoderGraph(input_dim=train_set.num_features,
                            hidden_dim=8,
                            embedding_dim=16,
                            num_layers=6,
                            pred_hidden_dims=[],
                            label_dim=2)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=0
                                 )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    best_accuracy = 0

    for epoch in range(1, epochs+1):
        model.train()
        loss_all = 0
        optimizer.zero_grad()
        for data in train_loader:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            y = data.y.to(device)
            batch = data.batch.to(device)
            y_pred = model(x, edge_index, batch)
            loss = criterion(y_pred, y)
            # l1_regularization = sum(torch.abs(param) for name, param in model.named_parameters() if 'weight' in name)
            # loss += 1e-4 * l1_regularization

            loss.backward()
            loss_all += loss.item() * data.num_graphs
        optimizer.step()

        if epoch % 10 == 0:
            accuracy = evaluate_graphs_accuracy(test_loader, model, device)
            best_accuracy = accuracy if accuracy > best_accuracy else best_accuracy
            print(f'Epoch: {epoch:03d}, Loss: {loss_all:.4f}, Test: {accuracy:.4f}, Curr_Best: {best_accuracy:.4f}')