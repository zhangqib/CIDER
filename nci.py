import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import pandas as pd
import os.path as osp
from torch_geometric.utils import unbatch, unbatch_edge_index
import random


class NCI(InMemoryDataset):

    splits = ['training', 'evaluation', 'testing']

    def __init__(self,
                 root='data/NCI1/',
                 mode='testing',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None) -> None:

        assert mode in self.splits
        self.mode = mode

        super(NCI, self).__init__(root)

        idx = self.processed_file_names.index('{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return [
            i for i in [
                'NCI1_A.txt', 'NCI1.graph_idx', 'NCI1.graph_labels',
                'NCI1.node_labels'
            ]
        ]

    @property
    def processed_file_names(self):
        return ['training.pt', 'evaluation.pt', 'testing.pt']

    def process(self):
        edge = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[0]),
                           header=None,
                           dtype=int)
        graph_idx = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[1]),
                                header=None,
                                dtype=int)
        graph_labels = pd.read_csv(osp.join(self.raw_dir,
                                            self.raw_file_names[2]),
                                   header=None)
        node_lebels = pd.read_csv(osp.join(self.raw_dir,
                                           self.raw_file_names[3]),
                                  header=None)

        node = node_lebels.iloc[:, 1].values
        x = torch.nn.functional.one_hot(
            torch.tensor(node, dtype=torch.int64) - 1).float()
        edge_index = torch.tensor(edge.values.T, dtype=torch.int64)
        edge_index -= 1
        batch = torch.tensor(graph_idx.values, dtype=torch.int64).squeeze()
        batch -= 1
        y_list = torch.tensor(graph_labels.values, dtype=torch.int64)
        edge_index_list = unbatch_edge_index(edge_index, batch)
        x_list = unbatch(x, batch)
        data_list = []
        for x, edge_index, y in zip(x_list, edge_index_list, y_list):
            data = Data(x=x, y=y, edge_index=edge_index)
            data_list.append(data)

        assert len(data_list) == 4110
        random.shuffle(data_list)
        torch.save(self.collate(data_list[1000:]), self.processed_paths[0])
        torch.save(self.collate(data_list[500:1000]), self.processed_paths[1])
        torch.save(self.collate(data_list[:500]), self.processed_paths[2])


if __name__ == "__main__":
    from models import GcnEncoderGraph
    from torch_geometric.loader import DataLoader
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

    # training setting
    batch_size = 128
    lr = 1e-2
    epochs = 10000
    num_workers = 32

    train_set = NCI(mode='training')
    test_set = NCI(mode="testing")
    val_set = NCI(mode='evaluation')

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    criterion = CrossEntropyLoss()

    model = GcnEncoderGraph(input_dim=train_set.num_features,
                            hidden_dim=20,
                            embedding_dim=20,
                            num_layers=4,
                            pred_hidden_dims=[],
                            label_dim=2)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=1e-4
                                 )

    model.cuda()

    for epoch in range(1, epochs+1):
        model.train()
        loss_all = 0
        optimizer.zero_grad()
        for data in train_loader:
            data = data.cuda()
            y_pred = model(data.x, data.edge_index, data.batch)
            loss = criterion(y_pred, data.y)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
        optimizer.step()
        print(loss_all)

        if epoch % 10 == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for data in test_loader:
                    data = data.cuda()
                    y_pred = model(data.x,
                                data.edge_index,
                                data.batch,
                                )
                    correct += float(y_pred.argmax(dim=1).eq(data.y).sum().item())
            print('test acc: ', correct / len(test_set))
