"""
File: ba.py
Purpose: BA-2motif Dataset class in pyg InMemoryDataset format
Author: Anonymous Author
Date: May 20, 2023
License: MIT License
"""

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif
from torch_geometric.datasets.motif_generator import CycleMotif
import random


class BA(InMemoryDataset):
    splits = ["training", "evaluation", "testing"]

    def __init__(self, root="data/BA-2motif", mode="testing") -> None:
        assert mode in self.splits
        self.mode = mode

        super(BA, self).__init__(root)

        idx = self.processed_file_names.index("{}.pt".format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def processed_file_names(self):
        return ["training.pt", "evaluation.pt", "testing.pt"]

    def process(self):
        num_nodes_ba_shape = 20
        num_features = 10
        num_graphs_per_motif = 1024 + 128 + 128

        dataset1 = ExplainerDataset(
            graph_generator=BAGraph(num_nodes=num_nodes_ba_shape, num_edges=1),
            motif_generator=HouseMotif(),
            num_motifs=1,
            num_graphs=num_graphs_per_motif,
        )

        dataset2 = ExplainerDataset(
            graph_generator=BAGraph(num_nodes=num_nodes_ba_shape, num_edges=1),
            motif_generator=CycleMotif(5),
            num_motifs=1,
            num_graphs=num_graphs_per_motif,
        )

        x = torch.ones(num_nodes_ba_shape + 5, num_features) / num_features

        data_list = []

        for i in range(num_graphs_per_motif):
            data_list.append(
                Data(
                    x=x,
                    edge_index=dataset1[i].edge_index,
                    y=torch.tensor([0], dtype=torch.long),
                    index=i,
                )
            )

        for i in range(num_graphs_per_motif):
            data_list.append(
                Data(
                    x=x,
                    edge_index=dataset2[i].edge_index,
                    y=torch.tensor([1], dtype=torch.long),
                    index=i + num_graphs_per_motif,
                )
            )

        random.shuffle(data_list)
        torch.save(self.collate(data_list[512:]), self.processed_paths[0])
        torch.save(self.collate(data_list[256:512]), self.processed_paths[1])
        torch.save(self.collate(data_list[:256]), self.processed_paths[2])


if __name__ == "__main__":
    from models import GcnEncoderGraph
    from torch_geometric.loader import DataLoader
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

    # training setting
    batch_size = 512
    lr = 1e-4
    epochs = 30000
    num_workers = 32

    train_set = BA(mode="training")
    test_set = BA(mode="testing")
    val_set = BA(mode="evaluation")

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    criterion = CrossEntropyLoss()

    model = GcnEncoderGraph(
        input_dim=train_set.num_features,
        hidden_dim=20,
        embedding_dim=20,
        num_layers=4,
        pred_hidden_dims=[],
        label_dim=2,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.cuda()
    model.load_state_dict(torch.load('params/ba2_net.pt'))
    print(model)
    best_acc = 0
    for epoch in range(1, epochs + 1):
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
        # print(loss_all)

        if epoch % 10 == 0:
            model.eval()
            correct_test = 0
            correct_train = 0
            with torch.no_grad():
                for data in test_set:
                    data = data.cuda()
                    y_pred = model(
                        data.x,
                        data.edge_index,
                        # data.batch,
                    )
                    correct_test += float(y_pred.argmax(dim=1).eq(data.y).sum().item())
                print("test acc: ", correct_test / len(test_set))
                for data in train_loader:
                    data = data.cuda()
                    y_pred = model(
                        data.x,
                        data.edge_index,
                        data.batch,
                    )
                    correct_train += float(y_pred.argmax(dim=1).eq(data.y).sum().item())
                print("train acc: ", correct_train / len(train_set))
                if correct_test / len(test_set) > best_acc:
                    best_acc = correct_test / len(test_set)
                    torch.save(model.state_dict(), "params/ba2_net.pt")
                    print("best acc", best_acc)
