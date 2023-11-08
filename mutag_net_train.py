import torch
from torch_geometric.loader import DataLoader
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from nci import NCI
from models import GcnEncoderGraph
from mutag import Mutagenicity

# training setting
batch_size = 128
lr = 1e-2
epochs = 10000
num_workers = 32

train_set = Mutagenicity('data/MUTAG', mode='training')
test_set = Mutagenicity('data/MUTAG', mode="testing")
val_set = Mutagenicity('data/MUTAG', mode='evaluation')

test_loader = DataLoader(test_set,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=num_workers)
val_loader = DataLoader(val_set,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers)
train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers)

criterion = CrossEntropyLoss()

model = GcnEncoderGraph(input_dim=14,
                        hidden_dim=50,
                        embedding_dim=10,
                        num_layers=3,
                        pred_hidden_dims=[10, 10],
                        label_dim=2)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

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

    if epoch % 10 == 0:
        model.eval()
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.cuda()
                y_pred = model(
                    data.x,
                    data.edge_index,
                    data.batch,
                )
                correct += float(y_pred.argmax(dim=1).eq(data.y).sum().item())
        if correct / len(test_set) > best_acc:
            best_acc = correct / len(test_set)
            torch.save(model.state_dict(), 'params/mutag_net.pt')
        print('test acc: ', correct / len(test_set), ' best acc', best_acc)