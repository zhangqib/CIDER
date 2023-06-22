# CIDER: Counterfactual-Invariant Diffusion-based GNN Explainer for Causal Subgraph Inference

This office implement of CIDER is based on the [PyTorch Geometric]

## Using CIDER
### Requirements
```shell
conda create -n cider requirements.yaml
conda activate cider
```
### Replication the paper's results
```shell
python main.py --dataset=[nci1, mutag]
```

### Subgraph Inference for explaining phenomena
To do this, you should write some code to prepare the dataset and a model, and then use the example code to do this task. The example code is as follows:
```python
from cider import CIDER
from util import train_one_epoch
from torch_geometric.data import DataLoader
from model import ExampleNet
from dataset import ExampleDataset

# prepare the dataset and model
dataset = ExampleDataset()
model = ExampleNet()
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# train the example model
for data in loader:
    ## train the model
    ## ...

# use CIDER to explain the model
## train the cider model
cider = CIDER(...)
for epoch in range(100):
    ## please see main.py for more details
    train_one_epoch(...)

## use the trained cider model to explain the model for a specific data(e.g. dataset[0])
explainations = cider.get_explainations(model, dataset[0])

```


