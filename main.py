"""
File: main.py
Purpose: main script for training and testing
Author: Qibin Zhang
Date: May 20, 2023
License: MIT License
"""

import os
import argparse
import torch
import torch.distributed as dist
from nci import NCI
from mutag import Mutagenicity
from ba import BA
from torch_geometric.loader import DataLoader
from datetime import datetime
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from models import GcnEncoderGraph
from cider import CIDER
from utils import train_one_epoch, validate
from torch.utils.tensorboard import SummaryWriter

dataset_mapping = {
    "nci1": NCI,
    "mutag": Mutagenicity,
    "ba2": BA,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument(
        "--batch_size", default=256, type=int, help="batch size per GPU"
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        help="start epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--epochs", default=500, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--alpha_r", default=0.01, type=float)
    parser.add_argument("--alpha_c", default=0.1, type=float)
    parser.add_argument("--alpha_kld", default=1, type=float)
    parser.add_argument("--alpha_reg", default=0.1, type=float)
    parser.add_argument("--alpha_diff", default=1.0, type=float)
    parser.add_argument(
        "--N", default=5, type=int, help="times of sampling while computing causal loss"
    )
    parser.add_argument("--in_channels", default=37, type=int)
    parser.add_argument("--hidden_channels1", default=32, type=int)
    parser.add_argument("--hidden_channels2", default=64, type=int)
    parser.add_argument("--hidden_channels3", default=10, type=int)
    parser.add_argument("--workers", default=16, type=int)
    parser.add_argument("--sparsity", default=0.3, type=float)
    parser.add_argument(
        "--resume", action="store_true", help="resume task model or not"
    )
    parser.add_argument(
        "--random_sparsity",
        action="store_true",
        help="random sparsity in training process or not",
    )
    parser.add_argument("--task", action="store_true", help="use task output")
    parser.add_argument(
        "--dataset",
        default="nci1",
        type=str,
        choices=dataset_mapping.keys(),
        required=True,
        help="dataset name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation",
    )
    parser.add_argument("--writer", action="store_true", help="use tensorboard or not")

    # Debug Mode
    parser.add_argument("--debug_mode", action="store_true", help="debug mode or not")
    args = parser.parse_args()
    return args


def main(args):
    ### device ###
    device = torch.device(args.device)

    ### data ###
    train_set = dataset_mapping[args.dataset](mode="training")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    val_set = dataset_mapping[args.dataset](mode="evaluation")

    test_set = dataset_mapping[args.dataset](mode="testing")

    ### Debug Mode ###
    if args.debug_mode:
        train_set = dataset_mapping[args.dataset](mode="training")
        train_loader = DataLoader(
            train_set[0:12],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )

    ### model ###
    if args.dataset == "ba2":
        task_model = GcnEncoderGraph(
            input_dim=train_set.num_features,
            hidden_dim=20,
            embedding_dim=20,
            num_layers=4,
            pred_hidden_dims=[],
            label_dim=2,
        )
    else:
        task_model = GcnEncoderGraph(
            input_dim=train_set.num_features,
            hidden_dim=50,
            embedding_dim=10,
            num_layers=3,
            pred_hidden_dims=[10, 10],
            label_dim=2,
        )

    explainer_model = CIDER(
        train_set.num_features,
        hidden_channels1=args.hidden_channels1,
        hidden_channels2=args.hidden_channels2,
        hidden_channels3=args.hidden_channels3,
        task_model=task_model,
    )
    explainer_model.to(device)
    task_model.to(device)

    ### optimizer ###
    optimizer = torch.optim.Adam(
        explainer_model.parameters(), lr=args.lr, weight_decay=1e-5
    )

    if args.task:
        args.resume = True
        path = "./params/" + args.dataset + "_net.pt"
        task_model.load_state_dict(torch.load(path))
        for data in train_set:
            data.y[:] = (
                task_model(data.x.to(device), data.edge_index.to(device))
                .argmax(dim=1)
                .cpu()
                .detach()
            )

        for data in val_set:
            data.y[:] = (
                task_model(data.x.to(device), data.edge_index.to(device))
                .argmax(dim=1)
                .cpu()
                .detach()
            )

        for data in test_set:
            data.y[:] = (
                task_model(data.x.to(device), data.edge_index.to(device))
                .argmax(dim=1)
                .cpu()
                .detach()
            )

    ### resume training if necessary ###
    if args.resume:
        path = "./params/" + args.dataset + "_net.pt"
        task_model.load_state_dict(torch.load(path))

    ### tensorboard setting ###
    if args.writer:
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        train_log_dir = "logs/" + args.dataset + "/" + TIMESTAMP
        writer = SummaryWriter(train_log_dir)

    ### print info about training ###
    print(args)

    ### main loop ###
    best_result = {}
    test_result = {}
    for epoch in range(args.start_epoch, args.epochs):
        losses = train_one_epoch(
            train_loader,
            explainer_model,
            CrossEntropyLoss(),
            BCEWithLogitsLoss(),
            optimizer,
            task_model,
            epoch,
            device,
            args,
        )
        if args.writer:
            for key, value in losses.items():
                writer.add_scalar("train/" + key, value, epoch)
            writer.add_scalars("train/losses", losses, epoch)

        if epoch % 5 == 0:
            result_val = validate(
                val_set,
                explainer_model,
                explainer_model.task_model,
                batch_size=args.batch_size * args.N,
                device=device
            )
            if args.writer:
                writer.add_scalars("validate/test set acc", result_val, epoch)
            print("epoch ", str(epoch), ":", result_val)

            if sum(result_val.values()) > sum(best_result.values()):
                best_result = result_val
                test_result = validate(
                    test_set,
                    explainer_model,
                    explainer_model.task_model,
                    batch_size=args.batch_size * args.N,
                    device=device
                )
                torch.save(
                    explainer_model.state_dict(),
                    os.path.join("./params/explainer_" + args.dataset + ".ckpt"),
                )
                print("best result:", best_result)
                print("test result:", test_result)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(parse_args())
