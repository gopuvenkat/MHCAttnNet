import os
import time
import sys
import argparse

from data_loader import get_dataset
from model import MHCAttnNet
import config

import torch
import torch.nn as nn
import torch.optim as optim

def fit(model, train_dl, val_dl, loss_fn, opt, epochs):
    num_batch = len(train_dl)
    for epoch in range(epochs):
        y_true_train = list()
        y_pred_train = list()
        total_loss_train = 0

        for row in train_dl:
            print("row.peptide:shape", row.peptide.shape)
            print("row.mhc_amino_acid:shape", row.mhc_amino_acid.shape)
            y_pred = model(row.peptide, row.mhc_amino_acid)
            print("y_pred:shape", y_pred.shape)
            y_actual = row.bind
            print("y_actual:shape", y_actual.shape)

if __name__ == "__main__":

    device = config.device
    epochs = config.epochs

    dataset_cls, train_loader, val_loader, test_loader, peptide_embedding, mhc_embedding = get_dataset(device)
    model = MHCAttnNet(peptide_embedding, mhc_embedding)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    fit(model, train_loader, val_loader, loss_fn, optimizer, epochs)