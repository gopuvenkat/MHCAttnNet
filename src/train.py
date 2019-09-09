import os
import time
import sys
import argparse
from tqdm import tqdm

from data_loader import get_dataset
from model import MHCAttnNet
import config

from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def fit(model, train_dl, val_dl, loss_fn, opt, epochs, device):
    num_batch = len(train_dl)
    for epoch in range(1, epochs+1):
        print("Epoch", epoch)
        
        y_actual_train = list()
        y_pred_train = list()
        for row in tqdm(train_dl):
            y_pred = model(row.peptide, row.mhc_amino_acid)
            y_pred_idx = torch.max(y_pred, dim=1)[1]
            y_actual = row.bind
            y_actual_train += list(y_actual.cpu().data.numpy())
            y_pred_train += list(y_pred_idx.cpu().data.numpy())
            loss = loss_fn(y_pred, y_actual)
            opt.zero_grad()
            loss.backward()
            optimizer.step()
        accuracy = accuracy_score(y_actual_train, y_pred_train)
        precision = precision_score(y_actual_train, y_pred_train)
        recall = recall_score(y_actual_train, y_pred_train)
        print(f"Train - Loss : {loss}, Accuracy : {accuracy}, Precision : {precision}, Recall : {recall}")

        y_actual_val = list()
        y_pred_val = list()
        for row in tqdm(val_dl):
            y_pred = model(row.peptide, row.mhc_amino_acid)
            y_pred_idx = torch.max(y_pred, dim=1)[1]
            y_actual = row.bind
            y_actual_val += list(y_actual.cpu().data.numpy())
            y_pred_val += list(y_pred_idx.cpu().data.numpy())
            loss = loss_fn(y_pred, y_actual)            
        accuracy = accuracy_score(y_actual_val, y_pred_val)
        precision = precision_score(y_actual_train, y_pred_train)
        recall = recall_score(y_actual_train, y_pred_train)
        print(f"Validation - Loss : {loss}, Accuracy : {accuracy}, Precision : {precision}, Recall : {recall}")

        if(epoch%2 == 0):
            torch.save(model.state_dict(), config.model_name)


if __name__ == "__main__":

    device = config.device
    epochs = config.epochs

    dataset_cls, train_loader, val_loader, test_loader, peptide_embedding, mhc_embedding = get_dataset(device)
    model = MHCAttnNet(peptide_embedding, mhc_embedding)
    model.load_state_dict(torch.load(config.model_name))
    model.to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    fit(model=model, train_dl=train_loader, val_dl=val_loader, loss_fn=loss_fn, opt=optimizer, epochs=epochs, device=device)
