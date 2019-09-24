import os
import time
import sys
import argparse
from tqdm import tqdm

from data_loader import get_dataset
from model import MHCAttnNet
import config

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, f1_score, precision_recall_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np

writer = SummaryWriter()

def fit(model, train_dl, val_dl, loss_fn, opt, epochs, device):
    for epoch in range(epochs):
        print("Epoch", epoch)
        y_actual_train = list()
        y_pred_train = list()
        for row in tqdm(train_dl):
            if row.batch_size == config.batch_size:
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
        f1 = f1_score(y_actual_train, y_pred_train)
        roc_auc = roc_auc_score(y_actual_train, y_pred_train)
        prc_auc = average_precision_score(y_actual_train, y_pred_train)
        # p_train, r_train, _ = precision_recall_curve(y_actual_train, y_pred_train)
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)
        writer.add_scalar('Precision/train', precision, epoch)
        writer.add_scalar('Recall/train', recall, epoch)
        writer.add_scalar('F1/train', f1, epoch)
        writer.add_scalar('ROC_AUC/train', roc_auc, epoch)
        writer.add_scalar('PRC_AUC/train', prc_auc, epoch)
        writer.add_pr_curve('PR_Curve/train', np.asarray(y_actual_train), np.asarray(y_pred_train))
        print(f"Train - Loss : {loss}, Accuracy : {accuracy}, Precision : {precision}, Recall : {recall}, F1-score : {f1}, ROC_AUC : {roc_auc}, PRC_AUC : {prc_auc}")

        y_actual_val = list()
        y_pred_val = list()
        for row in tqdm(val_dl):
            if row.batch_size == config.batch_size:
                y_pred = model(row.peptide, row.mhc_amino_acid)
                y_pred_idx = torch.max(y_pred, dim=1)[1]
                y_actual = row.bind
                y_actual_val += list(y_actual.cpu().data.numpy())
                y_pred_val += list(y_pred_idx.cpu().data.numpy())
                loss = loss_fn(y_pred, y_actual)
        accuracy = accuracy_score(y_actual_val, y_pred_val)
        precision = precision_score(y_actual_val, y_pred_val)
        recall = recall_score(y_actual_val, y_pred_val)
        f1 = f1_score(y_actual_val, y_pred_val)
        roc_auc = roc_auc_score(y_actual_val, y_pred_val)
        prc_auc = average_precision_score(y_actual_val, y_pred_val)
        # p_val, r_val, _ = precision_recall_curve(y_actual_val, y_pred_val)
        writer.add_scalar('Loss/val', loss, epoch)
        writer.add_scalar('Accuracy/val', accuracy, epoch)
        writer.add_scalar('Precision/val', precision, epoch)
        writer.add_scalar('Recall/val', recall, epoch)
        writer.add_scalar('F1/val', f1, epoch)
        writer.add_scalar('ROC_AUC/val', roc_auc, epoch)
        writer.add_scalar('PRC_AUC/val', prc_auc, epoch)
        writer.add_pr_curve('PR_Curve/val', np.asarray(y_actual_train), np.asarray(y_pred_train))
        print(f"Validation - Loss : {loss}, Accuracy : {accuracy}, Precision : {precision}, Recall : {recall}, F1-score : {f1}, ROC_AUC : {roc_auc}, PRC_AUC : {prc_auc}")

        if epoch % 2 == 0:
            torch.save(model.state_dict(), config.model_name)


if __name__ == "__main__":
    torch.manual_seed(3)  # for reproducibility

    device = config.device
    epochs = config.epochs
    print(device)
    dataset_cls, train_loader, val_loader, test_loader, peptide_embedding, mhc_embedding = get_dataset(device)
    model = MHCAttnNet(peptide_embedding, mhc_embedding)
    # model.load_state_dict(torch.load(config.model_name))
    model.to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    fit(model=model, train_dl=train_loader, val_dl=val_loader, loss_fn=loss_fn, opt=optimizer, epochs=epochs, device=device)

writer.close()