import os
import time
import sys
import argparse
from tqdm import tqdm

from data_loader import get_dataset
from model import MHCAttnNet
import config

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, \
    f1_score, precision_recall_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np


def fit(model, train_dl, val_dl, loss_fn, opt, epochs, device):
    num_batch = len(train_dl)
    scores = dict()
    # scores['loss'] = list()
    # scores['accuracy'] = list()
    # scores['precision'] = list()
    # scores['recall'] = list()
    # scores['roc_auc'] = list()

    loss_train = list()
    loss_val = list()
    accuracy_train = list()
    accuracy_val = list()
    recall_train = list()
    recall_val = list()
    precision_train = list()
    precision_val = list()
    auc_train = list()
    auc_val = list()
    f1_train = list()
    f1_val = list()
    rpc_train = list()
    rpc_val = list()
    pr = list()

    for epoch in range(1, epochs + 1):
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
        auc_rpc = average_precision_score(y_actual_train, y_pred_train)
        p_train, r_train, _ = precision_recall_curve(y_actual_train, y_pred_train)
        print(
            f"Train - Loss : {loss}, Accuracy : {accuracy}, Precision : {precision}, Recall : {recall}, F1-score : {f1}, ROC_AUC : {roc_auc}, AuRpC : {auc_rpc}")

        loss_train.append(loss)
        accuracy_train.append(accuracy)
        precision_train.append(precision)
        recall_train.append(recall)
        auc_train.append(roc_auc)
        f1_train.append(f1)
        rpc_train.append(auc_rpc)

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
        auc_rpc = average_precision_score(y_actual_val, y_pred_val)
        p_val, r_val, _ = precision_recall_curve(y_actual_val, y_pred_val)

        print(
            f"Validation - Loss : {loss}, Accuracy : {accuracy}, Precision : {precision}, Recall : {recall}, F1-score : {f1}, ROC_AUC : {roc_auc}, AuRpC : {auc_rpc}")

        loss_val.append(loss)
        accuracy_val.append(accuracy)
        precision_val.append(precision)
        recall_val.append(recall)
        auc_val.append(roc_auc)
        f1_val.append(f1)
        rpc_val.append(auc_rpc)
        pr.append([(p_train, r_train), (p_val, r_val)])

        if epoch % 2 == 0:
            torch.save(model.state_dict(), config.model_name)

    loss = [loss_train, loss_val]
    accuracy = [accuracy_train, accuracy_val]
    precision = [precision_train, precision_val]
    recall = [recall_train, recall_val]
    roc_auc = [auc_train, auc_val]
    auc_rpc = [rpc_train, rpc_val]
    f1 = [f1_train, f1_val]
    pr = pr

    scores['loss'] = loss
    scores['accuracy'] = accuracy
    scores['precision'] = precision
    scores['recall'] = recall
    scores['roc_auc'] = roc_auc
    scores['auc_rpc'] = auc_rpc
    scores['f1'] = f1
    scores['pr'] = pr

    return scores


def plot(metrics, scores, epochs):
    x = np.asarray(range(1, epochs + 1))
    i = 0
    for metric in metrics:
        if metric == 'auc_rpc':
            p_train = np.asarray(scores['pr'][-1][0][0])
            p_val = np.asarray(scores['pr'][-1][1][0])
            r_train = np.asarray(scores['pr'][-1][0][1])
            r_val = np.asarray(scores['pr'][-1][1][1])

            train_score = scores['auc_rpc'][0][-1]
            val_score = scores['auc_rpc'][1][-1]

            plt.figure(i)
            plt.plot(r_train, p_train)
            plt.plot(r_val, p_val)
            plt.legend(['auRPC-train = {}'.format(train_score), 'auRPC-val = {}'.format(val_score)], loc=0)
            plt.xlabel("recall")
            plt.ylabel("precision")
            plt.savefig('../visualizations/' + metric + '-' + str(epochs) + '.png')
            i += 1
        else:
            try:
                y1 = np.asarray(scores[metric][0])
                y2 = np.asarray(scores[metric][1])
            except:
                print("Metric not found")

            plt.figure(i)
            plt.plot(x, y1)
            plt.plot(x, y2)
            plt.legend(['train', 'val'], loc=0)
            plt.xlabel("epochs")
            plt.ylabel(metric)
            plt.savefig('../visualizations/' + metric + '-' + str(epochs) + '.png')
            i += 1


if __name__ == "__main__":
    torch.manual_seed(3)  # for reproducibility

    device = config.device
    epochs = config.epochs

    dataset_cls, train_loader, val_loader, test_loader, peptide_embedding, mhc_embedding = get_dataset(device)
    model = MHCAttnNet(peptide_embedding, mhc_embedding)
    # model.load_state_dict(torch.load(config.model_name))
    model.to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    scores = fit(model=model, train_dl=train_loader, val_dl=val_loader, loss_fn=loss_fn, opt=optimizer, epochs=epochs,
                 device=device)

    metrics = ['loss', 'f1', 'precision', 'recall', 'auc_rpc']
    plot(metrics, scores, epochs)
