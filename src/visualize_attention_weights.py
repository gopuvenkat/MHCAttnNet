import math
import os

from data_loader import get_dataset
from model import MHCAttnNet
import config 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch 
import torch.nn as nn

def plot_matrix(attn_model, inp, text_inp):
    weight = attn_model.weight
    b = attn_model.b
    eij = torch.matmul(inp, weight)
    tanh = nn.Tanh()
    eij = tanh(torch.add(eij, b))
    v = torch.exp(torch.matmul(eij, attn_model.context_vector))
    v = v / torch.sum(v, dim=1, keepdim=True)
    weighted_input = inp * v
    matrix = weighted_input.squeeze(0).cpu().detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, cmap="bone")
    fig.colorbar(cax)

    # set up axes
    ax.set_yticklabels(text_inp[0])

    # show label at every tick
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

if __name__ == "__main__":
    torch.manual_seed(3)  # for reproducibility

    device = config.device
    epochs = config.epochs
    dataset_cls, train_loader, val_loader, test_loader, peptide_embedding, mhc_embedding = get_dataset(device)
    model = MHCAttnNet(peptide_embedding, mhc_embedding)
    model.load_state_dict(torch.load(config.model_name))
    model.to(device)

    row = next(iter(train_loader))
    print("row.raw_peptide", row.raw_peptide)
    print("row.raw_mhc_amino_acid", row.raw_mhc_amino_acid)
    pep_emb = model.peptide_embedding(row.peptide)
    mhc_emb = model.mhc_embedding(row.mhc_amino_acid)

    inp, _ = model.mhc_lstm(mhc_emb)
    text_inp = row.raw_mhc_amino_acid
    attn_model = model.mhc_attn
    plot_matrix(attn_model, inp, text_inp)

    inp, _ = model.peptide_lstm(pep_emb)
    text_inp = row.raw_peptide
    attn_model = model.peptide_attn
    plot_matrix(attn_model, inp, text_inp)

    plt.show()