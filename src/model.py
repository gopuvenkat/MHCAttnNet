import os
import math
import sys
import time
from tqdm import tqdm

import config

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import torchtext

class MHCAttnNet(nn.Module):

    def __init__(self, peptide_embedding, mhc_embedding):
        super(MHCAttnNet, self).__init__()
        self.hidden_size = config.BiLSTM_HIDDEN_SIZE
        self.num_layers = config.BiLSTM_NUM_LAYERS

        self.peptide_embedding = peptide_embedding
        self.mhc_embedding = mhc_embedding
        self.relu = nn.ReLU()

        self.peptide_lstm = nn.LSTM(config.EMBED_DIM, self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        self.mhc_lstm = nn.LSTM(config.EMBED_DIM, self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        self.peptide_linear = nn.Linear(self.hidden_size*2, config.LINEAR1_OUT)
        self.mhc_linear = nn.Linear(self.hidden_size*2, config.LINEAR1_OUT)
        self.out_linear = nn.Linear(config.LINEAR1_OUT*2, config.LINEAR2_OUT)
    
    def forward(self, peptide, mhc):
        pep_emb = self.peptide_embedding(peptide)
        mhc_emb = self.mhc_embedding(mhc)
        print("pep_emb:shape", pep_emb.shape)
        print("mhc_emb:shape", pep_emb.shape)

        pep_lstm, _ = self.peptide_lstm(pep_emb.view(len(peptide), 1, -1))
        print("pep_lstm:shape", pep_lstm.shape)
        mhc_lstm, _ = self.mhc_lstm(mhc_emb.view(len(mhc), 1, -1))
        print("mhc_lstm:shape", mhc_lstm.shape)

        pep_linear_out = self.relu(self.peptide_linear(pep_lstm.view(len(peptide), -1)))
        mhc_linear_out = self.relu(self.mhc_linear(mhc_lstm.view(len(peptide), -1)))
        print("pep_linear_out:shape", pep_linear_out.shape)
        print("mhc_linear_out:shape", mhc_linear_out.shape)
        conc = torch.cat((pep_linear_out, mhc_linear_out), 2)
        print("conc:shape", conc.shape)
        out = self.relu(self.out_linear(conc))
        print("out:shape", out.shape)
        return out

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))