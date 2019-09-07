import os
import math
import sys
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import torchtext

class MHCAttnNet(nn.Module):

    def __init__(self, peptide_embedding, mhc_embedding):
        super(MHCAttnNet, self).__init__()
        self.hidden_size = config.BiLSTM_HIDDEN_SIZE
        self.peptide_embedding = peptide_embedding
        self.mhc_embedding = mhc_embedding
        self.relu = nn.ReLU()

        self.peptide_lstm = nn.LSTM(config.EMBED_DIM, self.hidden_size, bidirectional=True, batch_first=True)
        self.mhc_lstm = nn.LSTM(config.EMBED_DIM, self.hidden_size, bidirectional=True, batch_first=True)
        self.peptide_linear = nn.Linear(self.hidden_size*2, config.LINEAR1_OUT)
        self.mhc_linear = nn.Linear(self.hidden_size*2, config.LINEAR1_OUT)
        self.out_linear = nn.Linear(config.LINEAR1_OUT*2, config.LINEAR2_OUT)
    
    def forward(self, peptide, mhc):
        pep_emb = self.peptide_embedding(peptide)
        pep_emb = torch.squeeze(torch.unsqueeze(pep_emb, 0))
        mhc_emb = self.mhc_embedding(mhc)
        mhc_emb = torch.squeeze(torch.unsqueeze(mhc_emb, 0))

        pep_lstm, _ = self.peptide_lstm(pep_emb)
        mhc_lstm, _ = self.mhc_lstm(mhc_emb)

        pep_linear_out = self.relu(self.peptide_linear(pep_lstm))
        mhc_linear_out = self.relu(self.mhc_linear(mhc_lstm))

        conc = torch.cat((pep_linear_out, mhc_linear_out), 1)
        out = self.relu(self.out_linear(conc))
        return out
