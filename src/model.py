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

        self.peptide_embedding = peptide_embedding
        self.mhc_embedding = mhc_embedding
        self.relu = nn.ReLU()

        self.peptide_lstm = nn.LSTM(config.EMBED_DIM, self.hidden_size, batch_first=True, bidirectional=True)
        self.mhc_lstm = nn.LSTM(config.EMBED_DIM, self.hidden_size, batch_first=True, bidirectional=True)
        self.peptide_linear = nn.Linear(self.hidden_size*2, config.LINEAR1_OUT)
        self.mhc_linear = nn.Linear(self.hidden_size*2, config.LINEAR1_OUT)
        self.out_linear = nn.Linear(config.LINEAR1_OUT*2, config.LINEAR2_OUT)

    def forward(self, peptide, mhc):
        pep_emb = self.peptide_embedding(peptide)        
        mhc_emb = self.mhc_embedding(mhc)
        # sen_emb = [batch_size, seq_len, emb_dim]

        pep_lstm_output, (pep_last_hidden_state, pep_last_cell_state) = self.peptide_lstm(pep_emb)
        mhc_lstm_output, (mhc_last_hidden_state, mhc_last_cell_state) = self.mhc_lstm(mhc_emb)
        # sen_last_hidden_state = [2, batch_size, hidden_dim]   -> 2 : bidirectional
        
        pep_last_hidden_state = pep_last_hidden_state.transpose(0, 1).contiguous().view(config.batch_size, -1)
        mhc_last_hidden_state = mhc_last_hidden_state.transpose(0, 1).contiguous().view(config.batch_size, -1)
        # sen_last_hidden_state = [batch_size, 2*hidden_dim]    -> 2 : bidirectional

        pep_linear_out = self.relu(self.peptide_linear(pep_last_hidden_state))
        mhc_linear_out = self.relu(self.mhc_linear(mhc_last_hidden_state))
        # sen_linear_out = [batch_size, LINEAR1_OUT]

        conc = torch.cat((pep_linear_out, mhc_linear_out), dim=1)
        # conc = [batch_size, 2*LINEAR1_OUT]
        out = self.relu(self.out_linear(conc))
        # out = [batch_size, LINEAR2_OUT]
        return out
