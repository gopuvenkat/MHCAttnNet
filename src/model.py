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


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, context_dim):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.context_dim = context_dim
        self.tanh = nn.Tanh()

        weight = torch.zeros(feature_dim, context_dim)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)
        self.b = nn.Parameter(torch.zeros(step_dim, context_dim))

        u = torch.zeros(context_dim, 1)
        nn.init.kaiming_uniform_(u)
        self.context_vector = nn.Parameter(u)

    def forward(self, x):
        eij = torch.matmul(x, self.weight)
        # eij = [batch_size, seq_len, context_dim]
        eij = self.tanh(torch.add(eij, self.b))
        # eij = [batch_size, seq_len, context_dim]
        v = torch.exp(torch.matmul(eij, self.context_vector))  # dot product
        # v = [batch_size, seq_len, 1]
        v = v / (torch.sum(v, dim=1, keepdim=True))
        # v = [batch_size, seq_len, 1]
        weighted_input = x * v
        # weighted_input = [batch_size, seq_len, 2*hidden_dim]             -> 2 : bidirectional
        s = torch.sum(weighted_input, dim=1)
        # s = [batch_size, 2*hidden_dim]                                   -> 2 : bidirectional
        return s

class MHCAttnNet(nn.Module):

    def __init__(self, peptide_embedding, mhc_embedding):
        super(MHCAttnNet, self).__init__()
        self.hidden_size = config.BiLSTM_HIDDEN_SIZE
        self.peptide_num_layers = config.BiLSTM_PEPTIDE_NUM_LAYERS
        self.mhc_num_layers = config.BiLSTM_MHC_NUM_LAYERS

        self.peptide_embedding = peptide_embedding
        self.mhc_embedding = mhc_embedding
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.peptide_lstm = nn.LSTM(config.EMBED_DIM, self.hidden_size, num_layers=self.peptide_num_layers, batch_first=True, bidirectional=True)
        self.mhc_lstm = nn.LSTM(config.EMBED_DIM, self.hidden_size, num_layers=self.mhc_num_layers, batch_first=True, bidirectional=True)

        self.peptide_attn = Attention(2*self.hidden_size, config.PEPTIDE_LENGTH, config.CONTEXT_DIM)
        self.mhc_attn = Attention(2*self.hidden_size, config.MHC_AMINO_ACID_LENGTH, config.CONTEXT_DIM)

        self.peptide_linear = nn.Linear(2*self.hidden_size, config.LINEAR1_OUT)
        self.mhc_linear = nn.Linear(2*self.hidden_size, config.LINEAR1_OUT)
        self.hidden_linear = nn.Linear(2*config.LINEAR1_OUT, config.LINEAR1_OUT)
        self.out_linear = nn.Linear(config.LINEAR1_OUT, config.LINEAR2_OUT)

    def forward(self, peptide, mhc):
        pep_emb = self.peptide_embedding(peptide)        
        mhc_emb = self.mhc_embedding(mhc)
        # sen_emb = [batch_size, seq_len, emb_dim]

        pep_lstm_output, (pep_last_hidden_state, pep_last_cell_state) = self.peptide_lstm(pep_emb)
        mhc_lstm_output, (mhc_last_hidden_state, mhc_last_cell_state) = self.mhc_lstm(mhc_emb)
        # sen_lstm_output = [batch_size, seq_len, 2*hidden_dim]            -> 2 : bidirectional
        # sen_last_hidden_state = [2*num_layers, batch_size, hidden_dim]   -> 2 : bidirectional

        pep_attn_linear_inp = self.peptide_attn(pep_lstm_output)
        mhc_attn_linear_inp = self.mhc_attn(mhc_lstm_output)
        # sen_attn_linear_inp = [batch_size, 2*hidden_dim]                 -> 2 : bidirectional
        
        pep_linear_out = self.relu(self.peptide_linear(pep_attn_linear_inp))
        mhc_linear_out = self.relu(self.mhc_linear(mhc_attn_linear_inp))
        # sen_linear_out = [batch_size, LINEAR1_OUT]

        conc = torch.cat((pep_linear_out, mhc_linear_out), dim=1)
        # conc = [batch_size, 2*LINEAR1_OUT]
        out = self.relu(self.out_linear(self.hidden_linear(conc)))
        # out = [batch_size, LINEAR2_OUT]
        return out
