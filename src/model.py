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

    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.attention = nn.Parameter(torch.FloatTensor(attention_size, 1))
        torch.nn.init.xavier_normal(self.attention)

    def forward(self, x_in):
        attention_score = torch.matmul(x_in, self.attention).squeeze()
        attention_score = F.softmax(attention_score, dim=1).view(x_in.shape[0], x_in.shape[1], 1)
        scored_x = x_in * attention_score
        condensed_x = torch.sum(scored_x, dim=1)
        return condensed_x

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
        # self.pep_attention = Attention(2*self.hidden_size)
        # self.mhc_attention = Attention(2*self.hidden_size)
        self.peptide_linear = nn.Linear(2*self.hidden_size, config.LINEAR1_OUT)
        self.mhc_linear = nn.Linear(2*self.hidden_size, config.LINEAR1_OUT)
        self.out_linear = nn.Linear(config.LINEAR1_OUT*2, config.LINEAR2_OUT)

    def attention(self, rnn_out, state):
        merged_state = torch.cat([s for s in state], dim=1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)

        weights = torch.bmm(rnn_out, merged_state)
        weigths = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)

        out = torch.bmm(torch.transpose(rnn_out, 1, 2), weigths).squeeze(2)
        return out

    def forward(self, peptide, mhc):
        pep_emb = self.peptide_embedding(peptide)        
        mhc_emb = self.mhc_embedding(mhc)
        # sen_emb = [batch_size, seq_len, emb_dim]

        pep_lstm_output, (pep_last_hidden_state, pep_last_cell_state) = self.peptide_lstm(pep_emb)
        mhc_lstm_output, (mhc_last_hidden_state, mhc_last_cell_state) = self.mhc_lstm(mhc_emb)
        # sen_lstm_output = [batch_size, seq_len, 2*hidden_dim]            -> 2 : bidirectional
        # sen_last_hidden_state = [2*num_layers, batch_size, hidden_dim]   -> 2 : bidirectional

        # pep_attn_linear_inp = self.pep_attention(pep_lstm_output)
        # mhc_attn_linear_inp = self.mhc_attention(mhc_lstm_output)

        pep_attn_linear_inp = self.attention(pep_lstm_output, pep_last_hidden_state)
        mhc_attn_linear_inp = self.attention(mhc_lstm_output, mhc_last_hidden_state)

        pep_linear_out = self.relu(self.peptide_linear(pep_attn_linear_inp))
        mhc_linear_out = self.relu(self.mhc_linear(mhc_attn_linear_inp))
        # sen_linear_out = [batch_size, LINEAR1_OUT]

        conc = torch.cat((pep_linear_out, mhc_linear_out), dim=1)
        # conc = [batch_size, 2*LINEAR1_OUT]
        out = self.relu(self.out_linear(conc))
        # out = [batch_size, LINEAR2_OUT]
        return out
