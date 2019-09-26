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

        self.peptide_embedding = peptide_embedding
        self.mhc_embedding = mhc_embedding

        self.pep_conv1 = nn.Conv1d(config.EMBED_DIM, config.OUTPUT_DIM1, kernel_size=config.PEP_KERNEL_CONV1)
        self.mhc_conv1 = nn.Conv1d(config.EMBED_DIM, config.OUTPUT_DIM1, kernel_size=config.MHC_KERNEL_CONV1)

        self.conv2_1 = nn.Conv2d(config.OUTPUT_DIM1, config.OUTPUT_DIM2_1, kernel_size=config.KERNEL_CONV2_1,
                                 stride=config.STRIDE_CONV2_1)
        self.conv2_2 = nn.Conv2d(config.OUTPUT_DIM2_1, config.OUTPUT_DIM2_2, kernel_size=config.KERNEL_CONV2_2,
                                 stride=config.STRIDE_CONV2_2)
        self.conv2_3 = nn.Conv2d(config.OUTPUT_DIM2_2, config.OUTPUT_DIM2_3, kernel_size=config.KERNEL_CONV2_3,
                                 stride=config.STRIDE_CONV2_3)
        self.conv2_4 = nn.Conv2d(config.OUTPUT_DIM2_3, config.OUTPUT_DIM2_4, kernel_size=config.KERNEL_CONV2_4,
                                 stride=config.STRIDE_CONV2_4)
        self.conv2_5 = nn.Conv2d(config.OUTPUT_DIM2_4, config.OUTPUT_DIM2_5, kernel_size=config.KERNEL_CONV2_5,
                                 stride=config.STRIDE_CONV2_5)
        self.conv2_6 = nn.Conv2d(config.OUTPUT_DIM2_5, config.OUTPUT_DIM2_6, kernel_size=config.KERNEL_CONV2_6,
                                 stride=config.STRIDE_CONV2_6)

        self.mhc_spatial_conv3_1 = nn.Conv1d(config.EMBED_DIM, config.OUTPUT_DIM3_1, kernel_size=config.MHC_KERNEL_CONV3_1, stride=config.MHC_STRIDE_CONV3_1)
        self.mhc_spatial_conv3_2 = nn.Conv1d(config.OUTPUT_DIM3_1, config.OUTPUT_DIM3_2,
                                             kernel_size=config.MHC_KERNEL_CONV3_2, stride=config.MHC_STRIDE_CONV3_2)

        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.5)

        self.out_linear = nn.Linear(6944, config.LINEAR_OUT) # input shape = shape(concatenate(mhc_spatial, linear(conv2)))

    def forward(self, peptide, mhc):
        pep_emb = self.peptide_embedding(peptide)        
        mhc_emb = self.mhc_embedding(mhc)
        # sen_emb = [batch_size, seq_len, emb_dim]

        pep_emb = pep_emb.transpose(1, 2)
        mhc_emb = mhc_emb.transpose(1, 2)

        mhc_spatial_out = self.relu(self.mhc_spatial_conv3_1(mhc_emb))
        mhc_spatial_out = self.relu(self.mhc_spatial_conv3_2(mhc_spatial_out))
        mhc_spatial_out = mhc_spatial_out.view(config.batch_size, -1)

        pep_conv_output = self.relu(self.pep_conv1(pep_emb))
        mhc_conv_output = self.relu(self.mhc_conv1(mhc_emb))
        pep_conv_output = pep_conv_output.transpose(1, 2)
        mhc_conv_output = mhc_conv_output.transpose(1, 2)

        pep_conv_output = pep_conv_output.unsqueeze(1)
        mhc_conv_output = mhc_conv_output.unsqueeze(2)

        image_out = mhc_conv_output + pep_conv_output

        image_out = image_out.transpose(1, 3)
        conv2_out = self.relu(self.conv2_1(image_out))
        conv2_out = self.relu(self.conv2_2(conv2_out))
        conv2_out = self.relu(self.conv2_3(conv2_out))
        conv2_out = self.relu(self.conv2_4(conv2_out))
        conv2_out = self.relu(self.conv2_5(conv2_out))
        conv2_out = self.relu(self.conv2_6(conv2_out))
        conv2_out = conv2_out.view(config.batch_size, -1)

        conc = torch.cat((conv2_out, mhc_spatial_out), dim=1)
        out = self.relu(self.out_linear(conc))
        return out
