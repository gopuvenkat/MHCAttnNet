import os
import math

import torch
import torchtext

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBED_DIM = 100 # change only after re-training the vectors in the new space
PEPTIDE_LENGTH = 10
MHC_AMINO_ACID_LENGTH = 150
BiLSTM_HIDDEN_SIZE = 64
LINEAR1_OUT = 64
LINEAR2_OUT = 1
