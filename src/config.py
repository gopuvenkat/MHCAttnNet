import os
import math

import torch
import torchtext

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 3
batch_size = 32
model_name = "../data/models/v1-classI.pt"
pep_n = 1
mhc_n = 3
pep_vectors_path = "../data/1-gram-vectors.txt" # set based on pep_n
mhc_vectors_path = "../data/3-gram-vectors.txt" # set based on mhc_n
cache_path = "../data/"
base_path = "../data/classI/"
train_file = "train.csv"
val_file = "val.csv"
test_file = "test.csv"

EMBED_DIM = 100 # change only after re-training the vectors in the new space
PEPTIDE_LENGTH = 45 # set based on pep_n
MHC_AMINO_ACID_LENGTH = 120 # set based on mhc_n
BiLSTM_HIDDEN_SIZE = 64
BiLSTM_PEPTIDE_NUM_LAYERS = 1
BiLSTM_MHC_NUM_LAYERS = 1
LINEAR1_OUT = 64
LINEAR2_OUT = 2
CONTEXT_DIM = 16