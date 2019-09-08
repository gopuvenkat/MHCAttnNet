import os
import math

import torch
import torchtext

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 3
batch_size = 8
vectors_path = "../data/3-gram-vectors.txt"
cache_path = "../data/"
base_path = "../data/small_classI/"
train_file = "train.csv"
val_file = "val.csv"
test_file = "test.csv"

EMBED_DIM = 100 # change only after re-training the vectors in the new space
PEPTIDE_LENGTH = 120
MHC_AMINO_ACID_LENGTH = 120
BiLSTM_HIDDEN_SIZE = 64
BiLSTM_NUM_LAYERS = 1
LINEAR1_OUT = 64
LINEAR2_OUT = 2
