import os
import math

import torch
import torchtext

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n = 1
epochs = 5
batch_size = 32
model_name = "../data/models/v1-classI.pt"
vectors_path = "../data/1-gram-vectors.txt"
cache_path = "../data/"
base_path = "../data/classI/"
train_file = "train.csv"
val_file = "val.csv"
test_file = "test.csv"

PEPTIDE_LENGTH = 45
MHC_AMINO_ACID_LENGTH = 350
EMBED_DIM = 100 # change only after re-training the vectors in the new space

OUTPUT_DIM1 = 32
PEP_KERNEL_CONV1 = 1
MHC_KERNEL_CONV1 = 1

OUTPUT_DIM2_1 = 32
KERNEL_CONV2_1 = (3, 5)
STRIDE_CONV2_1 = (1, 2)

OUTPUT_DIM2_2 = 32
KERNEL_CONV2_2 = (3, 5)
STRIDE_CONV2_2 = (1, 2)

OUTPUT_DIM2_3 = 32
KERNEL_CONV2_3 = (3, 5)
STRIDE_CONV2_3 = (1, 2)

OUTPUT_DIM2_4 = 64
KERNEL_CONV2_4 = (3, 5)
STRIDE_CONV2_4 = (1, 2)

OUTPUT_DIM2_5 = 64
KERNEL_CONV2_5 = (3, 5)
STRIDE_CONV2_5 = (1, 2)

OUTPUT_DIM2_6 = 64
KERNEL_CONV2_6 = (3, 5)
STRIDE_CONV2_6 = (1, 2)

OUTPUT_DIM3_1 = 32
MHC_KERNEL_CONV3_1 = 5
MHC_STRIDE_CONV3_1 = 2

OUTPUT_DIM3_2 = 32
MHC_KERNEL_CONV3_2 = 5
MHC_STRIDE_CONV3_2 = 2

LINEAR_OUT = 2