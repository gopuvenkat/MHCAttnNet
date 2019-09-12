import math
import sys
import os
import argparse

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchtext
from torchtext.data import BucketIterator, Field, interleave_keys, RawField
from torchtext.data.dataset import TabularDataset
from torchtext.data.pipeline import Pipeline

import config

n = config.n
def tokenize_amino_acid(seq):
    return list(map("".join, zip(*[iter(seq)]*n)))

class IEDB(TabularDataset):

    def __init__(self, path, format, fields, skip_header=True, **kwargs):
        super(IEDB, self).__init__(path, format, fields, skip_header, **kwargs)

        # keep a raw copy of the sentence for debugging
        # RAW_TEXT_FIELD = RawField()
        # for ex in self.examples:
        #     peptide, mhc_amino_acid = ex.peptide[:], ex.mhc_amino_acid[:]
        #     setattr(ex, "raw_peptide", peptide) 
        #     setattr(ex, "raw_mhc_amino_acid", mhc_amino_acid)
        # self.fields["raw_peptide"] = RAW_TEXT_FIELD
        # self.fields["raw_mhc_amino_acid"] = RAW_TEXT_FIELD
    
    @staticmethod
    def sort_key(ex):
        return interleave_keys(len(ex.peptide), len(ex.mhc_amino_acid))
    
    @classmethod
    def splits(cls, peptide_field, mhc_amino_acid_field, label_field, path=config.base_path, train=config.train_file, validation=config.val_file, test=config.test_file):
        return super(IEDB, cls).splits(
            path=path,
            train=train,
            validation=validation,
            test=test,
            format="csv",
            fields=[
                ("peptide", peptide_field), 
                ("mhc_amino_acid", mhc_amino_acid_field),
                ("bind", label_field)
            ],
            skip_header=True
        )

    @classmethod
    def iters(cls, batch_size=64, device=0, shuffle=True, vectors_path=config.vectors_path, cache_path=config.cache_path):
        cls.PEPTIDE = Field(sequential=True, tokenize=tokenize_amino_acid, batch_first=True, fix_length=config.PEPTIDE_LENGTH)
        cls.MHC_AMINO_ACID = Field(sequential=True, tokenize=tokenize_amino_acid, batch_first=True, fix_length=config.MHC_AMINO_ACID_LENGTH)
        cls.LABEL = Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)

        train, val, test = cls.splits(cls.PEPTIDE, cls.MHC_AMINO_ACID, cls.LABEL)

        vec = torchtext.vocab.Vectors(vectors_path, cache=cache_path)

        cls.PEPTIDE.build_vocab(train, val, vectors=vec)
        cls.MHC_AMINO_ACID.build_vocab(train, val, vectors=vec)

        return BucketIterator.splits((train, val, test), batch_size=batch_size, shuffle=shuffle, repeat=False, device=device)


def get_dataset(device):
    train_loader, val_loader, test_loader = IEDB.iters(batch_size=config.batch_size, device=device, shuffle=True)

    print("Peptide embedding dimension", IEDB.PEPTIDE.vocab.vectors.size())
    peptide_embedding_dim = IEDB.PEPTIDE.vocab.vectors.size()
    peptide_embedding = nn.Embedding(peptide_embedding_dim[0], peptide_embedding_dim[1])
    peptide_embedding.weight = nn.Parameter(IEDB.PEPTIDE.vocab.vectors)
    peptide_embedding.weight.required_grad = True

    print("MHC Amino Acid embedding dimension", IEDB.MHC_AMINO_ACID.vocab.vectors.size())
    mhc_amino_acid_embedding_dim = IEDB.MHC_AMINO_ACID.vocab.vectors.size()
    mhc_amino_acid_embedding = nn.Embedding(mhc_amino_acid_embedding_dim[0], mhc_amino_acid_embedding_dim[1])
    mhc_amino_acid_embedding.weight = nn.Parameter(IEDB.MHC_AMINO_ACID.vocab.vectors)
    mhc_amino_acid_embedding.weight.required_grad = True

    return IEDB, train_loader, val_loader, test_loader, peptide_embedding, mhc_amino_acid_embedding

if __name__ == "__main__":
    device = config.device

    dataset_cls, train_loader, val_loader, test_loader, peptide_embedding, mhc_embedding = get_dataset(device)

    print(next(iter(train_loader)))
