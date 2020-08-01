# MHCAttnNet
This is the official implementation of [MHCAttnNet: predicting MHC-peptide bindings for MHC alleles
classes I and II using an attention-based deep neural model](https://academic.oup.com/bioinformatics/article/36/Supplement_1/i399/5870494).


## Architecture
MHCAttnNet uses a Bi-directional Long Short Term Memory (Bi-LSTM) styled encoder to deal with variable-length peptide sequences. This permits the model to handle a large variety of peptides, and hence makes it more general. MHCAttnNet is the first model that is capable of working with both class I and II alleles. This has been made possible with the help of the attention mechanism used in the neural network. The attention mechanism is used to identify relevant subsequences responsible for determining the binding affinity and thereby increase the weights of these relevant subsequences. The model learns to focus on important areas of an amino acid sequence, making it more targeted and informative. 

<p align="center">
  <img src="https://github.com/gopuvenkat/MHCAttnNet/blob/master/Architecture.png" />
</p>

## Environment Setup

```
$ conda env create -f environment.yml
```

## Dataset
The dataset used for the experiments is maintained at [figshare](https://figshare.com/articles/dataset_zip/11770902).

## Pre-process
```
$ (pytorch_p36) python src/preprocess.py -f [INPUT_FILE_NAME] -o [OUTPUT_FILE_NAME]
```

`INPUT_FILE_NAME` - The input .csv file having same columns as [IEDB](https://www.iedb.org/).

`OUTPUT_FILE_NAME` - The output file in .csv format

The output file needs to split into `train.csv` , `val.csv` and `test.csv`.

## Training
First, one needs to change paths in `src/config.py`, especially paths for vector embeddings and paths to the pre-processed data file.

Once, that is done, run the following command - 

```
$ (pytorch_p36) python src/train.py
```

If you use these resources and methods, please cite the following paper:

```
@article{10.1093/bioinformatics/btaa479,
    author = {Venkatesh, Gopalakrishnan and Grover, Aayush and Srinivasaraghavan, G and Rao, Shrisha},
    title = "{MHCAttnNet: predicting MHC-peptide bindings for MHC alleles classes I and II using an attention-based deep neural model}",
    journal = {Bioinformatics},
    volume = {36},
    number = {Supplement_1},
    pages = {i399-i406},
    year = {2020},
    month = {07},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btaa479},
    url = {https://doi.org/10.1093/bioinformatics/btaa479},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/36/Supplement\_1/i399/33488968/btaa479.pdf},
}
```
