# MHCAttnNet

## Architecture
MHCAttnNet uses a gated Recurrent Neural Network (RNN) styled encoder to deal with variable-length peptide sequences. This permits the model to handle a large variety of peptides, and hence makes it more general. MHCAttnNet is the first model that is capable of working with both class I and II alleles. This has been made possible with the help of the attention mechanism used in the neural network. The attention mechanism is used to identify relevant subsequences responsible for determining the binding affinity and thereby increase the weights of these relevant subsequences. The model learns to focus on important areas of an amino acid sequence, making it more targeted and informative. MHCAttnNet can handle 161 different MHC class I alleles and 49 different class II alleles. Even while handling a large variety of MHC alleles, our model outperforms the current state-of-the-art models for predicting binding affinity between peptide and MHC class I alleles while at the same time, is competitive in case of MHC class II alleles while handling a larger number of alleles.

![Architecture Diagram]()

## Environment Setup

## 
