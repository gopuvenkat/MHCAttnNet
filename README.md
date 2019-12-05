# MHCAttnNet

## Architecture
MHCAttnNet uses a gated Recurrent Neural Network (RNN) styled encoder to deal with variable-length peptide sequences. This permits the model to handle a large variety of peptides, and hence makes it more general. MHCAttnNet is the first model that is capable of working with both class I and II alleles. This has been made possible with the help of the attention mechanism used in the neural network. The attention mechanism is used to identify relevant subsequences responsible for determining the binding affinity and thereby increase the weights of these relevant subsequences. The model learns to focus on important areas of an amino acid sequence, making it more targeted and informative. 

![Architecture Diagram](https://github.com/gopuvenkat/MHCAttnNet/blob/master/Architecture.png)

## Environment Setup

## 
