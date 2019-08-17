import argparse
import pandas as pd
import numpy as np

class Preprocess:
    def __init__(self, file_name, header_index, relevant_cols):
        self.file_name = file_name
        self.df = pd.read_csv(file_name, header=header_index)    # load the .csv file
        self.df = self.df[relevant_cols]    # retain only relevant columns


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the IEDB (mhc_ligand_full.csv) dataset')
    parser.add_argument('-f', '--file_name', help='path to the dataset')
    parser.add_argument('--header_index', default=1)
    parser.add_argument('--relevant_cols', default=['Description', 'Name', 'Qualitative Measure', 'Allele Name', 'MHC allele class'])
    parser.add_argument('--relevant_orgs', default=['human (Homo sapiens)', 'Homo sapiens'])
    args = parser.parse_args()

    preprocess = Preprocess(file_name=args.file_name, header_index=args.header_index, relevant_cols=args.relevant_cols)
