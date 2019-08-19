import argparse
import math

import numpy as np
import pandas as pd

class Preprocess:
    def __init__(self, file_name, header_index, relevant_class, relevant_orgs, is_quan):
        self.file_name = file_name
        # load the .csv file
        self.data = pd.read_csv(self.file_name, header=header_index)
        relevant_cols = ['Description', 'Name', 'Qualitative Measure', 'Allele Name', 'MHC allele class', 'Measurement Inequality', 'Quantitative measurement', "Assay Group"]
        self.data = self.data[relevant_cols]    # retain only relevant columns
        self.data.columns = ["peptide", "host", "qualitative", "mhc_allele", "mhc_class", "inequality", "quantitative", "assay_group"]
        self.data = self.data[self.data["host"].isin(relevant_orgs)]
        self.data = self.data[self.data["mhc_class"].isin(relevant_class)]
        # self.data = self.data[self.data["assay_group"].isin(["dissociation constant KD (~IC50)", "half maximal inhibitory concentration (IC50)", "dissociation constant KD (~EC50)"])]

        if is_quan:
            df1 = self.data.inequality == "="
            df2 = self.data.inequality.isnull()
            self.data = self.data[df1 | df2]
            self.data = self.data[self.data["quantitative"].notnull()]
            self.data = self.data[self.data.quantitative <= 50000]
            self.data = self.data[self.data.quantitative > 0]
            self.data['binding_probability'] = self.data.quantitative.apply(lambda x: 1-math.log(x, 50000))
            # only for class I
            # self.data.loc[(self.data.quantitative) < 500, "bind_class"] = "Positive"
            # self.data.loc[(self.data.quantitative) >= 500, "bind_class"] = "Negative"
            # only for class II
            # self.data.loc[(self.data.quantitative) < 1000, "bind_class"] = "Positive"
            # self.data.loc[(self.data.quantitative) >= 1000, "bind_class"] = "Negative"

        else:
            self.data.loc[(self.data.qualitative == "Positive") | (self.data.qualitative == "Positive-High") | (self.data.qualitative == "Positive-Intermediate"), 'bind_class'] = 'Positive'
            self.data.loc[(self.data.qualitative == "Negative") | (self.data.qualitative == "Positive-Low"), 'bind_class'] = 'Negative'

        self.data['valid'] = self.data['peptide'].apply(lambda x: (str(x).isupper() and str(x).isalpha()))
        self.data = self.data[self.data.valid == True]

        self.data = self.data[self.data["peptide"].apply(lambda x: "+" not in x)]
        self.data = self.data[self.data["mhc_allele"].apply(lambda x: "/" not in x)]
        self.data = self.data[self.data["mhc_allele"].apply(lambda x: "mutant" not in x)]
        self.data = self.data[self.data["mhc_allele"].apply(lambda x: "class" not in x)]
        # only for class I
        # self.data = self.data[self.data["mhc_allele"].apply(lambda x: (("HLA-A*" in x) or ("HLA-B*" in x) or ("HLA-C*" in x)))]
        # self.data = self.data[self.data["mhc_allele"].apply(lambda x: "HLA-A*30:14L" not in x)]
        # only for class II
        # self.data = self.data[self.data["mhc_allele"].apply(lambda x: (("H2-IA" in x) or ("DRB1*" in x) or ("DRB5*" in x) or ("DRB4*" in x) or ("DRB3*" in x)))]

        print(self.data.mhc_allele.unique())
        print(self.data.bind_class.value_counts())


    def save(self, file_name):
        self.data.to_csv(file_name, index=None, header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess the IEDB (mhc_ligand_full.csv) dataset')
    parser.add_argument('-f', '--input_file_name', help='path to the input dataset')
    parser.add_argument('-o', '--output_file_name', help='path to the processed dataset')
    parser.add_argument('--header_index', type=int, default=1)
    parser.add_argument('--relevant_class', nargs="+", type=str, default=["I", "II"])
    parser.add_argument('--relevant_orgs', default=['human (Homo sapiens)', 'Homo sapiens'])
    parser.add_argument('--is_quan', default=False)
    parser.add_argument("--save", default=False)
    args = parser.parse_args()

    preprocess = Preprocess(file_name=args.input_file_name,
                            header_index=args.header_index, relevant_class=args.relevant_class, relevant_orgs=args.relevant_orgs, is_quan=args.is_quan)
    if args.save:
        preprocess.save(args.output_file_name)
