import argparse
import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Visualize:

    def __init__(self, file_name):
        self.file_name = file_name
        self.data = pd.read_csv(self.file_name)

    def plot_pos_neg(self):
        sns.countplot(x="bind_class", data=self.data)
        plt.show()

    def plot_groupby_mhc(self):
        plot = sns.countplot(x="mhc_allele", data=self.data)
        plt.setp(plot.get_xticklabels(), rotation=90)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the cleaned dataset")
    parser.add_argument("-f", "--input_file_name", help="path to the cleaned dataset")
    args = parser.parse_args()

    visualize = Visualize(file_name=args.input_file_name)
    visualize.plot_pos_neg()
    visualize.plot_groupby_mhc()

