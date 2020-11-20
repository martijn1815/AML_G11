#!/usr/bin/python3
"""
File:       main.py
Author:     AML Project Group 11
Date:       November 2020
"""

import sys
import pandas as pd


def main(argv):
    train_labels = pd.read_csv("train_labels.csv")
    print(train_labels.head())


if __name__ == "__main__":
    main(sys.argv)
