#!/usr/bin/python3
"""
File:       torch_cnn.py
Author:     AML Project Group 11
Date:       November 2020

Based on pytorch tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import sys
import csv
from collections import defaultdict, Counter
from itertools import combinations


def overlap(dict, x, y):
    count = 0
    for _, value in dict.items():
        if value[x-1] == value[y-1]:
            count+=1
    return count/len(dict)


def print_overlap(files, dict):
    for comb in combinations([i for i in range(1, len(files) + 1)], 2):
        print("Overlap {0} and {1}: \t {2}".format(files[comb[0]-1],
                                                   files[comb[1]-1],
                                                   overlap(dict, comb[0], comb[1])))


def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


def write_voted_predictions(dict):
    print("Writing predictions:", end=" ")
    with open("predictions_voted.csv", "w") as f:
        f.write("img_name,label\n")
        for key, value in dict.items():
            f.write("{0},{1}\n".format(key, most_frequent(value)))
    print("Done")


def main(argv):
    files = ["predictions_resnet_2.csv",
             "predictions_wideResNet_3.csv",
             "predictions_mobilenet_2.csv"]

    predictions_dict = defaultdict(list)

    for file_name in files:
        with open(file_name, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            first = True
            for row in csv_reader:
                if first:
                    first = False
                else:
                    predictions_dict[row[0]].append(row[1])

    print_overlap(files, predictions_dict)
    write_voted_predictions(predictions_dict)


if __name__ == "__main__":
    main(sys.argv)