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
import operator


def load_files(files, os_system="MacOs"):
    predictions_dict = defaultdict(list)
    if os_system == "MacOs":
        dir_os = "predictions/"
    else:
        dir_os = "predictions\\"

    for file_name in files:
        with open("{0}{1}.csv".format(dir_os, file_name), mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            first = True
            for row in csv_reader:
                if first:
                    first = False
                elif len(row) == 2:
                    predictions_dict[row[0]].append(row[1])
                else:
                    predictions_dict[row[0]].append(row[1:])

    return predictions_dict


def get_top_k(files):
    k_list = []

    for f in files:
        x = f.split('.')[0].split('_')[-1]
        if len(x) > 1 and x[0] == "t":
            k_list.append(int(x[-1]))
        else:
            k_list.append(1)

    return max(k_list)


def overlap(dict, x, y):
    count = 0
    for _, value in dict.items():
        if value[x-1] == value[y-1]:
            count += 1
    return count/len(dict)


def overlap2(dict, x, y):
    count = 0
    for _, value in dict.items():
        if value[x-1][0] == value[y-1][0]:
            count += 1
    return count/len(dict)


def print_overlap(files, dict, top1=True):
    for comb in combinations([i for i in range(1, len(files) + 1)], 2):
        if top1:
            print("Overlap {0} and {1}: \t {2}".format(files[comb[0]-1],
                                                       files[comb[1]-1],
                                                       overlap(dict, comb[0], comb[1])))
        else:
            print("Overlap {0} and {1}: \t {2}".format(files[comb[0]-1],
                                                       files[comb[1]-1],
                                                       overlap2(dict, comb[0], comb[1])))


def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


def write_voted_predictions(dict, os_system="MacOs"):
    if os_system == "MacOs":
        dir_os = "predictions/"
    else:
        dir_os = "predictions\\"

    print("Writing predictions:", end=" ")
    with open("{0}predictions_voted.csv".format(dir_os), "w") as f:
        f.write("img_name,label\n")
        for key, value in dict.items():
            f.write("{0},{1}\n".format(key, most_frequent(value)))
    print("Done")


def rank_predictions(predictions_dict, top_k):
    ranked = []

    print("Ranking predictions:", end=" ")
    for key, value in predictions_dict.items():
        score_dict = defaultdict(int)
        for v in value:
            for i, pred in enumerate(v):
                score_dict[pred] += top_k-i
        prediction = max(score_dict.items(), key=operator.itemgetter(1))[0]
        #print(prediction, score_dict)
        ranked.append([key, prediction])
    print("Done")

    return ranked


def rank_weighted_predictions(predictions_dict, weights, top_k):
    ranked = []

    print("Ranking predictions:", end=" ")
    for key, value in predictions_dict.items():
        score_dict = defaultdict(int)
        for i, v in enumerate(value):
            for j, pred in enumerate(v):
                score_dict[pred] += (top_k - j) * weights[i]
        prediction = max(score_dict.items(), key=operator.itemgetter(1))[0]
        #print(prediction, score_dict)
        ranked.append([key, prediction])
    print("Done")

    return ranked


def write_rank_predictions(predictions_dict, top_k, weights=None, os_system="MacOs"):
    if os_system == "MacOs":
        dir_os = "predictions/"
    else:
        dir_os = "predictions\\"

    if weights:
        file_name = "predictions_ranked_weighted"
        ranked = rank_weighted_predictions(predictions_dict, weights, top_k)
    else:
        file_name = "predictions_ranked"
        ranked = rank_predictions(predictions_dict, top_k)

    print("Writing predictions:", end=" ")
    with open("{0}{1}.csv".format(dir_os, file_name), "w") as f:
        f.write("img_name,label\n")
        for output in ranked:
            f.write("{0},{1}\n".format(output[0], output[1]))
    print("Done")


def main(argv):
    os_system = "MacOs"
    # For best results (with out weigths) put the files in order of accuracy, from best too worst:
    files = ["predictions_resnet101_org_9_top3",
             "predictions_resnet101_aug_4_top5",
             "predictions_augmented_mobilenet_25_top3",
             "predictions_mobilenet_vertflip_11_top5",
             "predictions_resnet34_freeze_org_19_top5",
             "predictions_squeezenet_7_top3"]
    # Use accuracy as weight:
    weights = [0.64824, 0.64379, 0.63595, 0.60995, 0.48151, 0.48059]

    predictions_dict = load_files(files, os_system=os_system)
    top_k = get_top_k(files)

    if top_k == 1:
        print_overlap(files, predictions_dict)
        write_voted_predictions(predictions_dict, os_system=os_system)
    else:
        print_overlap(files, predictions_dict, top1=False)
        write_rank_predictions(predictions_dict, top_k, weights=weights, os_system=os_system)


if __name__ == "__main__":
    main(sys.argv)