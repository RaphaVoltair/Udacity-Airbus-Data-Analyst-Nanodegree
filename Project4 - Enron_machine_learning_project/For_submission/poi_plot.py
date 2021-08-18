#!/usr/bin/python

import matplotlib.pyplot as plt

import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


def scatterplot(dataset, var1, var2):
    """
    Creates and shows a scatterplot given a dataset and two features.
    """
    features_name = [str(var1), str(var2)]
    features = [var1, var2]
    data = featureFormat(dataset, features)

    for point in data:
        var1 = point[0]
        var2 = point[1]
        plt.scatter(var1, var2)

    plt.xlabel(features_name[0])
    plt.ylabel(features_name[1])
    plt.show()

    return
