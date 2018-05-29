import os

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from definitions import RESOURCES_DIR

pathToRedWinesDataset = os.path.join(RESOURCES_DIR, "winequality-red.csv")
pathToWhiteWinesDataset = os.path.join(RESOURCES_DIR, "winequality-white.csv")

classesMapping = {0: "bad", 1: "average", 2: "good"}


def mapLabels(labels):
    mappedLabels = labels.copy()
    mappedLabels = mappedLabels.replace(range(-1, 5), 0)
    mappedLabels = mappedLabels.replace(range(5, 7), 1)
    mappedLabels = mappedLabels.replace(range(7, 11), 2)
    return mappedLabels


def main():
    print("\nRed wines:")
    performClasification(pathToRedWinesDataset)
    print("\nWhite wines:")
    performClasification(pathToWhiteWinesDataset)


def performClasification(pathToDataSet):
    data = pd.read_csv(pathToDataSet, sep=";")
    labels = data.quality
    mappedLabels = mapLabels(labels)
    data = data.drop("quality", axis=1)
    scaledData = preprocessing.scale(data)
    trainData, testData, trainLabels, testLabels = train_test_split(scaledData, mappedLabels, test_size=0.3)

    criterions = ["gini", "entropy"]
    print("Decision tree:")
    for criterion in criterions:
        clf = DecisionTreeClassifier(criterion=criterion)
        classifyData(clf, testData, testLabels, trainData, trainLabels, clf.criterion)
    kernels = ["rbf", "linear", "poly", "sigmoid"]
    print("SVM:")
    for kernel in kernels:
        clf = SVC(kernel=kernel, random_state=0)
        classifyData(clf, testData, testLabels, trainData, trainLabels, clf.kernel)


def classifyData(clf, testData, testLabels, trainData, trainLabels, metric):
    clf.fit(trainData, trainLabels)
    confidence = clf.score(testData, testLabels)
    print("For metric: {} confidence score is: {}".format(metric, confidence))


if __name__ == '__main__':
    main()
