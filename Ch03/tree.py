from math import log


def calcshannonent(dataset):
    numenties = len(dataset)
    labelcounts = {}
    for featvec in dataset:
        currentlabel = featvec[-1]
        if currentlabel not in labelcounts.keys():
            labelcounts[currentlabel] = 0
        labelcounts[currentlabel] += 1
    shannonent = 0.0
    for key in labelcounts:
        prob = float(labelcounts[key]) / numenties
        shannonent -= prob * log(prob, 2)
    return shannonent


def createdataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels
