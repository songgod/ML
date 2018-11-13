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
               [0, 1, 'no'],
               [0, 0, 'maybe']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def splitdataset(dataset, axis, value):
    retdataset = []
    for featvec in dataset:
        if featvec[axis] == value:
            reducedfeatvec = featvec[:axis]
            reducedfeatvec.extend(featvec[axis+1:])
            retdataset.append(reducedfeatvec)
    return retdataset


def choosebestfeaturetosplit(dataset):
    numberfeatures = len(dataset[0])-1
    baseentropy = calcshannonent(dataset)  # 所有数据熵
    baseinfogain = 0.0
    bestfeature = -1
    for i in range(numberfeatures):
        featurelist = [example[i] for example in dataset]
        uniquevals = set(featurelist)
        newentropy = 0.0
        for value in uniquevals:
            subdataset = splitdataset(dataset, i, value)
            prob = len(subdataset)/float(len(dataset))
            newentropy += prob * calcshannonent(subdataset)
        infogain = baseentropy - newentropy
        if infogain > baseinfogain:
            baseinfogain = infogain
            bestfeature = i
    return bestfeature
