from numpy import *
import operator


def createdataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    labels = ['a', 'a', 'b', 'b']
    return group, labels


def classify0(inx, dataset, labels, k):
    datasetsize = dataset.shape[0]
    diffmat = tile(inx, (datasetsize, 1))-dataset
    sqdiffmat = diffmat**2
    sqdistance = sqdiffmat.sum(axis=1)
    distance = sqdistance**0.5
    sorteddistindicies = distance.argsort()
    classcount = {}
    for i in range(k):
        voteilabel = labels[sorteddistindicies[i]]
        classcount[voteilabel] = classcount.get(voteilabel, 0)+1
        
    sortedclasscount = sorted(classcount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedclasscount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayolines = fr.readlines()
    numberoflines = len(arrayolines)
    returnmat = zeros((numberoflines, 3))
    classlabelvector = []
    idx = 0
    for line in arrayolines:
        line = line.strip()
        listfromline = line.split('\t')
        returnmat[idx, :] = listfromline[0:3]
        classlabelvector.append(int(listfromline[-1]))
        idx += 1
    return returnmat, classlabelvector


def autonormal(dataset):
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    ranges = maxvals-minvals
    normadataset = zeros(shape(dataset))
    m = dataset.shape[0]
    t = tile(minvals, (m, 1))
    normadataset = dataset-t
    normadataset = normadataset/tile(ranges, (m, 1))
    return normadataset, ranges, minvals


def datingclasstest():
    horatio = 0.10
    datingdata, datinglabels = file2matrix('datingtestset2.txt')
    normmat, ranges, minvals = autonormal(datingdata)
    m = normmat.shape[0]
    numtestvecs = int(m*horatio)
    errorcount = 0.0  # type: float
    for i in range(numtestvecs):
        classifierresult = classify0(normmat[i, :], normmat[numtestvecs:m, :], datinglabels[numtestvecs:m], 10)
        print("the classifier came back with:%d,the real answer is:%d", classifierresult, datinglabels[i])
        if classifierresult != datinglabels[i]:
            errorcount += 1.0
    print("the total eroor rate is:%f", errorcount/float(numtestvecs))