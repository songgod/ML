from numpy import *
from os import listdir
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


def img2vector(filename):
    returnvector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        linstr = fr.readline()
        for j in range(32):
            returnvector[0,32 * i + j] = int(linstr[j])
    return returnvector


def handwritingclasstest():
    hwlabels = []
    trainingfilelist = listdir('trainingDigits')
    m = len(trainingfilelist)
    trainingmat=zeros((m,1024))
    for i in range(m):
        filenamestr = trainingfilelist[i]
        filestr = filenamestr.split('.')[0]
        classnumstr = int(filestr.split('_')[0])
        hwlabels.append(classnumstr)
        trainingmat[i, :] = img2vector('trainingDigits/%s' % filenamestr)
    testfilelist = listdir('testDigits')
    mtest = len(testfilelist)
    errorcount = 0
    for i in range(mtest):
        filenamestr = testfilelist[i]
        filestr = filenamestr.split('.')[0]
        classnumstr = int(filestr.split('_')[0])
        vectorundertest = img2vector('testDigits/%s' % filenamestr)
        classfileresult = classify0(vectorundertest, trainingmat, hwlabels, 3)
        print("the classfier came back with: %d, the real answer is: %d", classfileresult, classnumstr)
        if classnumstr != classfileresult:
            errorcount += 1
    print("\nthe total number of errors is: %d", errorcount)
    print("\nthe error rate is: %f", errorcount/float(mtest))