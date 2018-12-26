import os
import random
import numpy as np


def loaddataset(filename):
    datamat = []
    labelmat = []
    fr = open(filename)
    for line in fr.readlines():
        lineattr = line.strip().split('\t')
        datamat.append([float(lineattr[0]), float(lineattr[1])])
        labelmat.append(float(lineattr[2]))
    return datamat, labelmat


def selectjrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipalpha(aj, high, low):
    if aj > high:
        aj = high
    if aj < low:
        aj = low
    return aj


def smosimple(datamatin, labelmatin, c, toler, matiter):
    datamatrix = np.matrix(datamatin)
    labelmatrix = np.matrix(labelmatin).transpose()
    b = 0
    m, n = np.shape(datamatrix)
    alphas = np.mat(np.zeros((m,1)))
    iter = 0



if __name__ == "__main__":
    dataatrr, labelatrr = loaddataset("testset.txt")
    print(labelatrr)
