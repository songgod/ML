import numpy as np
import matplotlib.pyplot as plt


def loaddataset():
    datamat = []
    labelmat = []
    fr = open('testset.txt')
    for line in fr.readlines():
        linearr = line.strip().split()
        datamat.append([1.0, float(linearr[0]), float(linearr[1])])
        labelmat.append(int(linearr[2]))
    return datamat, labelmat


def sigmoid(inx):
    return 1.0 / (1 + np.exp(-inx))


def gradascent(datamat, labelmat):
    datamatrix = np.mat(datamat)
    labelmatrix = np.mat(labelmat).transpose()
    m, n = np.shape(datamatrix)
    alpha = 0.001
    maxcycles = 500
    weights = np.ones((n, 1))
    for k in range(maxcycles):
        h = sigmoid(datamatrix*weights)
        error = (labelmatrix-h)
        weights = weights + alpha * datamatrix.transpose() * error
    return weights


def plotbestfit(weights):
    datamat, labelmat = loaddataset()
    dataarr = np.array(datamat)
    n = np.shape(dataarr)[0]
    xcords1 = []
    ycords1 = []
    xcords2 = []
    ycords2 = []
    for i in range(n):
        if int(labelmat[i]) == 1:
            xcords1.append(dataarr[i, 1])
            ycords1.append(dataarr[i, 2])
        else:
            xcords2.append(dataarr[i, 1])
            ycords2.append(dataarr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcords1, ycords1, s=30, c='red', marker='s')
    ax.scatter(xcords2, ycords2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


if __name__ == '__main__':
    datamatv, labelmatv = loaddataset()
    ws = gradascent(datamatv, labelmatv)
    plotbestfit(ws.getA())

