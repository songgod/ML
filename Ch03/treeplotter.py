import matplotlib.pyplot as plt
decisionnode = dict(boxstyle="sawtooth", fc="0.8")
leafnode = dict(boxstyle="round4", fc="0.8")
arrayargs = dict(arrowstyle="<-")


def creataplot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    creataplot.ax1 = plt.subplot(111, frameon=False)
    plotnode(U'决策节点', (0.5, 0.1), (0.1, 0.5), decisionnode)
    plotnode(U'叶节点', (0.8, 0.1), (0.3,0.8), leafnode)
    plt.show()


def plotnode(nodetext, centerpt, parentpt, nodetype):
    creataplot.ax1.annotate(nodetext,
                            xy=parentpt,
                            xycoords='axes fraction',
                            xytext=centerpt,
                            textcoords='axes fraction',
                            va='center', ha='center',
                            bbox=nodetype,
                            arrowprops=arrayargs)


def getnumleafs(mytree):
    numleafs = 0
    firststr = list(mytree.keys())[0]
    seconddict = dict(mytree[firststr])
    for key in seconddict.keys():
        if type(seconddict[key]).__name__ == 'dict':
            numleafs += getnumleafs(seconddict[key])
        else:
            numleafs += 1
    return numleafs


def gettreedepth(mytree):
    maxdepth = 0
    firststr = list(mytree.keys())[0]
    seconddict = dict(mytree[firststr])
    for key in seconddict.keys():
        if type(seconddict[key]).__name__ == 'dict':
            thisdepth = 1 + gettreedepth(seconddict[key])
        else:
            thisdepth = 1
        if thisdepth > maxdepth:
            maxdepth = thisdepth
    return maxdepth


def retrievetree(i):
    listoftrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listoftrees[i]


def plotmidtext(cntrpt, parentpt, txtstring):
    xmid = (parentpt[0] + cntrpt[0]) / 2.0
    ymid = (parentpt[1] + cntrpt[1]) / 2.0
    creataplot.ax1.text(xmid,ymid, txtstring)


def classfy(mytree, featlabels, testvec):
    firststr = list(mytree.keys())[0]
    seconddict = dict(mytree[firststr])
    featindex = featlabels.index(firststr)
    for key in seconddict.keys():
        if testvec[featindex] == key:
            if type(seconddict[key]).__name__ == 'dict':
                classlabel = classfy(seconddict[key], featlabels, testvec)
            else:
                classlabel = seconddict[key]
    return classlabel