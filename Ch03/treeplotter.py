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

