import kNN
import numpy as np
datingDataMat,datingLabels=kNN.file2matrix('datingTestSet2.txt')
normalMat,ranges,minVals=kNN.autoNormal(datingDataMat)
import matplotlib
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(normalMat[:,0],normalMat[:,1],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
plt.show()