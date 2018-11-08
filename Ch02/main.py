import knn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

datingdatamat, datinglabels = knn.file2matrix('datingtestset2.txt')
normalmat, ranges, minvals = knn.autonormal(datingdatamat)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(normalmat[:, 0], normalmat[:, 1], 15.0*np.array(datinglabels), 15.0*np.array(datinglabels))
plt.show()