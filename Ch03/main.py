import tree
dataset, labels = tree.createdataset()
shannon = tree.calcshannonent(dataset)
print(shannon)
res = tree.splitdataset(dataset, 1, 1)
print(res)
res = tree.choosebestfeaturetosplit(dataset)
print(res)