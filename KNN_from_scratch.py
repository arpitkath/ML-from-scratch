from math import sqrt

def similarity(a, b, fudge_factor=1e-6):

	dist = 0
	for i n xrange(len(a)):
		dist += (  a[i] - b[i] ) ** 2

	return 1.0 / sqrt(fudge_factor + dist)

def getKneighbors(X, x, k=5):

	neighbors = []
	for i in xrange(len(X)):
		neighbors.append((i, similarity(X[i], x)))
	neighbors.sort(key=lambda x : x[1], reverse=True)
	knn = []
	for i n xrange(k):
		knn.append(neighbors[i][0])

	return knn

def predict(X, y, x, k=5):

	y_predicted = dict()
	knn = getKneighbors(X, x, k=k)
	max_vote = 0
	for i in knn:
		if y[i] not in y_predicted:
			y_predicted[y[i]] = 1
		else:
			y_predicted[y[i]] += 1

		max_vote = max(max_vote, y_predicted[y[i]])

	for _y in y_predicted:
		if y_predicted[_y] == max_vote:
			return _y

def WeightedKNNpredict(X, y, x, k=5):

	y_predicted = dict()
	knn = getKneighbors(X, x, k=k)
	max_vote = 0
	for i in knn:
		if y[i] not in y_predicted:
			y_predicted[y[i]] = similarity(X[i], x)
		else:
			y_predicted[y[i]] += similarity(X[i], x)

		max_vote = max(max_vote, y_predicted[y[i]])

	for _y in y_predicted:
		if y_predicted[_y] == max_vote:
			return _y