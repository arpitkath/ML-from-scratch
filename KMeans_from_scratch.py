import random

MAX_ITERS = 1000

def kmeans_pp(X, n):
	
	assert n < len(X)
	centroids = [X[0]]
	while len(centroids) < n:
		dist = scipy.array([min([scipy.inner(centroid - x, centroid - x) for centroid in centroids]) for x in X])
		probs = dist / dist.sum()
		cumm_probs = probs.cumsum()
		thres = random.random()
		for i, prob in enumerate(cumm_probs):
			if prob > thres:
				centroids.append(X[i])
				break
	return centroids

def initialize_centroids(X, n):

	assert n <= len(X)
	random_index = range(len(X))
	random_index = random.shuffle(random_index)[ : n]
	centroids = []
	for i in random_index:
		centroids.append(X[i])

	return centroids

def converged(current_centroids, previous_centroids, n_iters):

	assert len(current_centroids) == len(previous_centroids)
	return n_iters > MAX_ITERS or list(current_centroids) == list(previous_centroids)

def distance(x, y):

	distance = 0
	for i in xrange(len(x)):
		distance += (x[i] - y[i]) ** 2

	return distance

def centroid_of(x, centroids):

	assert len(centroids) > 0
	min_dist = distance(x, centroids[0])
	centroid = 0
	for i in xrange(1, len(centroids)):
		curr_dist = distance(x, centroids[i])
		if curr_dist < min_dist:
			curr_dist = min_dist
			centroid = i

	return centroid

def rearrange_centroids(X, centroids):

	centroid = dict()
	for i in xrange(len(centroids)):
		centroid[i] = []

	for x in X:
		c = centroid_of(x, centroids)
		centroid[c].append(x)

	new_centroids = []
	for c in centroid:
		new_centroid = [0 for _ in xrange(len(X[0]))]
		for x in centroid[c]:
			for i in xrange(len(x)):
				new_centroid[i] += x[i]
		for i in xrange(len(new_centroid)):
			new_centroid[i] /= float(len(centroid[c]))
		new_centroids.append(new_centroid)

	return new_centroids

def KMeans(X, n):

	centroids = initialize_centroids(X, n)
	previous_centroids = [[0 for _ in xrange(len(X[0]))] for _ in xrange(len(x[0])) ]
	n_iters = 0
	while not converged(centroids, previous_centroids, n_iters):
		centroids = rearrange_centroids(X, centroids)
		n_iters += 1
	y = [centroid_of(x, centroids) for x in X]

	return y

