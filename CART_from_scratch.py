
# Most of the code is referenced from : http://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

def gini(groups, classes):

	g = 0
	for value in classes:
		for group in groups:
			if len(group) == 0:
				continue
			prop = [x[-1] for x in group].count(value) / len(group) * 1.0
			g += prop * (1.0 - prop)

	return g

def entropy(groups, classes):
	from math import log
	ent = 0
	for value in classes:
		for group in groups:
			if len(group) == 0:
				continue
			prop = [x[-1] for x in group].count(value) / len(group) * 1.0
			if prop != 0:
				ent -= prop * log(prop, 2)

	return ent

def create_split(X, attr_index, val):

	left = []
	right = []
	for x in X:
		if x[attr_index] < val:
			left.append(x)
		else:
			right.append(x)
	return left, right

def get_split(X):

	classes = list(set(x[-1] for x in X))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for attr_index in xrange(len(X[0]) - 1):
		for x in X:
			groups = create_split(X, attr_index, x[attr_index])
			g = gini(groups, classes)
			if g < b_score:
				b_index, b_value, b_score, b_groups = index, x[index], gini, groups

	return {'index':b_index, 'value':b_value, 'groups':b_groups}

def create_terminal(group):
	classes = [x[-1] for x in group]
	return max(set(classes), key=classes.count)

def _build_tree(node, max_depth, min_size, depth):

	left, right = node['groups']
	del(node['groups'])
	if not left or not right:
		node['left'] = node['right'] = create_terminal(left + right)
		return
	if depth >= max_depth:
		node['left'], node['right'] = create_terminal(left), create_terminal(right)
		return
	if len(left) <= min_size:
		node['left'] = create_terminal(left)
	else:
		node['left'] = get_split(left)
		_build_tree(node['left'], max_depth, min_size, depth+1)
	if len(right) <= min_size:
		node['right'] = create_terminal(right)
	else:
		node['right'] = get_split(right)
		_build_tree(node['right'], max_depth, min_size, depth+1)

def build_tree(train, max_depth, min_size):
	root = get_split(dataset)
	_build_tree(root, max_depth, min_size, 1)
	return root

def predict(node, x):

	if x[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], x)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], x)
		else:
			return node['right']

