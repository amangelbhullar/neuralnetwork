import math
import numpy as np
from sys import argv

fn = argv[1]

# Partition training and test data
def partition(w, size = 0.75):
	# Randomly shuffle the array and split it into two parts
	p = np.random.permutation(w)
	c = round(size * len(w))
	return p[:c], p[c:]

# Determine sign of dot product
def sign(w,x):
	return 1 if np.dot(w.transpose(),x) > 0 else -1

# Perceptron learning with optional pocket feature (returns both)
# Arguments: data set, learning rate, error after which to stop, iterations after which to stop
def perceptron(dataset, rate = 0.1, cutoff = 0.01, maxiter = math.inf):
	best_err = math.inf
	best_w = None
	err = math.inf
	m = len(dataset)
	r_m = 1/m
	# Randomly initialize weights
	#w = np.array([np.random.uniform(-1,1),np.random.uniform(-1,1),np.random.uniform(-1,1)])
	# Initialize weights as zeros
	w = np.array([0.0, 0.0,0.0])
	count = 0
	while err > cutoff and count < maxiter:
		err = 0
		for ex in dataset:
			ex_sign = sign(w,ex[:3])
			ex_err = 0.5 * (ex[3] - ex_sign)
			err += r_m * abs(ex_err)
			w += rate * ex_err * ex[:3]
		# Keep track of the lowest seen error for the pocket algorithm
		if err < best_err:
			best_err = err
			best_w = w[:]
		count += 1
	return w, best_w

# Gradient descent (Adaline)
# Arguments: data set, learning rate, error after which to stop, iterations after which to stop,
# boolean option for batch (True) or incremental/stochastic (False)
def gradientDescent(dataset, rate = 0.1, cutoff = 0.01, maxiter = math.inf, batch = True):
	err = math.inf
	w = np.array([np.random.uniform(-1.0,1.0),np.random.uniform(-1.0,1.0),np.random.uniform(-1.0,1.0)])
	count = 0
	while err > cutoff and count < maxiter:
		delw = np.array([0.0,0.0,0.0])
		err = 0
		for ex in dataset:
			y = np.dot(w.transpose(), ex[:3])
			# Update weights based on the mode we're using
			if batch:
				delw += rate * (ex[3] - y) * ex[:3]
			else:
				w += rate * (ex[3] - y) * ex[:3]
			err += 0.5 * (ex[3] - np.dot(w.transpose(), ex[:3])) ** 2
		if batch:
			w += delw
		count += 1
	return w

# Determine the % accuracy of weights on a dataset
def accuracy_perceptron(dataset, w):
	count = 0
	for ex in dataset:
		if sign(w,ex[:3]) == ex[3]:
			count += 1
	return count / len(dataset)

# Determine the MSE for gradient descent
def err_gd(dataset, w):
	m = len(dataset)
	err = 0.0
	for ex in dataset:
		err += 0.5 * (ex[3] - np.dot(w.transpose(), ex[:3])) ** 2
	return 1/m * err

# Read the file to sample into memory
with open(fn) as f:
	contents = f.read()
# Split into lines
contents = contents.split('\n')
# Break each valid line into points and make a list
pts = []
for line in contents:
	line = [1] + line.split(" ")
	if len(line) == 4:
		pts.append(np.array([float(x) for x in line]))

# Randomly select test/validation sets
test, val = partition(pts,0.75)

# Test perceptron and pocket perceptron
w, pocket_w = perceptron(test, 1e-10, 0, 1000)
print(w)
print(pocket_w)
print(accuracy_perceptron(val,w))
print(accuracy_perceptron(val,pocket_w))

# Test batch gradient descent
# Note: 2e-10 seems to be the highest learning rate that gives reasonable results
# 1e-10 seems about the highest that converges
w = gradientDescent(test, 2e-10, 0, 1000, True)
print(w)
print(err_gd(val,w))

# Test iterative/stochastic gradient descent
w = gradientDescent(test, 2e-10, 0, 1000, False)
print(w)
print(err_gd(val,w))
