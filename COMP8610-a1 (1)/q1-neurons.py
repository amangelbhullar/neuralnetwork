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

# Gradient descent (Adaline)
# Arguments: data set, learning rate, error after which to stop, iterations after which to stop,
# boolean option for batch (True) or incremental/stochastic (False)
def gradientDescent(dataset, rate = 0.1, cutoff = 0.01, maxiter = math.inf, deg = 2, batch = True):
	err = math.inf
	# Increment degree to allow for a constant term
	deg += 1
	w = np.array([np.random.uniform(-1.0,1.0) for _ in range(deg)])
	count = 0
	while err > cutoff and count < maxiter:
		delw = np.array([0.0 for _ in range(deg)])
		err = 0
		for ex in dataset:
			y = np.dot(w.transpose(), np.array([ex[1] ** i for i in range(deg)]))
			# Update weights based on the mode we're using
			if batch:
				delw += rate * (ex[2] - y) * np.array([ex[1] ** i for i in range(deg)])
			else:
				w += rate * (ex[2] - y) * np.array([ex[1] ** i for i in range(deg)])
			err += 0.5 * (ex[2] - np.dot(w.transpose(), np.array([ex[1] ** i for i in range(deg)]))) ** 2
		if batch:
			w += delw
		count += 1
	return w

# Determine logistic regression function
def sigmoid(w,x):
	sig = 1 / (1 + math.exp(-(np.dot(w.transpose(),x))))
	if sig > 0.5:
	   return 1 
	else: 
		return -1


# Sigmoid neuron 
# Arguments: data set, learning rate, error after which to stop, iterations after which to stop
def sigmoidNeuron(dataset, rate = 0.1, cutoff = 0.01, maxiter = 100, deg = 2):
	err = math.inf
	deg += 1
	m = len(dataset)
	r_m = 1/m
	# Initialize weights as zeros
	w = np.array([0.0 for _ in range(deg)])
	count = 0
	while err > cutoff and count < maxiter:
		err = 0
		for ex in dataset:
			ex_net = sigmoid(w,np.array([ex[1] ** i for i in range(deg)]))
			ex_err = 0.5 * (ex[2] - ex_net)
			err += r_m * abs(ex_err)
			w += rate * ex_err * np.array([ex[1] ** i for i in range(deg)])
		count += 1
	return w

# Determine the MSE for gradient descent
def mse(dataset, w):
	m = len(dataset)
	deg = len(w)
	err = 0.0
	for ex in dataset:
		err += 0.5 * (ex[2] - np.dot(w.transpose(), np.array([ex[1] ** i for i in range(deg)]))) ** 2
	return 1/m * err

# Read the file to sample into memory
with open(fn) as f:
	contents = f.read()
# Split into lines
contents = contents.split('\n')
# Break each valid line into points and make a list
pts = []
for line in contents:
	line = line.split(" ")
	if len(line) == 2:
		pts.append(np.array([1.0, float(line[0]), float(line[1])]))

test, val = partition(pts,0.75)

print("Batch Gradient Descent:")
w = gradientDescent(test, 1e-10, 0, 1000, 3, True)
print(w)
print(mse(val,w))
print("Iterative Gradient Descent:")
w = gradientDescent(test, 1e-10, 0, 1000, 3, False)
print(w)
print(mse(val,w))
print("Sigmoid:")
# Run sigmoid
w = sigmoidNeuron(test, 1e-10, 0, 1000, 4)
print(w)
print(mse(val,w))
