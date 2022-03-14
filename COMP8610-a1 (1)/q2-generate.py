import numpy as np
from sys import argv

# Take line info as command arguments
m = int(argv[1])
b = int(argv[2])
# Name of file to write to
fn = argv[3]

with open(fn, 'w+') as f:
	for i in range(2500):
		# Generate + instance
		plusx = np.random.uniform(-1000,1000)
		# Subtract negative y-adjustment to get around half-closed interval
		plusy = m * plusx + b - np.random.uniform(-1000,0)
		# Generate - instance
		minusx = np.random.uniform(-1000,1000)
		minusy = m * minusx + b + np.random.uniform(-1000,0)
		# Write generated points to file
		f.write(str(plusx) + " " + str(plusy) + " +1\n")
		f.write(str(minusx) + " " + str(minusy) + " -1\n")
