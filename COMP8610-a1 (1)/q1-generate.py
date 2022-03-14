import numpy as np
from sys import argv

# Name of file to write to
fn = argv[1]

# Draw 5000 random normal samples
x = np.random.normal(0,1,(5000))
# Draw 5000 random normal samples
eps = np.random.normal(0,0.25,(5000))
# Generate the y values
y = []
for i in range(5000):
	y.append(-1 + 0.5*x[i] - 2*(x[i]**2) + 0.3*(x[i]**3) + eps[i])

with open(fn, 'w+') as f:
	for i in range(5000):
		# Write generated points to file
		f.write(str(x[i]) + " " + str(y[i]) + "\n")
