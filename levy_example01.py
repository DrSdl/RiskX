import numpy
from pylab import plot, show, grid, xlabel, ylabel
from levy import *
import matplotlib.pyplot as plt

# Simple Levy flight in Python

# The Levy Parameters.
alpha= 1.0
beta = 0.5
# Total time.
T = 10.0
# Number of steps.
N = 500
# Time step size
dt = T/N
# Number of realizations to generate.
m = 20
# Create an empty array to store the realizations.
x = numpy.empty((m,N+1))
#print(x.shape)
# Initial values of x.
x[:, 0] = 0.0
#print(x[:, 0])

levyflight(x[:,0], N, dt, alpha, beta, out=x[:,1:])

t = numpy.linspace(0.0, N*dt, N+1)
for k in range(m):
    plot(t, x[k])
xlabel('t', fontsize=16)
ylabel('x', fontsize=16)
grid(True)
show()

# now we want to do an histogram of all the sample values
y = x.flatten()
plt.hist(y, bins='auto')
plt.title("Histogram of Levyflight data with 'auto' bins")
plt.show()
