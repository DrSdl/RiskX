"""
Power generation cost estimates
We assume a 300MW power station with random power output
"""

import math as mt
import numpy as np
import matplotlib.pyplot as plt

# example of random number generation
power0=300
mu, sigma = power0, 10 # mean and standard deviation
# 100 hours of power generation
N = 10000
times=np.arange(1,N+1,1)


power = np.random.normal(mu, sigma, N) # power produced per hour of N steps forward
print(power)

# plot points
plt.plot(times, power, 'ro')
plt.ylabel('power (MW)')
plt.xlabel('time (h)')
plt.show()


# the histogram of the point data
n, bins, patches = plt.hist(power, 50, normed=1, facecolor='green', alpha=0.75)
plt.show()
