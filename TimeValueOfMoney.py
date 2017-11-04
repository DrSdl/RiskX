"""
Time value of money
If I promise you that I give you 1 USD per year for the next 100 years, what is this promise
worth today?
"""

import math as mt
import numpy as np

# we want to generate an array filled with 1 for 100 entries

money=np.empty(100)
money.fill(1.0)
#print(money)

rate=0.05
N=money.size
sum=0

for i in range(0,N):
    sum=sum+money[i]/(np.power(1+rate,i+1))

# present day, discounted value of money stream
print(sum)
