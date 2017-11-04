"""
Power generation cost estimates
Solve simple Lagrange equation for minimum cost dispatch
C1(PG1)=1000+20PG1+0.01PG1**2  generator G1 cost curve ($/hr)
C2(PG2)=400+15PG2+0.03PG2**2   generator G2 cost curve ($/hr)
PG1+PG2=500MW                  total demand
"""

import math as mt
import numpy as np

M=np.matrix([[0.02, 0, -1.0], [0, 0.06, -1.0], [-1.0, -1.0, 0]])
v=np.matrix([[-20.0], [-15.0], [-500.0]])

#print(M)
#print(v)

x = np.linalg.solve(M, v)

print(x)
