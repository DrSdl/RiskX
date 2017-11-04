"""
Power generation cost estimates
Solve simple Lagrange equation for minimum cost dispatch
C1(PG1)=1000+20PG1+0.01PG1**2  generator G1 cost curve ($/hr)
C2(PG2)=400+15PG2+0.03PG2**2   generator G2 cost curve ($/hr)
PG1+PG2=500MW                  total demand

Derive solution with the help of a tensorflow cost minimisation approach
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.5
training_epochs = 100


PG1 = tf.Variable(100,dtype=tf.float32, name="PG1")
PG2 = tf.Variable(100,dtype=tf.float32, name="PG2")
lag = tf.Variable(10,dtype=tf.float32,  name="lagrange")

# challenge: why does this naive minimisation not work?
#cost = (1000.0+20.0*PG1+0.01*PG1*PG1) + (400.0+15.0*PG2+0.03*PG2*PG2) - lag*(PG1+PG2-500.0)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# instead use: https://www.tensorflow.org/api_docs/python/tf/contrib/opt/ScipyOptimizerInterface
cost = (1000.0+20.0*PG1+0.01*PG1*PG1) + (400.0+15.0*PG2+0.03*PG2*PG2)
# Ensure x is == 0.
equalities   = [PG1+PG2-500]
# Ensure x is >= 0.
inequalities = [PG1]

# install numpy+mkl from https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
optimizer = tf.contrib.opt.ScipyOptimizerInterface(cost, equalities=equalities, inequalities=inequalities, method='SLSQP')


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    optimizer.minimize(sess)

    # Fit all training data
    for epoch in range(training_epochs):
        #print(sess.run([PG1, PG2, cost]))
        #sess.run(optimizer)
        lag=1



    print("Optimization Finished!")
    #print(sess.run([PG1, PG2, lag, cost]))
    print(sess.run([PG1, PG2]))
