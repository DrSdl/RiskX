"""
levy() implements one dimensional Levy Flight.
"""

# File: levy.py

from math import sqrt
from scipy.stats import norm
from scipy.stats import levy_stable
import numpy as np


def levyflight(x0, n, dt, alpha, beta, out=None):
    """
    Generate an instance of a Levy flight:

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.

    alpha, beta: Levy parameters

    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.

    Note that the initial value `x0` is not included in the returned array.
    """

    #can we really calculate the stats?
    #mean, var, skew, kurt = levy_stable.stats(alpha, beta, moments='mvsk')
    #print("Levy mean: ",mean," variance: ",var)

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = levy_stable.rvs(alpha, beta, size=x0.shape + (n,), scale= sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out