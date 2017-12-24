# -*- coding: utf-8 -*-

from __future__ import division
from numpy import polyfit, arange


def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
   
    f is a vector and x is an index for that vector.
   
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
   
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
   
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
   
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
   
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return xv, yv


def parabolic_polyfit(f, x, n):
    """Use the built-in polyfit() function to find the peak of a parabola
    
    f is a vector and x is an index for that vector.
    
    n is the number of samples of the curve used to fit the parabola.

    """    
    a, b, c = polyfit(arange(x-n//2, x+n//2+1), f[x-n//2:x+n//2+1], 2)
    xv = -0.5 * b/a
    yv = a * xv**2 + b * xv + c
    return xv, yv


if __name__ == "__main__":
    from numpy import argmax
    import matplotlib.pyplot as plt
    
    y = [2, 1, 4, 8, 11, 10, 7, 3, 1, 1]
    
    xm, ym = argmax(y), y[int(argmax(y))]
    xp, yp = parabolic(y, argmax(y))
    
    plot = plt.plot(y)
    plt.hold(True)
    plt.plot(xm, ym, 'o', color='silver')
    plt.plot(xp, yp, 'o', color='blue')
    plt.title('silver = max, blue = estimated max')
