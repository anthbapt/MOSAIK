from math import floor, sqrt
import numpy as np

def pair(x, y):
    """Encode a pair of non-negative integers

    From http://szudzik.com/ElegantPairing.pdf

    Args:
        x (int): first number
        y (int): second number
    """    
    if x == max(x, y):
        return x * x + x + y
    else:
        return y * y + x

def unpair(z):
    """Return encoded pair

    Args:
        z (int): encoded fov and CellId
    """    
    zflr = floor(sqrt(z))
    if (z - zflr**2) < zflr:
        return (z - zflr**2, zflr)
    else:
        return (zflr, z - zflr**2 - zflr)

def pair_np(x, y):
    """Encode x to array of y

    Zero remains zero.

    Args:
        x (int): number
        y (ndarray): numpy array of int
    """    
    z = y != 0
    a = (x >= y) & z
    b = (x < y) & z
    np.putmask(y, a, x * x + x + y)
    np.putmask(y, b, y * y + x)