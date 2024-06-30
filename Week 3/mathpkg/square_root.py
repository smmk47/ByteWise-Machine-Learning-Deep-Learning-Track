# square_root.py

import math

def square_root(a):
   
    if a < 0:
        raise ValueError("square root not defined for negative numbers")
    return math.sqrt(a)
