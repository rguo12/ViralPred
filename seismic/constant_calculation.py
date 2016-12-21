import numpy as np
from sympy.solvers import solve
from sympy import Symbol
def compute_constant(theta,s_0):
    c = Symbol('c')
    return solve(c*(1/theta+1)*s_0-1,c)

if __name__ == '__main__':
    print compute_constant(0.440,30000)