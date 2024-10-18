import numpy as np
from numba import njit, prange

from Helper_sPODG import make_Da


@njit
def Calc_Grad(mask, f, qs_adj, lamda):
    dL_du = lamda * f + mask.T @ qs_adj

    return dL_du
