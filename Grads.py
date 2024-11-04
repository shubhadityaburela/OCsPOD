from numba import njit, prange


@njit
def Calc_Grad(mask, f, qs_adj, lamda):
    dL_du = lamda * f + mask.T @ qs_adj

    return dL_du
