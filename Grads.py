from numba import njit, prange
import numpy as np

@njit
def Calc_Grad(mask, f, qs_adj, lamda):
    dL_du = lamda * f + mask.T @ qs_adj

    return dL_du


@njit(parallel=True)
def Calc_Grad_sPODG(mask, f, V, as_adj, intIds, weights, lamda):
    qs_adj = np.zeros((mask.shape[0], f.shape[1]))

    for i in prange(f.shape[1]):
        V_idx = intIds[i]
        V_delta = weights[i] * V[V_idx] + (1 - weights[i]) * V[V_idx + 1]
        qs_adj[:, i] = V_delta @ as_adj[:, i]

    dL_du = lamda * f + mask.transpose() @ qs_adj

    return dL_du
