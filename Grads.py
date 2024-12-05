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


@njit(parallel=True)
def Calc_Grad_sPODG_FRTO(f, C, as_adj, as_, intIds, weights, lamda):
    as_adj_1 = np.zeros_like(f)
    as_adj_2 = np.zeros_like(f)

    for i in prange(f.shape[1]):
        V_idx = intIds[i]
        m1 = np.add(weights[i] * C[0, V_idx], (1 - weights[i]) * C[0, V_idx + 1]).T
        m2 = np.add(weights[i] * C[1, V_idx], (1 - weights[i]) * C[1, V_idx + 1]).T
        Da = as_[:-1, i].reshape(-1, 1)
        as_adj_1[:, i] = m1 @ as_adj[:-1, i]
        as_adj_2[:, i] = m2 @ (Da @ as_adj[-1:, i])

    dL_du = lamda * f + as_adj_1 + as_adj_2

    return dL_du
