from numba import njit, prange
import numpy as np

from Helper import compute_state


@njit
def Calc_Grad_smooth(mask, f, qs_adj, lamda2):
    dL_du = lamda2 * f + mask.T @ qs_adj

    return dL_du


@njit
def Calc_Grad_PODG_smooth(psir_, f, as_adj, lamda2):
    dL_du = lamda2 * f + psir_.T @ as_adj
    return dL_du


@njit
def Calc_Grad_sPODG_smooth(mask, f, V, as_adj, intIds, weights, lamda2):

    qs_adj = compute_state(V, V[0].shape[0], f.shape[1], as_adj, intIds, weights)

    dL_du = lamda2 * f + mask.T @ qs_adj

    return dL_du, qs_adj


@njit(parallel=True)
def Calc_Grad_sPODG_FRTO_smooth(f, C, as_adj, as_, intIds, weights, lamda2):
    as_adj_1 = np.zeros_like(f)
    as_adj_2 = np.zeros_like(f)

    for i in prange(f.shape[1]):
        V_idx = intIds[i]
        m1 = np.add(weights[i] * C[0, V_idx], (1 - weights[i]) * C[0, V_idx + 1]).T
        m2 = np.add(weights[i] * C[1, V_idx], (1 - weights[i]) * C[1, V_idx + 1]).T
        Da = as_[:-1, i].reshape(-1, 1)
        as_adj_1[:, i] = m1 @ as_adj[:-1, i]
        as_adj_2[:, i] = m2 @ (Da @ as_adj[-1:, i])

    dL_du = lamda2 * f + as_adj_1 + as_adj_2

    return dL_du


@njit
def prox_l1(data, reg_param):
    tmp = np.abs(data) - reg_param
    tmp = (tmp + np.abs(tmp)) / 2
    y = np.sign(data) * tmp
    return y


@njit
def Calc_Grad_mapping(u, dL_du_s, omega, lamda1):
    return (1 / omega) * (u - prox_l1(u - omega * dL_du_s, omega * lamda1))