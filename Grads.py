from numba import njit, prange
import numpy as np


@njit
def Calc_Grad(mask, f, qs_adj, lamda):
    dL_du = lamda * f + mask.T @ qs_adj

    return dL_du


@njit
def Calc_Grad_PODG(psir_, f, as_adj, lamda):
    dL_du = lamda * f + psir_.T @ as_adj
    return dL_du


@njit(parallel=True)
def Calc_Grad_sPODG(mask, f, V, as_adj, intIds, weights, lamda):
    qs_adj = np.zeros((mask.shape[0], f.shape[1]))

    for i in prange(f.shape[1]):
        V_idx = intIds[i]
        V_delta = weights[i] * V[V_idx] + (1 - weights[i]) * V[V_idx + 1]
        qs_adj[:, i] = V_delta @ as_adj[:, i]

    dL_du = lamda * f + mask.T @ qs_adj

    return dL_du, qs_adj


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


@njit
def Calc_Grad_smooth(mask, qs_adj, f, lamda2):
    dL_du = mask.T @ qs_adj + lamda2 * f

    return dL_du


@njit
def prox_l1(data, reg_param):
    tmp = np.abs(data) - reg_param
    tmp = (tmp + np.abs(tmp)) / 2
    y = np.sign(data) * tmp
    return y


@njit
def Calc_Grad_mapping(u, psi, qs_adj, omega, lamda1, lamda2):
    return (1 / omega) * (u - prox_l1(u - omega * Calc_Grad_smooth(psi, qs_adj, u, lamda2), omega * lamda1))