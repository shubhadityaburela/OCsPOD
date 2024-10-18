from numba import njit

from Helper import *


def Calc_Cost(q, q_target, f, **kwargs):
    q_res = np.copy(q - q_target)
    cost = 1 / 2 * (L2norm_FOM(q_res, **kwargs)) + (kwargs['lamda'] / 2) * (L2norm_ROM(f, **kwargs))

    return cost


def Calc_Cost_PODG(V, as_, qs_target, f, **kwargs):
    q_res = np.copy(V @ as_ - qs_target)

    cost = 1 / 2 * (L2norm_FOM(q_res, **kwargs)) + (kwargs['lamda'] / 2) * (L2norm_ROM(f, **kwargs))

    return cost


def Calc_Cost_sPODG(V, as_, qs_target, f, intIds, weights, **kwargs):
    q = np.zeros_like(qs_target)
    for i in range(f.shape[1]):
        V_delta = weights[i] * V[intIds[i]] + (1 - weights[i]) * V[intIds[i] + 1]
        q[:, i] = V_delta @ as_[:-1, i]

    q_res = np.copy(q - qs_target)

    cost = 1 / 2 * (L2norm_FOM(q_res, **kwargs)) + (kwargs['lamda'] / 2) * (L2norm_ROM(f, **kwargs))

    return cost
