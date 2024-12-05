from Helper import *
from numba import njit, prange


@njit
def Calc_Cost(q, q_target, f, dx, dt, lamda):
    q_res = q - q_target
    cost = 1 / 2 * (L2norm_FOM(q_res, dx, dt)) + (lamda / 2) * (L2norm_ROM(f, dt))

    return cost


@njit
def Calc_Cost_PODG(V, as_, qs_target, f, dx, dt, lamda):
    # Ensure arrays are contiguous
    V = np.ascontiguousarray(V)
    as_ = np.ascontiguousarray(as_)

    q_res = V @ as_ - qs_target

    cost = 1 / 2 * (L2norm_FOM(q_res, dx, dt)) + (lamda / 2) * (L2norm_ROM(f, dt))

    return cost


@njit(parallel=True)
def Calc_Cost_sPODG(V, as_, qs_target, f, intIds, weights, dx, dt, lamda):
    q = np.empty_like(qs_target)

    for i in prange(f.shape[1]):
        V_idx = intIds[i]
        V_delta = weights[i] * V[V_idx] + (1 - weights[i]) * V[V_idx + 1]
        q[:, i] = V_delta @ as_[:, i]

    q_res = q - qs_target

    cost = 1 / 2 * (L2norm_FOM(q_res, dx, dt)) + (lamda / 2) * (L2norm_ROM(f, dt))

    return cost, q


@njit
def Calc_Cost_sPODG_FRTO_NC(as_, as_target, f, dt, lamda):

    a_res = as_[:-1] - as_target[:-1]
    z_res = as_[-1:] - as_target[-1:]

    cost = 1 / 2 * (L2norm_ROM(a_res, dt)) + 1 / 2 * (L2norm_ROM(z_res, dt)) + (lamda / 2) * (L2norm_ROM(f, dt))

    return cost

