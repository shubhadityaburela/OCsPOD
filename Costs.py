
from Helper import *


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


@line_profiler.profile
@njit(parallel=True)
def Calc_Cost_sPODG(V, as_, qs_target, f, intIds, weights, dx, dt, lamda):
    q = np.zeros_like(qs_target)

    for i in prange(f.shape[1]):
        V_delta = weights[i] * V[intIds[i]] + (1 - weights[i]) * V[intIds[i] + 1]
        q[:, i] = V_delta @ as_[:, i]

    q_res = q - qs_target

    cost = 1 / 2 * (L2norm_FOM(q_res, dx, dt)) + (lamda / 2) * (L2norm_ROM(f, dt))

    return cost
