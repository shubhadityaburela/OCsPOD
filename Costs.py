from Helper import *
from numba import njit, prange


@njit
def J_smooth(q_res, f, lamda2, dx, dt):
    return 1 / 2 * (L2norm_FOM(q_res, dx, dt)) + (lamda2 / 2) * (L2norm_ROM(f, dt))


@njit
def J_nonsmooth(f, lamda1, dx, dt, adjust):
    return lamda1 * (L1norm_FOM(f, dx, dt, adjust))


@njit
def Calc_Cost(q, q_target, f, C, dx, dt, lamda1, lamda2, adjust):
    q_res = (q - q_target)[C]

    return J_smooth(q_res, f, lamda2, dx, dt), J_nonsmooth(f, lamda1, dx, dt, adjust)


@njit
def Calc_Cost_PODG(V, as_, qs_target, f, C, dx, dt, lamda1, lamda2, adjust):
    # Ensure arrays are contiguous
    V = np.ascontiguousarray(V)
    as_ = np.ascontiguousarray(as_)

    q_res = (V @ as_ - qs_target)[C]

    return J_smooth(q_res, f, lamda2, dx, dt), J_nonsmooth(f, lamda1, dx, dt, adjust)


@njit(parallel=True)
def Calc_Cost_sPODG(V, as_, qs_target, f, intIds, weights, dx, dt, lamda1, lamda2, adjust):
    q = np.empty_like(qs_target)

    for i in prange(f.shape[1]):
        V_idx = intIds[i]
        V_delta = weights[i] * V[V_idx] + (1 - weights[i]) * V[V_idx + 1]
        q[:, i] = V_delta @ as_[:, i]

    q_res = (q - qs_target)

    return J_smooth(q_res, f, lamda2, dx, dt), J_nonsmooth(f, lamda1, dx, dt, adjust), q


@njit
def Calc_Cost_sPODG_FRTO_NC(as_, as_target, f, dt, lamda):

    a_res = as_[:-1] - as_target[:-1]
    z_res = as_[-1:] - as_target[-1:]

    cost = 1 / 2 * (L2norm_ROM(a_res, dt)) + 1 / 2 * (L2norm_ROM(z_res, dt)) + (lamda / 2) * (L2norm_ROM(f, dt))

    return cost
