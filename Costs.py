
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


@njit(parallel=True)
def Calc_Cost_sPODG(V, as_, qs_target, f, intIds, weights, dx, dt, lamda):
    q = np.zeros_like(qs_target)

    for i in prange(f.shape[1]):
        V_delta = weights[i] * V[intIds[i]] + (1 - weights[i]) * V[intIds[i] + 1]
        q[:, i] = V_delta @ as_[:, i]

    q_res = q - qs_target

    cost = 1 / 2 * (L2norm_FOM(q_res, dx, dt)) + (lamda / 2) * (L2norm_ROM(f, dt))

    return cost








def finite_difference(f, q0, A_p, psi, qs_target, cost, wf, dx, dt, lamda, delta=None):
    m, n = f.shape
    gradient = np.zeros((m, n))

    # Loop over each element of the matrix X
    for i in range(m):
        for j in range(n):
            # Create a perturbation matrix
            delta_f = np.zeros_like(f)
            delta_f[i, j] = delta

            # Compute finite difference
            qs = wf.TI_primal(q0, f + delta_f, A_p, psi)
            gradient[i, j] = (Calc_Cost(qs, qs_target, f + delta_f, dx, dt, lamda) - cost) / delta
            print(i, j)
    return gradient