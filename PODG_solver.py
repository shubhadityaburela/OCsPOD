import numpy as np
import scipy
from numba import njit
from scipy import sparse

from Helper_sPODG import Target_offline_adjoint_FOTR_mix, findIntervalAndGiveInterpolationWeight_1D
from TI_schemes import rk4_PODG_prim, rk4_PODG_adj, implicit_midpoint_PODG_adj, DIRK_PODG_adj, bdf2_PODG_adj, \
    rk4_PODG_adj_, implicit_midpoint_PODG_adj_, DIRK_PODG_adj_, bdf2_PODG_adj_


#############
## FOTR POD
#############

@njit
def IC_primal_PODG_FOTR(V_p, q0):
    return V_p.T @ q0


@njit
def RHS_primal_PODG_FOTR(a, f, Ar_p, psir_p):
    Ar_p = np.ascontiguousarray(Ar_p)
    a = np.ascontiguousarray(a)
    psir_p = np.ascontiguousarray(psir_p)
    f = np.ascontiguousarray(f)
    return Ar_p @ a + psir_p @ f


@njit
def TI_primal_PODG_FOTR(a, f0, Ar_p, psir_p, Nt, dt):
    # Time loop
    as_ = np.zeros((a.shape[0], Nt))
    as_[:, 0] = a

    for n in range(1, Nt):
        as_[:, n] = rk4_PODG_prim(RHS_primal_PODG_FOTR, as_[:, n - 1], f0[:, n - 1], f0[:, n], dt, Ar_p, psir_p)

    return as_


def mat_primal_PODG_FOTR(A_p, V_p, psi):
    V_pT = V_p.T

    return (V_pT @ A_p) @ V_p, V_pT @ psi


@njit
def IC_adjoint_PODG_FOTR(Nm_a):
    return np.zeros(Nm_a)


@njit
def RHS_adjoint_PODG_FOTR_expl(a_adj, a_pri, Tarr_a, A_f, V_aTV_p, dx):
    A_f = np.ascontiguousarray(A_f)
    V_aTV_p = np.ascontiguousarray(V_aTV_p)
    Tarr_a = np.ascontiguousarray(Tarr_a)
    a_adj = np.ascontiguousarray(a_adj)
    a_pri = np.ascontiguousarray(a_pri)
    return - A_f @ a_adj - dx * (V_aTV_p @ a_pri - Tarr_a)


def RHS_adjoint_PODG_FOTR_impl(a_adj, a, Tarr_a, M_f, A_f, LU_M_F, V_aTV_p, dx, dt, scheme):
    if scheme == "implicit_midpoint":
        return scipy.linalg.lu_solve((LU_M_F[0], LU_M_F[1]), A_f @ a_adj - dt * dx * (V_aTV_p @ a - Tarr_a))
    elif scheme == "DIRK":
        return scipy.linalg.lu_solve((LU_M_F[0], LU_M_F[1]), - A_f @ a_adj - dx * (V_aTV_p @ a - Tarr_a))
    elif scheme == "BDF2":
        return scipy.linalg.lu_solve((LU_M_F[0], LU_M_F[1]), 4.0 * a_adj[1] - 1.0 * a_adj[0] -
                                     2.0 * dt * dx * (V_aTV_p @ a - Tarr_a))


def TI_adjoint_PODG_FOTR(at_adj, a_, M_f, A_f, LU_M_f, V_aTV_p, Tarr_a, Nt, dt, dx, scheme):
    as_adj = np.zeros((at_adj.shape[0], Nt))
    as_adj[:, -1] = at_adj

    if scheme == "RK4":
        for n in range(1, Nt):
            as_adj[:, -(n + 1)] = rk4_PODG_adj_(RHS_adjoint_PODG_FOTR_expl, as_adj[:, -n], a_[:, -n], a_[:, -(n + 1)],
                                                Tarr_a[:, -n], Tarr_a[:, -(n + 1)], - dt, A_f, V_aTV_p, dx)
    elif scheme == "implicit_midpoint":
        for n in range(1, Nt):
            as_adj[:, -(n + 1)] = implicit_midpoint_PODG_adj_(RHS_adjoint_PODG_FOTR_impl, as_adj[:, -n], a_[:, -n],
                                                              a_[:, -(n + 1)],
                                                              Tarr_a[:, -n], Tarr_a[:, -(n + 1)], - dt, M_f, A_f, LU_M_f
                                                              , V_aTV_p,
                                                              dx, scheme)
    elif scheme == "DIRK":
        for n in range(1, Nt):
            as_adj[:, -(n + 1)] = DIRK_PODG_adj_(RHS_adjoint_PODG_FOTR_impl, as_adj[:, -n], a_[:, -n],
                                                 a_[:, -(n + 1)],
                                                 Tarr_a[:, -n], Tarr_a[:, -(n + 1)], - dt, M_f, A_f, LU_M_f, V_aTV_p,
                                                 dx, scheme)
    elif scheme == "BDF2":
        # last 2 steps (x_{n-1}, x_{n-2}) with RK4 (Effectively 2nd order)
        for n in range(1, 2):
            as_adj[:, -(n + 1)] = rk4_PODG_adj_(RHS_adjoint_PODG_FOTR_expl, as_adj[:, -n], a_[:, -n], a_[:, -(n + 1)],
                                                Tarr_a[:, -n], Tarr_a[:, -(n + 1)], - dt, A_f, V_aTV_p, dx)
        for n in range(2, Nt):
            as_adj[:, -(n + 1)] = bdf2_PODG_adj_(RHS_adjoint_PODG_FOTR_impl, as_adj, a_[:, -(n + 1)],
                                                 Tarr_a[:, -(n + 1)], - dt, M_f, A_f,
                                                 LU_M_f, V_aTV_p, dx, scheme, n)

    return as_adj


def RHS_adjoint_PODG_FOTR_impl_mix(a_adj, a_, Tarr_a, ds, A_f, LU_M_F, V_aTVd_p, dx, dt, scheme):
    as_p = a_[:-1]
    z_p = a_[-1]
    intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], -z_p)
    V_aTVd_p_ = np.add(weight * V_aTVd_p[intervalIdx], (1 - weight) * V_aTVd_p[intervalIdx + 1])

    return scipy.linalg.lu_solve((LU_M_F[0], LU_M_F[1]), - A_f @ a_adj - dx * (V_aTVd_p_ @ as_p - Tarr_a))


def TI_adjoint_PODG_FOTR_mix(at_adj, a_, M_f, A_f, LU_M_f, V_aTVd_p, Tarr_a, delta_s, Nt, dt, dx, scheme):
    as_adj = np.zeros((at_adj.shape[0], Nt))
    as_adj[:, -1] = at_adj

    if scheme == "DIRK":
        # We have used a tiny dirty argument passing here. To spare ouselves from redefining DIRK_PODG_adj_,
        # we have instead to sending M_f have passed delta_s in the argument list. This does not make any significant
        # difference since M_f is anyway never used. So keep this in mind.
        for n in range(1, Nt):
            as_adj[:, -(n + 1)] = DIRK_PODG_adj_(RHS_adjoint_PODG_FOTR_impl_mix, as_adj[:, -n], a_[:, -n],
                                                 a_[:, -(n + 1)],
                                                 Tarr_a[:, -n], Tarr_a[:, -(n + 1)], - dt, delta_s, A_f, LU_M_f,
                                                 V_aTVd_p,
                                                 dx, scheme)
    else:
        print("Not implemented")
        exit()
    return as_adj


def mat_adjoint_PODG_FOTR_mix(A_a, V_a, Vd_p, qs_target, psi, samples, modes_a, modes_p):
    V_aT = V_a.T
    V_aTVd_p = Target_offline_adjoint_FOTR_mix(Vd_p, V_aT, samples, modes_a, modes_p)

    return (V_aT @ A_a) @ V_a, V_aTVd_p, V_aT @ qs_target, V_aT @ psi


def mat_adjoint_PODG_FOTR(A_a, V_a, V_p, qs_target, psi, CTC):
    V_aT = V_a.T

    return (V_aT @ A_a) @ V_a, V_aT[:, CTC] @ V_p[CTC, :], V_aT[:, CTC] @ qs_target[CTC, :], V_aT @ psi


#############
## FRTO POD
#############

@njit
def IC_primal_PODG_FRTO(V, q0):
    return V.T @ q0


@njit
def RHS_primal_PODG_FRTO(a, f, Ar_p, psir_p):
    Ar_p = np.ascontiguousarray(Ar_p)
    a = np.ascontiguousarray(a)
    psir_p = np.ascontiguousarray(psir_p)
    f = np.ascontiguousarray(f)
    return Ar_p @ a + psir_p @ f


@njit
def TI_primal_PODG_FRTO(a, f0, Ar_p, psir_p, Nt, dt):
    # Time loop
    as_ = np.zeros((a.shape[0], Nt))
    as_[:, 0] = a

    for n in range(1, Nt):
        as_[:, n] = rk4_PODG_prim(RHS_primal_PODG_FRTO, as_[:, n - 1], f0[:, n - 1], f0[:, n], dt, Ar_p, psir_p)

    return as_


def mat_primal_PODG_FRTO(A_p, V, psi):
    return (V.T @ A_p) @ V, V.T @ psi


@njit
def IC_adjoint_PODG_FRTO(modes):
    return np.zeros(modes)


@njit
def RHS_adjoint_PODG_FRTO_expl(a_adj, a, Tarr_a, Ar_a, dx):
    Ar_a = np.ascontiguousarray(Ar_a)
    a_adj = np.ascontiguousarray(a_adj)
    a = np.ascontiguousarray(a)
    Tarr_a = np.ascontiguousarray(Tarr_a)
    return - Ar_a @ a_adj - dx * (a - Tarr_a)


def RHS_adjoint_PODG_FRTO_impl(a_adj, a, Tarr_a, M_f, A_f, LU_M_F, Nx, dx, dt, scheme):
    if scheme == "implicit_midpoint":
        return scipy.linalg.lu_solve((LU_M_F[0], LU_M_F[1]), A_f @ a_adj - dt * dx * (a - Tarr_a))
    elif scheme == "DIRK":
        return scipy.linalg.lu_solve((LU_M_F[0], LU_M_F[1]), - A_f @ a_adj - dx * (a - Tarr_a))
    elif scheme == "BDF2":
        return scipy.linalg.lu_solve((LU_M_F[0], LU_M_F[1]), 4.0 * a_adj[1] - 1.0 * a_adj[0] -
                                     2.0 * dt * dx * (a - Tarr_a))


def TI_adjoint_PODG_FRTO(at_adj, as_, M_f, A_f, LU_M_f, Tarr_a, Nx, dx, Nt, dt, scheme):
    as_adj = np.zeros((at_adj.shape[0], Nt))
    as_adj[:, -1] = at_adj

    if scheme == "RK4":
        for n in range(1, Nt):
            as_adj[:, -(n + 1)] = rk4_PODG_adj(RHS_adjoint_PODG_FRTO_expl, as_adj[:, -n], as_[:, -n], as_[:, -(n + 1)],
                                               Tarr_a[:, -n], Tarr_a[:, -(n + 1)], - dt, A_f, dx)
    elif scheme == "implicit_midpoint":
        for n in range(1, Nt):
            as_adj[:, -(n + 1)] = implicit_midpoint_PODG_adj(RHS_adjoint_PODG_FRTO_impl, as_adj[:, -n], as_[:, -n],
                                                             as_[:, -(n + 1)],
                                                             Tarr_a[:, -n], Tarr_a[:, -(n + 1)], - dt, M_f, A_f,
                                                             LU_M_f,
                                                             Nx, dx,
                                                             scheme)
    elif scheme == "DIRK":
        for n in range(1, Nt):
            as_adj[:, -(n + 1)] = DIRK_PODG_adj(RHS_adjoint_PODG_FRTO_impl, as_adj[:, -n], as_[:, -n],
                                                as_[:, -(n + 1)],
                                                Tarr_a[:, -n], Tarr_a[:, -(n + 1)], - dt, M_f, A_f,
                                                LU_M_f,
                                                Nx, dx,
                                                scheme)
    elif scheme == "BDF2":
        # last 2 steps (x_{n-1}, x_{n-2}) with RK4 (Effectively 2nd order)
        for n in range(1, 2):
            as_adj[:, -(n + 1)] = rk4_PODG_adj(RHS_adjoint_PODG_FRTO_expl, as_adj[:, -n], as_[:, -n], as_[:, -(n + 1)],
                                               Tarr_a[:, -n], Tarr_a[:, -(n + 1)], - dt, A_f, dx)
        for n in range(2, Nt):
            as_adj[:, -(n + 1)] = bdf2_PODG_adj(RHS_adjoint_PODG_FRTO_impl, as_adj, as_[:, -(n + 1)],
                                                Tarr_a[:, -(n + 1)], - dt, M_f, A_f,
                                                LU_M_f, Nx, dx, scheme, n)

    return as_adj


def mat_adjoint_PODG_FRTO(A_a, V, qs_target, CTC):
    return (V.T @ A_a) @ V, V[CTC, :].T @ qs_target[CTC, :]
