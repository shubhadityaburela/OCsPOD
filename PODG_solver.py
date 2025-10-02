from typing import NamedTuple

import numpy as np
import scipy
from numba import njit
from scipy import sparse
from scipy.linalg import qr
from scipy.sparse import csc_matrix

from Helper_sPODG import Target_offline_adjoint_FOTR_mix, findIntervalAndGiveInterpolationWeight_1D
from TI_schemes import rk4_PODG_prim, rk4_PODG_adj, implicit_midpoint_PODG_adj, DIRK_PODG_adj, bdf2_PODG_adj, \
    rk4_PODG_adj_, implicit_midpoint_PODG_adj_, DIRK_PODG_adj_, bdf2_PODG_adj_, rk4_PODG_prim_kdvb, rk4_PODG_adj_kdvb, \
    implicit_midpoint_PODG_FRTO_primal_kdvb, implicit_midpoint_PODG_FRTO_adjoint_kdvb, \
    implicit_midpoint_PODG_FOTR_primal_kdvb, implicit_midpoint_PODG_FOTR_adjoint_kdvb, rk4_PODG_adj_kdvb_, \
    rk4_PODG_adj__, rk4_PODG_adj_kdvb__


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


def mat_adjoint_PODG_FOTR(A_a, V_a, V_p, qs_target, psi, CTC):
    V_aT = V_a.T

    return (V_aT @ A_a) @ V_a, V_aT[:, CTC] @ V_p[CTC, :], V_aT[:, CTC] @ qs_target[CTC, :], V_aT @ psi


@njit
def RHS_adjoint_PODG_FOTR_expl_mix(a_adj, a_, Tarr_a, ds, A_f, V_aTVd_p, dx):
    A_f = np.ascontiguousarray(A_f)
    Tarr_a = np.ascontiguousarray(Tarr_a)
    a_adj = np.ascontiguousarray(a_adj)

    as_p = a_[:-1]
    z_p = a_[-1]
    intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], -z_p)
    V_aTVd_p_ = np.add(weight * V_aTVd_p[intervalIdx], (1 - weight) * V_aTVd_p[intervalIdx + 1])

    return - A_f @ a_adj - dx * (V_aTVd_p_ @ as_p - Tarr_a)


def TI_adjoint_PODG_FOTR_mix(at_adj, a_, A_f, V_aTVd_p, Tarr_a, delta_s, Nt, dt, dx, scheme):
    as_adj = np.zeros((at_adj.shape[0], Nt))
    as_adj[:, -1] = at_adj

    if scheme == "RK4":
        for n in range(1, Nt):
            as_adj[:, -(n + 1)] = rk4_PODG_adj__(RHS_adjoint_PODG_FOTR_expl_mix, as_adj[:, -n], a_[:, -n],
                                                 a_[:, -(n + 1)],
                                                 Tarr_a[:, -n], Tarr_a[:, -(n + 1)], - dt, delta_s, A_f, V_aTVd_p, dx)
    else:
        print("Not implemented")
        exit()
    return as_adj


def mat_adjoint_PODG_FOTR_mix(A_a, V_a, Vd_p, qs_target, psi, CTC, samples, modes_a, modes_p):
    V_aT = V_a.T
    V_aTVd_p = Target_offline_adjoint_FOTR_mix(Vd_p, V_aT, CTC, samples, modes_a, modes_p)
    return (V_aT @ A_a) @ V_a, V_aTVd_p, V_aT[:, CTC] @ qs_target[CTC, :], V_aT @ psi


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


# -------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #
#############
## FOTR POD
#############

class PrimalMatrices_PODG_kdv(NamedTuple):
    D_1r: np.ndarray
    D_2r: np.ndarray
    D_3r: np.ndarray
    B_r: np.ndarray
    prefactor: np.ndarray
    ST_V: np.ndarray
    ST_D1V: np.ndarray


class AdjointMatrices_PODG_FOTR_kdv(NamedTuple):
    D_1r: np.ndarray
    D_2r: np.ndarray
    D_3r: np.ndarray
    prefactor: np.ndarray
    ST_Va: np.ndarray
    ST_D1Va: np.ndarray
    ST_Vp: np.ndarray
    ST_D1Vp: np.ndarray
    VaT_Vp: np.ndarray
    VaT_CTC_qT: np.ndarray
    VaT_B: np.ndarray


@njit
def IC_primal_PODG_FOTR_kdv(V_p, q0):
    return V_p.T @ q0


def mat_primal_PODG_FOTR_kdv(V_p, V_deim_p, **params_primal) -> PrimalMatrices_PODG_kdv:
    D1, D2, D3 = params_primal['D1'], params_primal['D2'], params_primal['D3']
    B = params_primal['B']
    Nm_deim = V_deim_p.shape[1]

    # Linear factors
    D_1r = (V_p.T @ D1) @ V_p
    D_2r = (V_p.T @ D2) @ V_p
    D_3r = (V_p.T @ D3) @ V_p
    B_r = V_p.T @ B

    # Nonlinear DEIM prep
    _, _, piv = qr(V_deim_p.T, pivoting=True)
    S = np.sort(piv[:Nm_deim])
    STVdeim_inv = np.linalg.inv(V_deim_p[S])
    VT_Vdeim_ST_Vdeim_inv = (V_p.T @ V_deim_p) @ STVdeim_inv

    ST_V = V_p[S]
    ST_D1V = (D1 @ V_p)[S]

    return PrimalMatrices_PODG_kdv(
        D_1r=D_1r,
        D_2r=D_2r,
        D_3r=D_3r,
        B_r=B_r,
        prefactor=VT_Vdeim_ST_Vdeim_inv,
        ST_V=ST_V,
        ST_D1V=ST_D1V
    )


@njit
def RHS_primal_PODG_FOTR_kdv_expl(a, f, D_1r, D_2r, D_3r, prefactor, ST_V, ST_D1V, B_r, c, alpha, omega, gamma, nu):
    a = np.ascontiguousarray(a)
    f = np.ascontiguousarray(f)
    D_1r = np.ascontiguousarray(D_1r)
    D_2r = np.ascontiguousarray(D_2r)
    D_3r = np.ascontiguousarray(D_3r)
    prefactor = np.ascontiguousarray(prefactor)
    ST_V = np.ascontiguousarray(ST_V)
    ST_D1V = np.ascontiguousarray(ST_D1V)
    B_r = np.ascontiguousarray(B_r)

    return (- alpha * c * D_1r - gamma * D_3r + nu * D_2r) @ a + B_r @ f


@njit
def TI_primal_PODG_FOTR_kdv_expl(a, f0, D_1r, D_2r, D_3r, prefactor, ST_V, ST_D1V, B_r, c, alpha, omega, gamma, nu, Nt,
                                 dt):
    # Time loop
    as_p = np.zeros((a.shape[0], Nt))
    as_p[:, 0] = a

    for n in range(1, Nt):
        as_p[:, n] = rk4_PODG_prim_kdvb(RHS_primal_PODG_FOTR_kdv_expl, as_p[:, n - 1], f0[:, n - 1], f0[:, n], dt,
                                        D_1r, D_2r, D_3r, prefactor, ST_V, ST_D1V, B_r, c, alpha, omega, gamma, nu)

    print('RK4 reduced primal finished')

    return as_p


# Nonlinear Jacobian for KdV steepening term
def J_nl_PODG_FOTR_kdv(a: np.ndarray,
                       prefactor: np.ndarray,
                       ST_D1V: np.ndarray,
                       ST_V: np.ndarray,
                       omega: float,
                       dt: float) -> np.ndarray:
    return 3.0 * omega * dt * (prefactor @ (ST_D1V.dot(a)[:, None] * ST_V + ST_V.dot(a)[:, None] * ST_D1V))


# Right-hand side for KdV-Burgers-advection
def RHS_primal_PODG_FOTR_kdv_impl(a: np.ndarray,
                                  u: np.ndarray,
                                  D_1r: np.ndarray,
                                  D_2r: np.ndarray,
                                  D_3r: np.ndarray,
                                  prefactor: np.ndarray,
                                  ST_V: np.ndarray,
                                  ST_D1V: np.ndarray,
                                  B_r: np.ndarray,
                                  c: float,
                                  alpha: float,
                                  omega: float,
                                  gamma: float,
                                  nu: float) -> np.ndarray:
    return (- alpha * c * D_1r - gamma * D_3r + nu * D_2r) @ a + B_r @ u


def TI_primal_PODG_FOTR_kdv_impl(a, f, primal_mat, J_l, Nx, Nt, dt, **params_primal):
    # Time loop
    as_p = np.zeros((a.shape[0], Nt))
    as_p[:, 0] = a
    for n in range(1, Nt):
        as_p[:, n] = implicit_midpoint_PODG_FOTR_primal_kdvb(RHS_primal_PODG_FOTR_kdv_impl,
                                                             J_nl_PODG_FOTR_kdv,
                                                             as_p[:, n - 1],
                                                             f[:, n - 1],
                                                             f[:, n],
                                                             J_l,
                                                             primal_mat,
                                                             dt,
                                                             **params_primal)
    print('Implicit midpoint reduced primal finished')
    return as_p


@njit
def IC_adjoint_PODG_FOTR_kdv(modes):
    return np.zeros(modes)


def mat_adjoint_PODG_FOTR_kdv(V_a, V_deim_a, V_p, B, qs_target, **params_adjoint) -> AdjointMatrices_PODG_FOTR_kdv:
    D1, D2, D3 = params_adjoint['D1'], params_adjoint['D2'], params_adjoint['D3']
    CTC = params_adjoint['CTC']
    Nm_deim_a = V_deim_a.shape[1]

    # Linear factors
    D_1r = (V_a.T @ D1.T) @ V_a
    D_2r = (V_a.T @ D2.T) @ V_a
    D_3r = (V_a.T @ D3.T) @ V_a
    VaT_Vp = V_a[CTC, :].T @ V_p[CTC, :]
    VaT_CTC_qT = V_a[CTC, :].T @ qs_target[CTC, :]
    B_r = V_a.T @ B

    # DEIM prep
    _, _, piv = qr(V_deim_a.T, pivoting=True)
    S = np.sort(piv[:Nm_deim_a])
    STVdeim_inv = np.linalg.inv(V_deim_a[S])
    VaT_Vdeim_ST_Vdeim_inv = (V_a.T @ V_deim_a) @ STVdeim_inv

    ST_Va = V_a[S]
    ST_D1Va = (D1 @ V_a)[S]
    ST_Vp = V_p[S]
    ST_D1Vp = (D1 @ V_p)[S]

    return AdjointMatrices_PODG_FOTR_kdv(
        D_1r=D_1r,
        D_2r=D_2r,
        D_3r=D_3r,
        prefactor=VaT_Vdeim_ST_Vdeim_inv,
        ST_Va=ST_Va,
        ST_D1Va=ST_D1Va,
        ST_Vp=ST_Vp,
        ST_D1Vp=ST_D1Vp,
        VaT_Vp=VaT_Vp,
        VaT_CTC_qT=VaT_CTC_qT,
        VaT_B=B_r
    )


@njit
def RHS_adjoint_PODG_FOTR_kdv_expl(a_adj, a, VaT_CTC_qT, D_1r, D_2r, D_3r, prefactor, ST_Vp, ST_Va,
                                   ST_D1Vp, ST_D1Va, VaT_Vp, c, alpha, omega, gamma, nu, dx):
    a_adj = np.ascontiguousarray(a_adj)
    a = np.ascontiguousarray(a)
    D_1r = np.ascontiguousarray(D_1r)
    D_2r = np.ascontiguousarray(D_2r)
    D_3r = np.ascontiguousarray(D_3r)
    prefactor = np.ascontiguousarray(prefactor)
    ST_Vp = np.ascontiguousarray(ST_Vp)
    ST_Va = np.ascontiguousarray(ST_Va)
    ST_D1Vp = np.ascontiguousarray(ST_D1Vp)
    ST_D1Va = np.ascontiguousarray(ST_D1Va)
    VaT_Vp = np.ascontiguousarray(VaT_Vp)
    VaT_CTC_qT = np.ascontiguousarray(VaT_CTC_qT)

    out = (alpha * c * D_1r - gamma * D_3r + nu * D_2r) @ a_adj - dx * (VaT_Vp @ a - VaT_CTC_qT)

    return out


@njit
def TI_adjoint_PODG_FOTR_kdv_expl(a_adj, as_p, VaT_CTC_qT, D_1r, D_2r, D_3r, prefactor, ST_Vp, ST_Va,
                                  ST_D1Vp, ST_D1Va, VaT_Vp, c, alpha, omega, gamma, nu,
                                  dx, Nt, dt):
    as_adj = np.zeros((a_adj.shape[0], Nt))
    as_adj[:, -1] = a_adj

    for n in range(1, Nt):
        as_adj[:, -(n + 1)] = rk4_PODG_adj_kdvb_(RHS_adjoint_PODG_FOTR_kdv_expl, as_adj[:, -n], as_p[:, -n],
                                                 as_p[:, -(n + 1)],
                                                 VaT_CTC_qT[:, -n], VaT_CTC_qT[:, -(n + 1)],
                                                 - dt, D_1r, D_2r, D_3r,
                                                 prefactor, ST_Vp, ST_Va, ST_D1Vp, ST_D1Va, VaT_Vp,
                                                 c, alpha, omega, gamma, nu, dx)

    print('RK4 reduced adjoint finished')
    return as_adj


def J_nl_PODG_FOTR_adj_kdv(a: np.ndarray,
                           prefactor: np.ndarray,
                           ST_D1Va: np.ndarray,
                           ST_Va: np.ndarray,
                           ST_D1Vp: np.ndarray,
                           ST_Vp: np.ndarray,
                           omega: float,
                           dt: float) -> np.ndarray:
    return 3.0 * omega * dt * (prefactor @ (ST_D1Vp.dot(a)[:, None] * ST_Va + ST_Vp.dot(a)[:, None] * ST_D1Va))


def RHS_adjoint_PODG_FOTR_kdv_impl(a_adj: np.ndarray,
                                   a: np.ndarray,
                                   VaT_CTC_qT: np.ndarray,
                                   dx: float,
                                   D_1r: np.ndarray,
                                   D_2r: np.ndarray,
                                   D_3r: np.ndarray,
                                   prefactor: np.ndarray,
                                   ST_Va: np.ndarray,
                                   ST_D1Va: np.ndarray,
                                   ST_Vp: np.ndarray,
                                   ST_D1Vp: np.ndarray,
                                   VaT_Vp: np.ndarray,
                                   c: float,
                                   alpha: float,
                                   omega: float,
                                   gamma: float,
                                   nu: float) -> np.ndarray:
    out = (alpha * c * D_1r - gamma * D_3r + nu * D_2r) @ a_adj - dx * (VaT_Vp @ a - VaT_CTC_qT)

    return out


def TI_adjoint_PODG_FOTR_kdv_impl(a_adj, as_p, adjoint_mat, J_l, Nx, Nt, dx, dt, **params_adjoint):
    # Time loop
    as_adj = np.zeros((a_adj.shape[0], Nt))
    as_adj[:, -1] = a_adj
    VaT_CTC_qT = adjoint_mat.VaT_CTC_qT

    for n in range(1, Nt):
        as_adj[:, -(n + 1)] = implicit_midpoint_PODG_FOTR_adjoint_kdvb(RHS_adjoint_PODG_FOTR_kdv_impl,
                                                                       J_nl_PODG_FOTR_adj_kdv,
                                                                       as_adj[:, -n],
                                                                       as_p[:, -n],
                                                                       as_p[:, -(n + 1)],
                                                                       VaT_CTC_qT[:, -n],
                                                                       VaT_CTC_qT[:, -(n + 1)],
                                                                       J_l,
                                                                       dx,
                                                                       dt,
                                                                       adjoint_mat,
                                                                       **params_adjoint)
    print('Implicit midpoint reduced adjoint finished')

    return as_adj


def mat_adjoint_PODG_FOTR_kdv_mix(V_a, V_deim_a, Vd_p, B, qs_target, samples, modes_a, modes_p,
                                  **params_adjoint) -> AdjointMatrices_PODG_FOTR_kdv:
    D1, D2, D3 = params_adjoint['D1'], params_adjoint['D2'], params_adjoint['D3']
    CTC = params_adjoint['CTC']

    # Linear factors
    D_1r = (V_a.T @ D1.T) @ V_a
    D_2r = (V_a.T @ D2.T) @ V_a
    D_3r = (V_a.T @ D3.T) @ V_a
    VaT_CTC_qT = V_a[CTC, :].T @ qs_target[CTC, :]
    B_r = V_a.T @ B

    VaT_Vdp = Target_offline_adjoint_FOTR_mix(Vd_p, V_a.T, CTC, samples, modes_a, modes_p)

    return AdjointMatrices_PODG_FOTR_kdv(
        D_1r=D_1r,
        D_2r=D_2r,
        D_3r=D_3r,
        prefactor=D_1r,
        ST_Va=D_1r,
        ST_D1Va=D_1r,
        ST_Vp=D_1r,
        ST_D1Vp=D_1r,
        VaT_Vp=VaT_Vdp,
        VaT_CTC_qT=VaT_CTC_qT,
        VaT_B=B_r
    )


@njit
def RHS_adjoint_PODG_FOTR_kdv_expl_mix(a_adj, a, VaT_CTC_qT, D_1r, D_2r, D_3r, prefactor, ST_Vp, ST_Va,
                                       ST_D1Vp, ST_D1Va, VaT_Vp, ds, c, alpha, omega, gamma, nu, dx):
    a_adj = np.ascontiguousarray(a_adj)
    D_1r = np.ascontiguousarray(D_1r)
    D_2r = np.ascontiguousarray(D_2r)
    D_3r = np.ascontiguousarray(D_3r)
    prefactor = np.ascontiguousarray(prefactor)
    ST_Vp = np.ascontiguousarray(ST_Vp)
    ST_Va = np.ascontiguousarray(ST_Va)
    ST_D1Vp = np.ascontiguousarray(ST_D1Vp)
    ST_D1Va = np.ascontiguousarray(ST_D1Va)
    VaT_CTC_qT = np.ascontiguousarray(VaT_CTC_qT)

    as_p = a[:-1]
    z_p = a[-1]
    intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], -z_p)
    V_aTVd_p = np.add(weight * VaT_Vp[intervalIdx], (1 - weight) * VaT_Vp[intervalIdx + 1])

    out = (alpha * c * D_1r - gamma * D_3r + nu * D_2r) @ a_adj - dx * (V_aTVd_p @ as_p - VaT_CTC_qT)

    return out


@njit
def TI_adjoint_PODG_FOTR_kdv_expl_mix(a_adj, as_p, VaT_CTC_qT, D_1r, D_2r, D_3r, prefactor, ST_Vp, ST_Va,
                                      ST_D1Vp, ST_D1Va, VaT_Vp, delta_s, c, alpha, omega, gamma, nu,
                                      dx, Nt, dt):
    as_adj = np.zeros((a_adj.shape[0], Nt))
    as_adj[:, -1] = a_adj

    for n in range(1, Nt):
        as_adj[:, -(n + 1)] = rk4_PODG_adj_kdvb__(RHS_adjoint_PODG_FOTR_kdv_expl_mix, as_adj[:, -n], as_p[:, -n],
                                                  as_p[:, -(n + 1)],
                                                  VaT_CTC_qT[:, -n], VaT_CTC_qT[:, -(n + 1)],
                                                  - dt, D_1r, D_2r, D_3r,
                                                  prefactor, ST_Vp, ST_Va, ST_D1Vp, ST_D1Va, VaT_Vp, delta_s,
                                                  c, alpha, omega, gamma, nu, dx)

    print('RK4 reduced adjoint finished')
    return as_adj


#############
## FRTO POD
#############


class AdjointMatrices_PODG_FRTO_kdv(NamedTuple):
    VT_CTC_qT: np.ndarray


@njit
def IC_primal_PODG_FRTO_kdv(V, q0):
    return V.T @ q0


def mat_primal_PODG_FRTO_kdv(V, V_deim, **params_primal) -> PrimalMatrices_PODG_kdv:
    D1, D2, D3 = params_primal['D1'], params_primal['D2'], params_primal['D3']
    B = params_primal['B']
    Nm_deim = V_deim.shape[1]

    # Linear factors
    D_1r = (V.T @ D1) @ V
    D_2r = (V.T @ D2) @ V
    D_3r = (V.T @ D3) @ V
    B_r = V.T @ B

    # Nonlinear DEIM prep
    _, _, piv = qr(V_deim.T, pivoting=True)
    S = np.sort(piv[:Nm_deim])
    STVdeim_inv = np.linalg.inv(V_deim[S])
    VT_Vdeim_ST_Vdeim_inv = (V.T @ V_deim) @ STVdeim_inv

    ST_V = V[S]
    ST_D1V = (D1 @ V)[S]

    return PrimalMatrices_PODG_kdv(
        D_1r=D_1r,
        D_2r=D_2r,
        D_3r=D_3r,
        B_r=B_r,
        prefactor=VT_Vdeim_ST_Vdeim_inv,
        ST_V=ST_V,
        ST_D1V=ST_D1V
    )


@njit
def RHS_primal_PODG_FRTO_kdv_expl(a, f, D_1r, D_2r, D_3r, prefactor, ST_V, ST_D1V, B_r, c, alpha, omega, gamma, nu):
    a = np.ascontiguousarray(a)
    f = np.ascontiguousarray(f)
    D_1r = np.ascontiguousarray(D_1r)
    D_2r = np.ascontiguousarray(D_2r)
    D_3r = np.ascontiguousarray(D_3r)
    prefactor = np.ascontiguousarray(prefactor)
    ST_V = np.ascontiguousarray(ST_V)
    ST_D1V = np.ascontiguousarray(ST_D1V)
    B_r = np.ascontiguousarray(B_r)

    return (- alpha * c * D_1r - gamma * D_3r + nu * D_2r) @ a + B_r @ f


@njit
def TI_primal_PODG_FRTO_kdv_expl(a, f0, D_1r, D_2r, D_3r, prefactor, ST_V, ST_D1V, B_r, c, alpha, omega, gamma, nu, Nt,
                                 dt):
    # Time loop
    as_p = np.zeros((a.shape[0], Nt))
    as_p[:, 0] = a

    for n in range(1, Nt):
        as_p[:, n] = rk4_PODG_prim_kdvb(RHS_primal_PODG_FRTO_kdv_expl, as_p[:, n - 1], f0[:, n - 1], f0[:, n], dt,
                                        D_1r, D_2r, D_3r, prefactor, ST_V, ST_D1V, B_r, c, alpha, omega, gamma, nu)

    print('RK4 reduced primal finished')

    return as_p


# Nonlinear Jacobian for KdV steepening term
def J_nl_PODG_FRTO_kdv(a: np.ndarray,
                       prefactor: np.ndarray,
                       ST_D1V: np.ndarray,
                       ST_V: np.ndarray,
                       omega: float,
                       dt: float) -> np.ndarray:
    return 3.0 * omega * dt * (prefactor @ (ST_D1V.dot(a)[:, None] * ST_V + ST_V.dot(a)[:, None] * ST_D1V))


# Right-hand side for KdV-Burgers-advection
def RHS_primal_PODG_FRTO_kdv_impl(a: np.ndarray,
                                  u: np.ndarray,
                                  D_1r: np.ndarray,
                                  D_2r: np.ndarray,
                                  D_3r: np.ndarray,
                                  prefactor: np.ndarray,
                                  ST_V: np.ndarray,
                                  ST_D1V: np.ndarray,
                                  B_r: np.ndarray,
                                  c: float,
                                  alpha: float,
                                  omega: float,
                                  gamma: float,
                                  nu: float) -> np.ndarray:
    return (- alpha * c * D_1r - gamma * D_3r + nu * D_2r) @ a + B_r @ u


def TI_primal_PODG_FRTO_kdv_impl(a, f, primal_mat, J_l, Nx, Nt, dt, **params_primal):
    # Time loop
    as_p = np.zeros((a.shape[0], Nt))
    as_p[:, 0] = a
    for n in range(1, Nt):
        as_p[:, n] = implicit_midpoint_PODG_FRTO_primal_kdvb(RHS_primal_PODG_FRTO_kdv_impl,
                                                             J_nl_PODG_FRTO_kdv,
                                                             as_p[:, n - 1],
                                                             f[:, n - 1],
                                                             f[:, n],
                                                             J_l,
                                                             primal_mat,
                                                             dt,
                                                             **params_primal)
    print('Implicit midpoint reduced primal finished')
    return as_p


@njit
def IC_adjoint_PODG_FRTO_kdv(modes):
    return np.zeros(modes)


def mat_adjoint_PODG_FRTO_kdv(V, CTC, qs_target) -> AdjointMatrices_PODG_FRTO_kdv:
    return AdjointMatrices_PODG_FRTO_kdv(
        VT_CTC_qT=V[CTC, :].T @ qs_target[CTC, :]
    )


@njit
def RHS_adjoint_PODG_FRTO_kdv_expl(a_adj, a, VT_CTC_qT, D_1r, D_2r, D_3r, prefactor, ST_V, ST_D1V, c, alpha, omega,
                                   gamma,
                                   nu, dx):
    a_adj = np.ascontiguousarray(a_adj)
    a = np.ascontiguousarray(a)
    D_1r = np.ascontiguousarray(D_1r)
    D_2r = np.ascontiguousarray(D_2r)
    D_3r = np.ascontiguousarray(D_3r)
    prefactor = np.ascontiguousarray(prefactor)
    ST_V = np.ascontiguousarray(ST_V)
    ST_D1V = np.ascontiguousarray(ST_D1V)
    VT_CTC_qT = np.ascontiguousarray(VT_CTC_qT)

    out = (alpha * c * D_1r.T + gamma * D_3r.T - nu * D_2r.T) @ a_adj - dx * (a - VT_CTC_qT)

    return out


@njit
def TI_adjoint_PODG_FRTO_kdv_expl(a_adj, as_p, VT_CTC_qT, D_1r, D_2r, D_3r, prefactor, ST_V, ST_D1V, c, alpha, omega,
                                  gamma,
                                  nu,
                                  dx, Nt, dt):
    as_adj = np.zeros((a_adj.shape[0], Nt))
    as_adj[:, -1] = a_adj

    for n in range(1, Nt):
        as_adj[:, -(n + 1)] = rk4_PODG_adj_kdvb(RHS_adjoint_PODG_FRTO_kdv_expl, as_adj[:, -n], as_p[:, -n],
                                                as_p[:, -(n + 1)],
                                                VT_CTC_qT[:, -n], VT_CTC_qT[:, -(n + 1)], - dt, D_1r, D_2r, D_3r,
                                                prefactor,
                                                ST_V, ST_D1V, c, alpha, omega, gamma, nu, dx)

    print('RK4 reduced adjoint finished')
    return as_adj


def RHS_adjoint_PODG_FRTO_kdv_impl(a_adj: np.ndarray,
                                   a: np.ndarray,
                                   VT_CTC_qT: np.ndarray,
                                   dx: float,
                                   D_1r: np.ndarray,
                                   D_2r: np.ndarray,
                                   D_3r: np.ndarray,
                                   prefactor: np.ndarray,
                                   ST_V: np.ndarray,
                                   ST_D1V: np.ndarray,
                                   c: float,
                                   alpha: float,
                                   omega: float,
                                   gamma: float,
                                   nu: float) -> np.ndarray:
    out = (alpha * c * D_1r.T + gamma * D_3r.T - nu * D_2r.T) @ a_adj - dx * (a - VT_CTC_qT)

    return out


def TI_adjoint_PODG_FRTO_kdv_impl(a_adj, as_p, adjoint_mat, primal_mat, J_l, Nx, Nt, dx, dt, **params_primal):
    # Time loop
    as_adj = np.zeros((a_adj.shape[0], Nt))
    as_adj[:, -1] = a_adj
    VT_CTC_qT = adjoint_mat.VT_CTC_qT

    for n in range(1, Nt):
        as_adj[:, -(n + 1)] = implicit_midpoint_PODG_FRTO_adjoint_kdvb(RHS_adjoint_PODG_FRTO_kdv_impl,
                                                                       J_nl_PODG_FRTO_kdv,
                                                                       as_adj[:, -n],
                                                                       as_p[:, -n],
                                                                       as_p[:, -(n + 1)],
                                                                       VT_CTC_qT[:, -n],
                                                                       VT_CTC_qT[:, -(n + 1)],
                                                                       J_l,
                                                                       dx,
                                                                       dt,
                                                                       primal_mat,
                                                                       **params_primal)
    print('Implicit midpoint reduced adjoint finished')

    return as_adj
