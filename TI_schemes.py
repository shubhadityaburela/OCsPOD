import scipy
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import factorized

from Helper import *


def rk4_FOM(RHS: callable,
            q0: np.ndarray,
            u1: np.ndarray,
            u2: np.ndarray,
            dt,
            A,
            psi) -> np.ndarray:
    u_mid = (u1 + u2) / 2

    k1 = RHS(q0, u1, A, psi)
    k2 = RHS(q0 + dt / 2 * k1, u_mid, A, psi)
    k3 = RHS(q0 + dt / 2 * k2, u_mid, A, psi)
    k4 = RHS(q0 + dt * k3, u2, A, psi)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1


def rk4_FOM_targ(RHS: callable,
                 q0: np.ndarray,
                 dt,
                 Grad,
                 v_x_t) -> np.ndarray:
    k1 = RHS(q0, Grad, v_x_t)
    k2 = RHS(q0 + dt / 2 * k1, Grad, v_x_t)
    k3 = RHS(q0 + dt / 2 * k2, Grad, v_x_t)
    k4 = RHS(q0 + dt * k3, Grad, v_x_t)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1


def rk4_FOM_adj(RHS: callable,
                q0: np.ndarray,
                a1: np.ndarray,
                a2: np.ndarray,
                b1: np.ndarray,
                b2: np.ndarray,
                dt,
                A,
                CTC,
                dx) -> np.ndarray:
    a_mid = (a1 + a2) / 2
    b_mid = (b1 + b2) / 2

    k1 = RHS(q0, a1, b1, A, CTC, dx)
    k2 = RHS(q0 + dt / 2 * k1, a_mid, b_mid, A, CTC, dx)
    k3 = RHS(q0 + dt / 2 * k2, a_mid, b_mid, A, CTC, dx)
    k4 = RHS(q0 + dt * k3, a2, b2, A, CTC, dx)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1


def implicit_midpoint_FOM_adj(RHS: callable,
                              q0: np.ndarray,
                              a1: np.ndarray,
                              a2: np.ndarray,
                              b1: np.ndarray,
                              b2: np.ndarray,
                              dt,
                              M,
                              A,
                              LU_M,
                              CTC,
                              Nx,
                              dx,
                              scheme):
    q_mid = (a1 + a2) / 2
    qt_mid = (b1 + b2) / 2

    q1 = RHS(q0, q_mid, qt_mid, M, A, LU_M, CTC, Nx, dx, dt, scheme)

    return q1


def DIRK_FOM_adj(RHS: callable,
                 q0: np.ndarray,
                 a1: np.ndarray,
                 a2: np.ndarray,
                 b1: np.ndarray,
                 b2: np.ndarray,
                 dt,
                 M,
                 A,
                 LU_M,
                 CTC,
                 Nx,
                 dx,
                 scheme):
    a_threefourth = 0.75 * a1 + 0.25 * a2
    a_onefourth = 0.25 * a1 + 0.75 * a2
    b_threefourth = 0.75 * b1 + 0.25 * b2
    b_onefourth = 0.25 * b1 + 0.75 * b2

    k1 = RHS(q0, a_threefourth, b_threefourth, M, A, LU_M, CTC, Nx, dx, dt, scheme)
    k2 = RHS(q0 + dt / 2 * k1, a_onefourth, b_onefourth, M, A, LU_M, CTC, Nx, dx, dt, scheme)

    q1 = q0 + dt / 2 * (k1 + k2)

    return q1


def F_start_FOM(RHS: callable,
                p1234: np.ndarray,
                p0: np.ndarray,
                q: np.ndarray,
                qt: np.ndarray,
                A,
                CTC,
                Nx,
                dx,
                dt):
    # the magic coefficients in this function come from a polynomial approach
    # the approach calculates 4 timesteps at once and is of order 4.
    # for details, see here: https://colab.research.google.com/drive/1DHtzD3U1PsMQbn-nsBF3crh_Tj3cgmMd?usp=sharing

    p1 = p1234[:Nx]
    p2 = p1234[Nx:2 * Nx]
    p3 = p1234[2 * Nx:3 * Nx]
    p4 = p1234[3 * Nx:]

    # entries of F
    pprime_t1 = -3.0 * p0 - 10.0 * p1 + 18.0 * p2 - 6.0 * p3 + p4
    pprime_t2 = p0 - 8.0 * p1 + 8.0 * p3 - 1.0 * p4
    pprime_t3 = -1.0 * p0 + 6.0 * p1 - 18.0 * p2 + 10.0 * p3 + 3.0 * p4
    pprime_t4 = 3.0 * p0 - 16.0 * p1 + 36.0 * p2 - 48.0 * p3 + 25.0 * p4

    return np.hstack((
        pprime_t1 - 12 * dt * RHS(p1, q[:, -2], qt[:, -2], A, CTC, dx),
        pprime_t2 - 12 * dt * RHS(p2, q[:, -3], qt[:, -3], A, CTC, dx),
        pprime_t3 - 12 * dt * RHS(p3, q[:, -4], qt[:, -4], A, CTC, dx),
        pprime_t4 - 12 * dt * RHS(p4, q[:, -5], qt[:, -5], A, CTC, dx)
    ))


def DF_start_FOM(A, Nx, dt):
    # identity matrix
    eye = np.eye(Nx)

    # the magic coefficients in this function come from a polynomial approach
    # the approach calculates 4 timesteps at once and is of order 4.
    # for details, see here: https://colab.research.google.com/drive/1DHtzD3U1PsMQbn-nsBF3crh_Tj3cgmMd?usp=sharing

    # first row
    DF_11 = -10.0 * eye - 12 * dt * (- A)
    DF_12 = 18.0 * eye
    DF_13 = -6.0 * eye
    DF_14 = 1.0 * eye
    DF_1 = np.hstack((DF_11, DF_12, DF_13, DF_14))

    # second row
    DF_21 = -8.0 * eye
    DF_22 = 0.0 * eye - 12 * dt * (- A)
    DF_23 = 8.0 * eye
    DF_24 = -1.0 * eye
    DF_2 = np.hstack((DF_21, DF_22, DF_23, DF_24))

    # third row
    DF_31 = 6.0 * eye
    DF_32 = -18.0 * eye
    DF_33 = 10.0 * eye - 12 * dt * (- A)
    DF_34 = 3.0 * eye
    DF_3 = np.hstack((DF_31, DF_32, DF_33, DF_34))

    # fourth row
    DF_41 = -16.0 * eye
    DF_42 = 36.0 * eye
    DF_43 = -48.0 * eye
    DF_44 = 25.0 * eye - 12 * dt * (- A)
    DF_4 = np.hstack((DF_41, DF_42, DF_43, DF_44))

    # return all rows together
    return np.vstack((DF_1, DF_2, DF_3, DF_4))


def poly_interp_FOM_adj(RHS: callable,
                        p0: np.ndarray,
                        q: np.ndarray,
                        qt: np.ndarray,
                        A,
                        Df,
                        CTC,
                        Nx,
                        dx,
                        dt):
    p1234 = np.hstack((p0, p0, p0, p0))
    f = F_start_FOM(RHS, p1234, p0, q, qt, A, CTC, Nx, dx, dt)

    p_update = p1234 - scipy.sparse.linalg.spsolve(Df, f)

    p1 = p_update[:Nx]
    p2 = p_update[Nx:2 * Nx]
    p3 = p_update[2 * Nx:3 * Nx]

    return np.stack([p3, p2, p1, p0], axis=-1)


def bdf2_FOM_adj(RHS: callable,
                 p: np.ndarray,
                 q: np.ndarray,
                 qt: np.ndarray,
                 dt,
                 M,
                 A,
                 LU_M,
                 CTC,
                 Nx,
                 dx,
                 scheme,
                 n):
    p_past = np.stack([p[:, -(n - 1)], p[:, -n]])

    q1 = RHS(p_past, q, qt, M, A, LU_M, CTC, Nx, dx, dt, scheme)

    return q1


def bdf3_FOM_adj(RHS: callable,
                 p: np.ndarray,
                 q: np.ndarray,
                 qt: np.ndarray,
                 dt,
                 M,
                 A,
                 LU_M,
                 CTC,
                 Nx,
                 dx,
                 scheme,
                 n):
    p_past = np.stack([p[:, -(n - 2)], p[:, -(n - 1)], p[:, -n]])

    q1 = RHS(p_past, q, qt, M, A, LU_M, CTC, Nx, dx, dt, scheme)

    return q1


def bdf4_FOM_adj(RHS: callable,
                 p: np.ndarray,
                 q: np.ndarray,
                 qt: np.ndarray,
                 dt,
                 M,
                 A,
                 LU_M,
                 CTC,
                 Nx,
                 dx,
                 scheme,
                 n):
    p_past = np.stack([p[:, -(n - 3)], p[:, -(n - 2)], p[:, -(n - 1)], p[:, -n]])
    q1 = RHS(p_past, q, qt, M, A, LU_M, CTC, Nx, dx, dt, scheme)

    return q1


#####################################################################################

@njit
def rk4_PODG_prim(RHS: callable,
                  q0: np.ndarray,
                  u1: np.ndarray,
                  u2: np.ndarray,
                  dt,
                  Ar_p,
                  psir_p) -> np.ndarray:
    u_mid = (u1 + u2) / 2

    k1 = RHS(q0, u1, Ar_p, psir_p)
    k2 = RHS(q0 + dt / 2 * k1, u_mid, Ar_p, psir_p)
    k3 = RHS(q0 + dt / 2 * k2, u_mid, Ar_p, psir_p)
    k4 = RHS(q0 + dt * k3, u2, Ar_p, psir_p)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1


@njit
def rk4_PODG_adj(RHS: callable,
                 q0: np.ndarray,
                 a1: np.ndarray,
                 a2: np.ndarray,
                 b1: np.ndarray,
                 b2: np.ndarray,
                 dt,
                 Ar_a,
                 dx) -> np.ndarray:
    a_mid = (a1 + a2) / 2
    b_mid = (b1 + b2) / 2
    k1 = RHS(q0, a1, b1, Ar_a, dx)
    k2 = RHS(q0 + dt / 2 * k1, a_mid, b_mid, Ar_a, dx)
    k3 = RHS(q0 + dt / 2 * k2, a_mid, b_mid, Ar_a, dx)
    k4 = RHS(q0 + dt * k3, a2, b2, Ar_a, dx)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1


def implicit_midpoint_PODG_adj(RHS: callable,
                               q0: np.ndarray,
                               a1: np.ndarray,
                               a2: np.ndarray,
                               b1: np.ndarray,
                               b2: np.ndarray,
                               dt,
                               M,
                               A,
                               LU_M,
                               Nx,
                               dx,
                               scheme):
    q_mid = (a1 + a2) / 2
    qt_mid = (b1 + b2) / 2

    q1 = RHS(q0, q_mid, qt_mid, M, A, LU_M, Nx, dx, dt, scheme)

    return q1


def DIRK_PODG_adj(RHS: callable,
                  q0: np.ndarray,
                  a1: np.ndarray,
                  a2: np.ndarray,
                  b1: np.ndarray,
                  b2: np.ndarray,
                  dt,
                  M,
                  A,
                  LU_M,
                  Nx,
                  dx,
                  scheme):
    a_threefourth = 0.75 * a1 + 0.25 * a2
    a_onefourth = 0.25 * a1 + 0.75 * a2
    b_threefourth = 0.75 * b1 + 0.25 * b2
    b_onefourth = 0.25 * b1 + 0.75 * b2

    k1 = RHS(q0, a_threefourth, b_threefourth, M, A, LU_M, Nx, dx, dt, scheme)
    k2 = RHS(q0 + dt / 2 * k1, a_onefourth, b_onefourth, M, A, LU_M, Nx, dx, dt, scheme)

    q1 = q0 + dt / 2 * (k1 + k2)

    return q1


def bdf2_PODG_adj(RHS: callable,
                  p: np.ndarray,
                  q: np.ndarray,
                  qt: np.ndarray,
                  dt,
                  M,
                  A,
                  LU_M,
                  Nx,
                  dx,
                  scheme,
                  n):
    p_past = np.stack([p[:, -(n - 1)], p[:, -n]])

    q1 = RHS(p_past, q, qt, M, A, LU_M, Nx, dx, dt, scheme)

    return q1


@njit
def rk4_PODG_adj_(RHS: callable,
                  q0: np.ndarray,
                  a1: np.ndarray,
                  a2: np.ndarray,
                  b1: np.ndarray,
                  b2: np.ndarray,
                  dt,
                  Ar_a,
                  V_aTV_p,
                  dx) -> np.ndarray:
    a_mid = (a1 + a2) / 2
    b_mid = (b1 + b2) / 2

    k1 = RHS(q0, a1, b1, Ar_a, V_aTV_p, dx)
    k2 = RHS(q0 + dt / 2 * k1, a_mid, b_mid, Ar_a, V_aTV_p, dx)
    k3 = RHS(q0 + dt / 2 * k2, a_mid, b_mid, Ar_a, V_aTV_p, dx)
    k4 = RHS(q0 + dt * k3, a2, b2, Ar_a, V_aTV_p, dx)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1


@njit
def rk4_PODG_adj__(RHS: callable,
                   q0: np.ndarray,
                   a1: np.ndarray,
                   a2: np.ndarray,
                   b1: np.ndarray,
                   b2: np.ndarray,
                   dt,
                   ds,
                   Ar_a,
                   V_aTVd_p,
                   dx) -> np.ndarray:
    a_mid = (a1 + a2) / 2
    b_mid = (b1 + b2) / 2

    k1 = RHS(q0, a1, b1, ds, Ar_a, V_aTVd_p, dx)
    k2 = RHS(q0 + dt / 2 * k1, a_mid, b_mid, ds, Ar_a, V_aTVd_p, dx)
    k3 = RHS(q0 + dt / 2 * k2, a_mid, b_mid, ds, Ar_a, V_aTVd_p, dx)
    k4 = RHS(q0 + dt * k3, a2, b2, ds, Ar_a, V_aTVd_p, dx)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1


def implicit_midpoint_PODG_adj_(RHS: callable,
                                q0: np.ndarray,
                                a1: np.ndarray,
                                a2: np.ndarray,
                                b1: np.ndarray,
                                b2: np.ndarray,
                                dt,
                                M,
                                A,
                                LU_M,
                                V_aTV_p,
                                dx,
                                scheme):
    q_mid = (a1 + a2) / 2
    qt_mid = (b1 + b2) / 2

    q1 = RHS(q0, q_mid, qt_mid, M, A, LU_M, V_aTV_p, dx, dt, scheme)

    return q1


def DIRK_PODG_adj_(RHS: callable,
                   q0: np.ndarray,
                   a1: np.ndarray,
                   a2: np.ndarray,
                   b1: np.ndarray,
                   b2: np.ndarray,
                   dt,
                   M,
                   A,
                   LU_M,
                   V_aTV_p,
                   dx,
                   scheme):
    a_threefourth = 0.75 * a1 + 0.25 * a2
    a_onefourth = 0.25 * a1 + 0.75 * a2
    b_threefourth = 0.75 * b1 + 0.25 * b2
    b_onefourth = 0.25 * b1 + 0.75 * b2

    k1 = RHS(q0, a_threefourth, b_threefourth, M, A, LU_M, V_aTV_p, dx, dt, scheme)
    k2 = RHS(q0 + dt / 2 * k1, a_onefourth, b_onefourth, M, A, LU_M, V_aTV_p, dx, dt, scheme)

    q1 = q0 + dt / 2 * (k1 + k2)

    return q1


def bdf2_PODG_adj_(RHS: callable,
                   p: np.ndarray,
                   q: np.ndarray,
                   qt: np.ndarray,
                   dt,
                   M,
                   A,
                   LU_M,
                   V_aTV_p,
                   dx,
                   scheme,
                   n):
    p_past = np.stack([p[:, -(n - 1)], p[:, -n]])

    q1 = RHS(p_past, q, qt, M, A, LU_M, V_aTV_p, dx, dt, scheme)

    return q1


#####################################################################################

@njit
def rk4_sPODG_prim(RHS: callable,
                   q0: np.ndarray,
                   u1: np.ndarray,
                   u2: np.ndarray,
                   dt,
                   lhs,
                   rhs,
                   c,
                   delta_s,
                   modes):
    u_mid = (u1 + u2) / 2

    k1, i, w = RHS(q0, u1, lhs, rhs, c, delta_s, modes)
    k2, _, _ = RHS(q0 + dt / 2 * k1, u_mid, lhs, rhs, c, delta_s, modes)
    k3, _, _ = RHS(q0 + dt / 2 * k2, u_mid, lhs, rhs, c, delta_s, modes)
    k4, _, _ = RHS(q0 + dt * k3, u2, lhs, rhs, c, delta_s, modes)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    q_dot = np.zeros((5, (modes + 1)))
    q_dot[0, :] = k1
    q_dot[2, :] = 1 / 2 * (k2 + k3)
    q_dot[1, :] = (k1 + q_dot[2, :]) / 2
    q_dot[3, :] = (q_dot[2, :] + k4) / 2
    q_dot[4, :] = k4

    return q1, q_dot, i, w


@njit
def rk4_sPODG_prim_kdvb(RHS: callable,
                        q0: np.ndarray,
                        u1: np.ndarray,
                        u2: np.ndarray,
                        dt,
                        lhs,
                        rhs,
                        deim,
                        c,
                        delta_s,
                        modes):
    u_mid = (u1 + u2) / 2

    k1, i, w = RHS(q0, u1, lhs, rhs, deim, c, delta_s, modes)
    k2, _, _ = RHS(q0 + dt / 2 * k1, u_mid, lhs, rhs, deim, c, delta_s, modes)
    k3, _, _ = RHS(q0 + dt / 2 * k2, u_mid, lhs, rhs, deim, c, delta_s, modes)
    k4, _, _ = RHS(q0 + dt * k3, u2, lhs, rhs, deim, c, delta_s, modes)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    q_dot = np.zeros((5, (modes + 1)))
    q_dot[0, :] = k1
    q_dot[2, :] = 1 / 2 * (k2 + k3)  # Midpoint
    q_dot[1, :] = (k1 + q_dot[2, :]) / 2
    q_dot[3, :] = (q_dot[2, :] + k4) / 2
    q_dot[4, :] = k4

    return q1, q_dot, i, w


@njit
def rk4_sPODG_adj(RHS: callable,
                  q0: np.ndarray,
                  u1: np.ndarray,
                  u2: np.ndarray,
                  a1: np.ndarray,
                  a2: np.ndarray,
                  b1: np.ndarray,
                  b2: np.ndarray,
                  q_dot: np.ndarray,
                  dt,
                  M1, M2, N, A1, A2, C, tara, CTC,
                  Vdp, Wdp, modes, delta_s, dx):
    u_mid = (u1 + u2) / 2
    a_mid = (a1 + a2) / 2
    b_mid = (b1 + b2) / 2

    k1 = RHS(q0, u1, a1, b1, q_dot[4], M1, M2, N, A1, A2, C, tara, CTC,
             Vdp, Wdp, modes, delta_s, dx)
    k2 = RHS(q0 + dt / 2 * k1, u_mid, a_mid, b_mid, q_dot[2], M1, M2, N, A1, A2, C, tara, CTC,
             Vdp, Wdp, modes, delta_s, dx)
    k3 = RHS(q0 + dt / 2 * k2, u_mid, a_mid, b_mid, q_dot[2], M1, M2, N, A1, A2, C, tara, CTC,
             Vdp, Wdp, modes, delta_s, dx)
    k4 = RHS(q0 + dt * k3, u2, a2, b2, q_dot[0], M1, M2, N, A1, A2, C, tara, CTC,
             Vdp, Wdp, modes, delta_s, dx)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1


def implicit_midpoint_sPODG_adj(RHS: callable,
                                q0: np.ndarray,
                                u1: np.ndarray,
                                u2: np.ndarray,
                                a1: np.ndarray,
                                a2: np.ndarray,
                                b1: np.ndarray,
                                b2: np.ndarray,
                                q_dot: np.ndarray,
                                dt,
                                M1, M2, N, A1, A2, C, tara, CTC,
                                Vdp, Wdp, modes, delta_s, dx, scheme):
    u_mid = (u1 + u2) / 2
    a_mid = (a1 + a2) / 2
    b_mid = (b1 + b2) / 2

    q1 = RHS(q0, u_mid, a_mid, b_mid, q_dot[2], dt, M1, M2, N, A1, A2, C, tara, CTC,
             Vdp, Wdp, modes, delta_s, dx, scheme)

    return q1


def DIRK_sPODG_adj(RHS: callable,
                   q0: np.ndarray,
                   u1: np.ndarray,
                   u2: np.ndarray,
                   a1: np.ndarray,
                   a2: np.ndarray,
                   b1: np.ndarray,
                   b2: np.ndarray,
                   q_dot: np.ndarray,
                   dt,
                   M1, M2, N, A1, A2, C, tara, CTC,
                   Vdp, Wdp, modes, delta_s, dx, scheme):
    u_threefourth = 0.75 * u1 + 0.25 * u2
    u_onefourth = 0.25 * u1 + 0.75 * u2
    a_threefourth = 0.75 * a1 + 0.25 * a2
    a_onefourth = 0.25 * a1 + 0.75 * a2
    b_threefourth = 0.75 * b1 + 0.25 * b2
    b_onefourth = 0.25 * b1 + 0.75 * b2

    k1 = RHS(q0, u_threefourth, a_threefourth, b_threefourth, q_dot[3], dt, M1, M2, N, A1, A2, C, tara, CTC,
             Vdp, Wdp, modes, delta_s, dx, scheme)
    k2 = RHS(q0 + dt / 2 * k1, u_onefourth, a_onefourth, b_onefourth, q_dot[1], dt, M1, M2, N, A1, A2, C, tara, CTC,
             Vdp, Wdp, modes, delta_s, dx, scheme)

    q1 = q0 + dt / 2 * (k1 + k2)

    return q1


def bdf2_sPODG_adj(RHS: callable,
                   p: np.ndarray,
                   u: np.ndarray,
                   q: np.ndarray,
                   qt: np.ndarray,
                   q_dot: np.ndarray,
                   dt,
                   M1, M2, N, A1, A2, C, tara, CTC,
                   Vdp, Wdp, modes, delta_s,
                   dx, scheme, n):
    p_past = np.stack([p[:, -(n - 1)], p[:, -n]])

    q1 = RHS(p_past, u, q, qt, q_dot[4], dt, M1, M2, N, A1, A2, C, tara, CTC,
             Vdp, Wdp, modes, delta_s, dx, scheme)

    return q1


@njit
def rk4_sPODG_adj_(RHS: callable,
                   q0: np.ndarray,
                   a1: np.ndarray,
                   a2: np.ndarray,
                   b1: np.ndarray,
                   b2: np.ndarray,
                   dt,
                   lhs, rhs, tar, CTC, Vda, Wda,
                   modes_a, modes_p,
                   delta_s, dx):
    a_mid = (a1 + a2) / 2
    b_mid = (b1 + b2) / 2

    k1 = RHS(q0, a1, b1, lhs, rhs, tar, CTC, Vda, Wda, modes_a, modes_p, delta_s, dx)
    k2 = RHS(q0 + dt / 2 * k1, a_mid, b_mid, lhs, rhs, tar, CTC, Vda, Wda, modes_a, modes_p, delta_s, dx)
    k3 = RHS(q0 + dt / 2 * k2, a_mid, b_mid, lhs, rhs, tar, CTC, Vda, Wda, modes_a, modes_p, delta_s, dx)
    k4 = RHS(q0 + dt * k3, a2, b2, lhs, rhs, tar, CTC, Vda, Wda, modes_a, modes_p, delta_s, dx)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1


@njit
def rk4_sPODG_adj_kdvb(RHS: callable,
                       q0: np.ndarray,
                       a1: np.ndarray,
                       a2: np.ndarray,
                       b1: np.ndarray,
                       b2: np.ndarray,
                       dt,
                       lhs, rhs, deim, deim_mix,
                       tar, CTC, Vda, Wda,
                       modes_a, modes_p,
                       delta_s, dx):
    a_mid = (a1 + a2) / 2
    b_mid = (b1 + b2) / 2

    k1 = RHS(q0, a1, b1, lhs, rhs, deim, deim_mix, tar, CTC, Vda, Wda, modes_a, modes_p, delta_s, dx)
    k2 = RHS(q0 + dt / 2 * k1, a_mid, b_mid, lhs, rhs, deim, deim_mix, tar, CTC, Vda, Wda, modes_a, modes_p, delta_s,
             dx)
    k3 = RHS(q0 + dt / 2 * k2, a_mid, b_mid, lhs, rhs, deim, deim_mix, tar, CTC, Vda, Wda, modes_a, modes_p, delta_s,
             dx)
    k4 = RHS(q0 + dt * k3, a2, b2, lhs, rhs, deim, deim_mix, tar, CTC, Vda, Wda, modes_a, modes_p, delta_s, dx)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1


# ---------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------- #

def rk4_FOM_kdvb(RHS: callable,
                 q0: np.ndarray,
                 u1: np.ndarray,
                 u2: np.ndarray,
                 dt,
                 D1,
                 D2,
                 D3,
                 B,
                 c,
                 alpha,
                 omega,
                 gamma,
                 nu) -> np.ndarray:
    u_mid = (u1 + u2) / 2

    k1 = RHS(q0, u1, D1, D2, D3, B, c, alpha, omega, gamma, nu)
    k2 = RHS(q0 + dt / 2 * k1, u_mid, D1, D2, D3, B, c, alpha, omega, gamma, nu)
    k3 = RHS(q0 + dt / 2 * k2, u_mid, D1, D2, D3, B, c, alpha, omega, gamma, nu)
    k4 = RHS(q0 + dt * k3, u2, D1, D2, D3, B, c, alpha, omega, gamma, nu)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1


@njit
def rk4_PODG_prim_kdvb(RHS: callable,
                       q0: np.ndarray,
                       u1: np.ndarray,
                       u2: np.ndarray,
                       dt,
                       D_1r,
                       D_2r,
                       D_3r,
                       prefactor,
                       ST_V,
                       ST_D1V,
                       B_r,
                       c,
                       alpha,
                       omega,
                       gamma,
                       nu) -> np.ndarray:
    u_mid = (u1 + u2) / 2

    k1 = RHS(q0, u1, D_1r, D_2r, D_3r, prefactor, ST_V, ST_D1V, B_r, c, alpha, omega, gamma, nu)
    k2 = RHS(q0 + dt / 2 * k1, u_mid, D_1r, D_2r, D_3r, prefactor, ST_V, ST_D1V, B_r, c, alpha, omega, gamma, nu)
    k3 = RHS(q0 + dt / 2 * k2, u_mid, D_1r, D_2r, D_3r, prefactor, ST_V, ST_D1V, B_r, c, alpha, omega, gamma, nu)
    k4 = RHS(q0 + dt * k3, u2, D_1r, D_2r, D_3r, prefactor, ST_V, ST_D1V, B_r, c, alpha, omega, gamma, nu)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1


def rk4_FOM_adj_kdvb(RHS: callable,
                     q0: np.ndarray,
                     a1: np.ndarray,
                     a2: np.ndarray,
                     b1: np.ndarray,
                     b2: np.ndarray,
                     dt,
                     D1,
                     D2,
                     D3,
                     CTC,
                     dx,
                     c,
                     alpha,
                     omega,
                     gamma,
                     nu) -> np.ndarray:
    a_mid = (a1 + a2) / 2
    b_mid = (b1 + b2) / 2

    k1 = RHS(q0, a1, b1, D1, D2, D3, CTC, dx, c, alpha, omega, gamma, nu)
    k2 = RHS(q0 + dt / 2 * k1, a_mid, b_mid, D1, D2, D3, CTC, dx, c, alpha, omega, gamma, nu)
    k3 = RHS(q0 + dt / 2 * k2, a_mid, b_mid, D1, D2, D3, CTC, dx, c, alpha, omega, gamma, nu)
    k4 = RHS(q0 + dt * k3, a2, b2, D1, D2, D3, CTC, dx, c, alpha, omega, gamma, nu)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1


@njit
def rk4_PODG_adj_kdvb(RHS: callable,
                      q0: np.ndarray,
                      a1: np.ndarray,
                      a2: np.ndarray,
                      b1: np.ndarray,
                      b2: np.ndarray,
                      dt,
                      D_1r,
                      D_2r,
                      D_3r,
                      prefactor,
                      ST_V,
                      ST_D1V,
                      c,
                      alpha,
                      omega,
                      gamma,
                      nu,
                      dx) -> np.ndarray:
    a_mid = (a1 + a2) / 2
    b_mid = (b1 + b2) / 2

    k1 = RHS(q0, a1, b1, D_1r, D_2r, D_3r, prefactor, ST_V, ST_D1V, c, alpha, omega, gamma, nu, dx)
    k2 = RHS(q0 + dt / 2 * k1, a_mid, b_mid, D_1r, D_2r, D_3r, prefactor, ST_V, ST_D1V, c, alpha, omega, gamma, nu, dx)
    k3 = RHS(q0 + dt / 2 * k2, a_mid, b_mid, D_1r, D_2r, D_3r, prefactor, ST_V, ST_D1V, c, alpha, omega, gamma, nu, dx)
    k4 = RHS(q0 + dt * k3, a2, b2, D_1r, D_2r, D_3r, prefactor, ST_V, ST_D1V, c, alpha, omega, gamma, nu, dx)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1


@njit
def rk4_PODG_adj_kdvb_(RHS: callable,
                       q0: np.ndarray,
                       a1: np.ndarray,
                       a2: np.ndarray,
                       b1: np.ndarray,
                       b2: np.ndarray,
                       dt,
                       D_1r,
                       D_2r,
                       D_3r,
                       prefactor,
                       ST_Vp,
                       ST_Va,
                       ST_D1Vp,
                       ST_D1Va,
                       VaT_Vp,
                       c,
                       alpha,
                       omega,
                       gamma,
                       nu,
                       dx) -> np.ndarray:
    a_mid = (a1 + a2) / 2
    b_mid = (b1 + b2) / 2

    k1 = RHS(q0, a1, b1, D_1r, D_2r, D_3r, prefactor, ST_Vp, ST_Va, ST_D1Vp, ST_D1Va, VaT_Vp, c, alpha, omega, gamma,
             nu, dx)
    k2 = RHS(q0 + dt / 2 * k1, a_mid, b_mid, D_1r, D_2r, D_3r, prefactor, ST_Vp, ST_Va, ST_D1Vp, ST_D1Va, VaT_Vp, c,
             alpha, omega, gamma, nu, dx)
    k3 = RHS(q0 + dt / 2 * k2, a_mid, b_mid, D_1r, D_2r, D_3r, prefactor, ST_Vp, ST_Va, ST_D1Vp, ST_D1Va, VaT_Vp, c,
             alpha, omega, gamma, nu, dx)
    k4 = RHS(q0 + dt * k3, a2, b2, D_1r, D_2r, D_3r, prefactor, ST_Vp, ST_Va, ST_D1Vp, ST_D1Va, VaT_Vp, c, alpha, omega,
             gamma, nu, dx)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1


@njit
def rk4_PODG_adj_kdvb__(RHS: callable,
                        q0: np.ndarray,
                        a1: np.ndarray,
                        a2: np.ndarray,
                        b1: np.ndarray,
                        b2: np.ndarray,
                        dt,
                        D_1r,
                        D_2r,
                        D_3r,
                        prefactor,
                        ST_Vp,
                        ST_Va,
                        ST_D1Vp,
                        ST_D1Va,
                        VaT_Vp,
                        delta_s,
                        c,
                        alpha,
                        omega,
                        gamma,
                        nu,
                        dx,
                        ) -> np.ndarray:
    a_mid = (a1 + a2) / 2
    b_mid = (b1 + b2) / 2

    k1 = RHS(q0, a1, b1, D_1r, D_2r, D_3r, prefactor, ST_Vp, ST_Va, ST_D1Vp, ST_D1Va, VaT_Vp, delta_s, c, alpha, omega, gamma,
             nu, dx)
    k2 = RHS(q0 + dt / 2 * k1, a_mid, b_mid, D_1r, D_2r, D_3r, prefactor, ST_Vp, ST_Va, ST_D1Vp, ST_D1Va, VaT_Vp, delta_s, c,
             alpha, omega, gamma, nu, dx)
    k3 = RHS(q0 + dt / 2 * k2, a_mid, b_mid, D_1r, D_2r, D_3r, prefactor, ST_Vp, ST_Va, ST_D1Vp, ST_D1Va, VaT_Vp, delta_s, c,
             alpha, omega, gamma, nu, dx)
    k4 = RHS(q0 + dt * k3, a2, b2, D_1r, D_2r, D_3r, prefactor, ST_Vp, ST_Va, ST_D1Vp, ST_D1Va, VaT_Vp, delta_s, c, alpha, omega,
             gamma, nu, dx)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1


# Newton solver for implicit midpoint (forward time)
def call_newton(RHS, J_nl, q0, u1, u2,
                J_l: csc_matrix, dt, max_iter=10,
                **kwargs):
    D1 = kwargs['D1']
    omega = kwargs['omega']

    u_mid = 0.5 * (u1 + u2)

    # Initialize Newton iterate
    q_k = q0.copy()
    delq = np.inf

    # Build a function to solve linear systems with (J_l + J_nl(D, qmid, dt))
    solver = None
    itr = 0

    while np.linalg.norm(delq) / np.linalg.norm(q_k) > 1e-6:
        if itr >= max_iter:
            print(f"Warning: Newton did not converge after {max_iter} iterations.")
            break
        # Midpoint state
        qmid = 0.5 * (q0 + q_k)

        # Residual: R = q_k - q0 - dt*RHS(qmid, u_mid)
        R_k = q_k - q0 - dt * RHS(qmid, u_mid, **kwargs)

        # Assemble Jacobian: J_l + J_nl
        J_nl_k = J_nl(qmid, D1, omega, dt)
        J_k = J_l + J_nl_k

        delq = scipy.sparse.linalg.spsolve(J_k, -R_k)
        q_k += delq

        itr += 1

    return q_k


# Driver for implicit midpoint
def implicit_midpoint_FOM_primal_kdvb(
        RHS: callable,
        J_nl: callable,
        q0: np.ndarray,
        u1: np.ndarray,
        u2: np.ndarray,
        J_l: csc_matrix,
        dt: float,
        **kwargs) -> np.ndarray:
    return call_newton(RHS, J_nl, q0, u1, u2, J_l, dt, **kwargs)


def implicit_midpoint_PODG_FRTO_primal_kdvb(
        RHS: callable,
        J_nl: callable,
        q0: np.ndarray,
        u1: np.ndarray,
        u2: np.ndarray,
        J_l: np.ndarray,
        primal_mat,
        dt: float,
        max_iter=10,
        **params_primal) -> np.ndarray:
    u_mid = 0.5 * (u1 + u2)

    # Initialize Newton iterate
    q_k = q0.copy()
    delq = np.inf

    # Build a function to solve linear systems with (J_l + J_nl(D, qmid, dt, ....))
    itr = 0

    while np.linalg.norm(delq) / np.linalg.norm(q_k) > 1e-6:
        if itr >= max_iter:
            print(f"Warning: Newton did not converge after {max_iter} iterations.")
            break
        # Midpoint state
        qmid = 0.5 * (q0 + q_k)

        # Residual: R = q_k - q0 - dt*RHS(qmid, u_mid)
        R_k = q_k - q0 - dt * RHS(qmid, u_mid, primal_mat.D_1r, primal_mat.D_2r, primal_mat.D_3r,
                                  primal_mat.prefactor, primal_mat.ST_V, primal_mat.ST_D1V,
                                  primal_mat.B_r, params_primal['c'], params_primal['alpha'],
                                  params_primal['omega'], params_primal['gamma'], params_primal['nu'])

        # Assemble Jacobian: J_l + J_nl
        # J_nl_k = J_nl(qmid, primal_mat.prefactor, primal_mat.ST_D1V, primal_mat.ST_V, params_primal['omega'], dt)
        J_k = J_l  # + J_nl_k

        delq = np.linalg.solve(J_k, -R_k)
        q_k += delq

        itr += 1

    return q_k


def implicit_midpoint_sPODG_FRTO_primal_kdvb(
        RHS: callable,
        Jacobian: callable,
        q0: np.ndarray,
        u1: np.ndarray,
        u2: np.ndarray,
        LHS_matrix: np.ndarray,
        RHS_matrix: np.ndarray,
        DEIM_matrix: np.ndarray,
        DEIM_mat: np.ndarray,
        C_matrix: np.ndarray,
        delta_samples: np.ndarray,
        num_modes: int,
        dt: float,
        max_iter=10):
    u_mid = 0.5 * (u1 + u2)

    # Initialize Newton iterate
    q_k = q0.copy()
    delq = np.inf

    # Build a function to solve linear systems
    itr = 0

    while np.linalg.norm(delq) / np.linalg.norm(q_k) > 1e-6:
        if itr >= max_iter:
            print(f"Warning: Newton did not converge after {max_iter} iterations.")
            break
        # Midpoint state
        q_mid = 0.5 * (q0 + q_k)

        # Residual: R = M(q_mid) * (q_k - q0) / dt - F(qmid, u_mid)
        M, A, D, M1, M2, N, A1, A2, VT_B, WT_B, UT_B, M1_dash, M2_dash, N_dash, A1_dash, A2_dash, \
            epsilon1, epsilon2, epsilon3, epsilon1_dash, epsilon2_dash, ST_V, ST_D1V, ST_D1V_dash, STdash_V, STdash_D1V, \
            ST_U_dash, ST_U_inv, intervalIdx, weight = RHS(LHS_matrix, RHS_matrix, DEIM_matrix, DEIM_mat, C_matrix,
                                                           u_mid, q_mid, delta_samples, num_modes)
        if itr == 0:
            i = intervalIdx
            w = weight

        R_k = M @ (q_k - q0) / dt - A

        # Assemble Jacobian
        dq = q_k - q0
        J_k = Jacobian(M1, M2, N, A1, A2, D, M1_dash, M2_dash, N_dash, A1_dash, A2_dash, WT_B, UT_B, ST_V, ST_D1V,
                       ST_D1V_dash, STdash_V, STdash_D1V, ST_U_dash, ST_U_inv, epsilon1, epsilon2, epsilon3,
                       epsilon1_dash, epsilon2_dash,
                       q_mid, u_mid, dq, num_modes, dt)

        delq = np.linalg.solve(J_k, -R_k)
        q_k += delq

        itr += 1

    # Derivatives of the state variable
    q_dot = np.zeros((2, (num_modes + 1)))

    # Derivative at the midpoint half step
    q_dot[0, :] = np.linalg.solve(M, A)  # At the midpoint between x_n and x_{n + 1} -> x_{n + 1/2}

    # Derivative at full step at x_{n + 1}
    q_dot[1, :] = (q_k - q0) / dt

    # if itrrr % 1000 == 1:
    #     q_mid = 0.5 * (q0 + q_k)
    #     M, A, D, M1, M2, N, A1, A2, VT_B, WT_B, UT_B, M1_dash, M2_dash, N_dash, A1_dash, A2_dash, \
    #         epsilon1, epsilon2, epsilon3, epsilon1_dash, epsilon2_dash, ST_V, ST_D1V, ST_D1V_dash, STdash_V, STdash_D1V, \
    #         ST_U_dash, ST_U_inv, intervalIdx, weight = RHS(LHS_matrix, RHS_matrix, DEIM_matrix, DEIM_mat, C_matrix,
    #                                                        u_mid, q_mid, delta_samples, num_modes)
    #
    #     dq = q_k - q0
    #     J_k = Jacobian(M1, M2, N, A1, A2, D, M1_dash, M2_dash, N_dash, A1_dash, A2_dash, WT_B, UT_B, ST_V, ST_D1V,
    #                    ST_D1V_dash, STdash_V, STdash_D1V, ST_U_dash, ST_U_inv, epsilon1, epsilon2, epsilon3,
    #                    epsilon1_dash, epsilon2_dash,
    #                    q_mid, u_mid, dq, num_modes, dt)
    #
    #     J_phys = M / dt - J_k
    #
    #     eigs = np.linalg.eigvals(J_phys)
    #     idx = np.argsort(eigs.real)[::-1]
    #     leading = eigs[idx[:5]]
    #
    #     # assume `eigs` and `dt` are already in your namespace
    #     z = eigs * dt
    #
    #     # choose plotting limits based on your spectrum
    #     re_max = max(np.abs(z.real)) * 1.1
    #     im_max = max(np.abs(z.imag)) * 1.1
    #
    #     # Create complex grid for RK4 contour
    #     re = np.linspace(-re_max, re_max, 500)
    #     im = np.linspace(-im_max, im_max, 500)
    #     RE, IM = np.meshgrid(re, im)
    #     Z = RE + 1j * IM
    #     R_abs = np.abs(1 + Z + Z**2 / 2 + Z**3 / 6 + Z**4 / 24)
    #
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #
    #     # 1) shade stability region Re(z)<0
    #     ax.fill_betweenx(
    #         y=[-im_max, im_max],
    #         x1=[-re_max, -re_max],
    #         x2=[0, 0],
    #         color='lightblue',
    #         alpha=0.3,
    #         label='A‑stable region\n(Re(z)<0)'
    #     )
    #
    #     contour = ax.contour(RE, IM, R_abs, levels=[1], colors='black', linewidths=1.5)
    #     contour.collections[0].set_label('|R(z)| = 1 (RK4 stability boundary)')
    #
    #     # 2) plot all eigs
    #     ax.scatter(z.real, z.imag, s=20, color='gray', alpha=0.6, label='all modes')
    #
    #     # 3) highlight leading ones, e.g. largest real part
    #     idx = np.argsort(z.real)[::-1]
    #     leading = z[idx[:10]]
    #     ax.scatter(leading.real, leading.imag, s=80, color='red', label='leading modes')
    #
    #     # 4) draw the imaginary axis (stability boundary)
    #     ax.axvline(0, color='black', linestyle='--', linewidth=1)
    #
    #     ax.set_xlim(-re_max, re_max)
    #     ax.set_ylim(-im_max, im_max)
    #     ax.set_xlabel(r'$\Re(\lambda\Delta t)$')
    #     ax.set_ylabel(r'$\Im(\lambda\Delta t)$')
    #     ax.set_title('Eigenvalues on implicit‑midpoint stability region')
    #     ax.grid(True)
    #     plt.show()

    return q_k, q_dot, i, w


def implicit_midpoint_PODG_FOTR_primal_kdvb(
        RHS: callable,
        J_nl: callable,
        q0: np.ndarray,
        u1: np.ndarray,
        u2: np.ndarray,
        J_l: np.ndarray,
        primal_mat,
        dt: float,
        max_iter=10,
        **params_primal) -> np.ndarray:
    u_mid = 0.5 * (u1 + u2)

    # Initialize Newton iterate
    q_k = q0.copy()
    delq = np.inf

    # Build a function to solve linear systems with (J_l + J_nl(D, qmid, dt, ....))
    itr = 0

    while np.linalg.norm(delq) / np.linalg.norm(q_k) > 1e-6:
        if itr >= max_iter:
            print(f"Warning: Newton did not converge after {max_iter} iterations.")
            break
        # Midpoint state
        qmid = 0.5 * (q0 + q_k)

        # Residual: R = q_k - q0 - dt*RHS(qmid, u_mid)
        R_k = q_k - q0 - dt * RHS(qmid, u_mid, primal_mat.D_1r, primal_mat.D_2r, primal_mat.D_3r,
                                  primal_mat.prefactor, primal_mat.ST_V, primal_mat.ST_D1V,
                                  primal_mat.B_r, params_primal['c'], params_primal['alpha'],
                                  params_primal['omega'], params_primal['gamma'], params_primal['nu'])

        # Assemble Jacobian: J_l + J_nl
        # J_nl_k = J_nl(qmid, primal_mat.prefactor, primal_mat.ST_D1V, primal_mat.ST_V, params_primal['omega'], dt)
        J_k = J_l  # + J_nl_k

        delq = np.linalg.solve(J_k, -R_k)
        q_k += delq

        itr += 1

    return q_k


def implicit_midpoint_FOM_adjoint_kdvb(
        RHS: callable,
        J_nl: callable,
        p0: np.ndarray,
        a1: np.ndarray,
        a2: np.ndarray,
        b1: np.ndarray,
        b2: np.ndarray,
        J_l: csc_matrix,
        dx: float,
        dt: float,
        **kwargs) -> np.ndarray:
    D1 = kwargs['D1']
    omega = kwargs['omega']

    # Initialize Newton iterate
    p_k = p0.copy()

    # Evaluate midpoint state
    p_mid = 0.5 * (p_k + p0)
    a_mid = 0.5 * (a1 + a2)
    b_mid = 0.5 * (b1 + b2)

    # Compute the residual
    R_k = p_k - p0 + dt * RHS(p_mid, a_mid, b_mid, dx, **kwargs)

    # Assemble Jacobian: J_l + J_nl
    J_k = J_l + J_nl(a_mid, D1, omega, dt)

    # Solve linear system
    delq = scipy.sparse.linalg.spsolve(J_k, -R_k)
    p_k += delq

    return p_k


def implicit_midpoint_PODG_FRTO_adjoint_kdvb(
        RHS: callable,
        J_nl: callable,
        p0: np.ndarray,
        a1: np.ndarray,
        a2: np.ndarray,
        b1: np.ndarray,
        b2: np.ndarray,
        J_l: np.ndarray,
        dx: float,
        dt: float,
        primal_mat,
        **params_primal) -> np.ndarray:
    # Initialize Newton iterate
    p_k = p0.copy()

    # Evaluate midpoint state
    p_mid = 0.5 * (p_k + p0)
    a_mid = 0.5 * (a1 + a2)
    b_mid = 0.5 * (b1 + b2)

    # Compute the residual
    R_k = p_k - p0 + dt * RHS(p_mid, a_mid, b_mid, dx, primal_mat.D_1r, primal_mat.D_2r, primal_mat.D_3r,
                              primal_mat.prefactor, primal_mat.ST_V, primal_mat.ST_D1V,
                              params_primal['c'], params_primal['alpha'], params_primal['omega'],
                              params_primal['gamma'], params_primal['nu'])

    # Assemble Jacobian: J_l + J_nl
    J_k = J_l.copy()  # + J_nl(a_mid, primal_mat.prefactor, primal_mat.ST_D1V, primal_mat.ST_V, params_primal['omega'], dt).T

    # Solve linear system
    delq = np.linalg.solve(J_k, -R_k)
    p_k += delq

    return p_k


def implicit_midpoint_sPODG_FRTO_adjoint_kdvb(
        RHS: callable,
        p0: np.ndarray,
        a1: np.ndarray,
        a2: np.ndarray,
        b1: np.ndarray,
        b2: np.ndarray,
        u1: np.ndarray,
        u2: np.ndarray,
        a_dot: np.ndarray,
        LHS_matrix: np.ndarray,
        RHS_matrix: np.ndarray,
        DEIM_matrix: np.ndarray,
        DEIM_mat: np.ndarray,
        C_matrix: np.ndarray,
        TAR_matrix: np.ndarray,
        Vdp: np.ndarray,
        Wdp: np.ndarray,
        delta_samples: np.ndarray,
        num_modes: int,
        dx: float,
        dt: float,
        params_adjoint) -> np.ndarray:
    CTC = params_adjoint['CTC']

    # Initialize Newton iterate
    p_k = p0.copy()

    # Evaluate midpoint state
    p_mid = 0.5 * (p_k + p0)
    a_mid = 0.5 * (a1 + a2)  # reduced primal
    b_mid = 0.5 * (b1 + b2)  # target
    u_mid = 0.5 * (u1 + u2)  # control

    # Residual: R = M(a_mid) * (p0 - p_k) / dt - F(p_mid, a_mid, b_mid, u_mid)
    # (a_dot[2] if RK4 and a_dot[0] if implicit midpoint)
    M, E, T = RHS(LHS_matrix, RHS_matrix, DEIM_matrix, DEIM_mat, C_matrix, TAR_matrix, CTC, Vdp, Wdp,
                  a_mid, a_dot[2], u_mid, b_mid, delta_samples, dx, num_modes)

    R_k = M @ (p0 - p_k) / dt + E @ p_mid + T

    # Assemble Jacobian
    J_k = - M / dt + E / 2

    # Solve linear system
    delp = np.linalg.solve(J_k, -R_k)
    p_k += delp

    return p_k


def implicit_midpoint_PODG_FOTR_adjoint_kdvb(
        RHS: callable,
        J_nl: callable,
        p0: np.ndarray,
        a1: np.ndarray,
        a2: np.ndarray,
        b1: np.ndarray,
        b2: np.ndarray,
        J_l: np.ndarray,
        dx: float,
        dt: float,
        adjoint_mat,
        **params_adjoint) -> np.ndarray:
    # Initialize Newton iterate
    p_k = p0.copy()

    # Evaluate midpoint state
    p_mid = 0.5 * (p_k + p0)
    a_mid = 0.5 * (a1 + a2)
    b_mid = 0.5 * (b1 + b2)

    # Compute the residual
    R_k = p_k - p0 + dt * RHS(p_mid, a_mid, b_mid, dx, adjoint_mat.D_1r, adjoint_mat.D_2r, adjoint_mat.D_3r,
                              adjoint_mat.prefactor, adjoint_mat.ST_Va, adjoint_mat.ST_D1Va, adjoint_mat.ST_Vp,
                              adjoint_mat.ST_D1Vp, adjoint_mat.VaT_Vp, params_adjoint['c'],
                              params_adjoint['alpha'], params_adjoint['omega'],
                              params_adjoint['gamma'], params_adjoint['nu'])

    # Assemble Jacobian: J_l + J_nl
    J_k = J_l.copy()  # + J_nl(a_mid, adjoint_mat.prefactor, adjoint_mat.ST_D1Va, adjoint_mat.ST_Va,
    #    adjoint_mat.ST_D1Vp, adjoint_mat.ST_Vp, params_adjoint['omega'], dt).T

    # Solve linear system
    delq = np.linalg.solve(J_k, -R_k)
    p_k += delq

    return p_k
