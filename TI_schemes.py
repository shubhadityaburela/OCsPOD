import scipy

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
