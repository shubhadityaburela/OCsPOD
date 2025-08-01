import numpy as np
import scipy
from numba import njit
from scipy import sparse
from scipy.sparse import csc_matrix, diags

from TI_schemes import rk4_FOM_adj, rk4_FOM, rk4_FOM_targ, implicit_midpoint_FOM_adj, DIRK_FOM_adj, \
    poly_interp_FOM_adj, bdf4_FOM_adj, bdf2_FOM_adj, bdf3_FOM_adj, rk4_FOM_kdvb, rk4_FOM_adj_kdvb, \
    implicit_midpoint_FOM_primal_kdvb, implicit_midpoint_FOM_adjoint_kdvb


@njit
def IC_primal(X, Lxi, offset, variance):
    q = np.exp(-((X - Lxi / offset) ** 2) / variance)
    return q


def RHS_primal(q, f, A, psi):
    return A @ q + psi @ f


def TI_primal(q, f, A, psi, Nxi, Nt, dt):
    # Time loop
    qs = np.zeros((Nxi, Nt))
    qs[:, 0] = q
    for n in range(1, Nt):
        qs[:, n] = rk4_FOM(RHS_primal, qs[:, n - 1], f[:, n - 1], f[:, n], dt, A, psi)
    return qs


@njit
def IC_adjoint(X):
    q_adj = np.zeros_like(X)
    return q_adj


def RHS_adjoint_expl(q_adj, q, q_tar, A, CTC, dx):
    out = -A @ q_adj
    out[CTC] -= dx * (q - q_tar)[CTC]
    return out


def RHS_adjoint_impl(q_adj, q, q_tar, M_f, A_f, LU_M_f, CTC, Nx, dx, dt, scheme):
    # Precompute the difference once
    diff = q - q_tar  # shape = (Nx,)

    # Depending on scheme, build the right‐hand side vector `rhs_vec` before solving
    if scheme == "implicit_midpoint":
        vec = A_f @ q_adj
        vec[CTC] -= dt * dx * diff[CTC]
        return LU_M_f.solve(vec)
    elif scheme == "DIRK":
        vec = - (A_f @ q_adj)
        vec[CTC] -= dx * diff[CTC]
        return LU_M_f.solve(vec)
    elif scheme == "BDF2":
        vec = 4.0 * q_adj[1] - 1.0 * q_adj[0]
        vec[CTC] -= 2.0 * dt * dx * diff[CTC]
        return LU_M_f.solve(vec)
    elif scheme == "BDF3":
        vec = 18.0 * q_adj[2] - 9.0 * q_adj[1] + 2.0 * q_adj[0]
        vec[CTC] -= 6.0 * dt * dx * diff[CTC]
        return LU_M_f.solve(vec)
    elif scheme == "BDF4":
        vec = 48.0 * q_adj[3] - 36.0 * q_adj[2] + 16.0 * q_adj[1] - 3.0 * q_adj[0]
        vec[CTC] -= 12.0 * dt * dx * diff[CTC]
        return LU_M_f.solve(vec)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")


def TI_adjoint(q0_adj, qs, qs_target, M_f, A_f, LU_M_f, CTC, Nxi, dx, Nt, dt, scheme, opt_poly_jacobian=None):
    # Time loop
    qs_adj = np.zeros((Nxi, Nt))
    qs_adj[:, -1] = q0_adj

    if scheme == "RK4":
        for n in range(1, Nt):
            qs_adj[:, -(n + 1)] = rk4_FOM_adj(RHS_adjoint_expl, qs_adj[:, -n], qs[:, -n], qs[:, -(n + 1)],
                                              qs_target[:, -n], qs_target[:, -(n + 1)], - dt, A_f, CTC, dx)
    elif scheme == "implicit_midpoint":
        for n in range(1, Nt):
            qs_adj[:, -(n + 1)] = implicit_midpoint_FOM_adj(RHS_adjoint_impl, qs_adj[:, -n], qs[:, -n], qs[:, -(n + 1)],
                                                            qs_target[:, -n], qs_target[:, -(n + 1)], - dt, M_f, A_f,
                                                            LU_M_f, CTC, Nxi, dx,
                                                            scheme)
    elif scheme == "DIRK":
        for n in range(1, Nt):
            qs_adj[:, -(n + 1)] = DIRK_FOM_adj(RHS_adjoint_impl, qs_adj[:, -n], qs[:, -n], qs[:, -(n + 1)],
                                               qs_target[:, -n], qs_target[:, -(n + 1)], - dt, M_f, A_f,
                                               LU_M_f, CTC, Nxi, dx,
                                               scheme)
    elif scheme == "BDF2":
        # last 2 steps (x_{n-1}, x_{n-2}) with RK4 (Effectively 2nd order)
        for n in range(1, 2):
            qs_adj[:, -(n + 1)] = rk4_FOM_adj(RHS_adjoint_expl, qs_adj[:, -n], qs[:, -n], qs[:, -(n + 1)],
                                              qs_target[:, -n], qs_target[:, -(n + 1)], - dt, A_f, CTC, dx)
        for n in range(2, Nt):
            qs_adj[:, -(n + 1)] = bdf2_FOM_adj(RHS_adjoint_impl, qs_adj, qs[:, -(n + 1)],
                                               qs_target[:, -(n + 1)], - dt, M_f, A_f,
                                               LU_M_f, CTC, Nxi, dx, scheme, n)
    elif scheme == "BDF3":
        # last 4 steps (x_{n-1}, x_{n-2}, x_{n-3}, x_{n-4}) with polynomial interpolation (4th order)
        qs_adj[:, -4:] = poly_interp_FOM_adj(RHS_adjoint_expl, q0_adj, qs, qs_target, A_f, opt_poly_jacobian, CTC, Nxi,
                                             dx, -dt)
        for n in range(4, Nt):
            qs_adj[:, -(n + 1)] = bdf3_FOM_adj(RHS_adjoint_impl, qs_adj, qs[:, -(n + 1)],
                                               qs_target[:, -(n + 1)], - dt, M_f, A_f,
                                               LU_M_f, CTC, Nxi, dx, scheme, n)
    elif scheme == "BDF4":
        # last 4 steps (x_{n-1}, x_{n-2}, x_{n-3}, x_{n-4}) with polynomial interpolation (4th order)
        qs_adj[:, -4:] = poly_interp_FOM_adj(RHS_adjoint_expl, q0_adj, qs, qs_target, A_f, opt_poly_jacobian, CTC, Nxi,
                                             dx, -dt)
        for n in range(4, Nt):
            qs_adj[:, -(n + 1)] = bdf4_FOM_adj(RHS_adjoint_impl, qs_adj, qs[:, -(n + 1)],
                                               qs_target[:, -(n + 1)], - dt, M_f, A_f,
                                               LU_M_f, CTC, Nxi, dx, scheme, n)

    return qs_adj


def RHS_primal_target(q, Grad, v_x):
    DT = v_x * Grad
    qdot = - DT.dot(q)
    return qdot


def TI_primal_target(q, Grad, v_x_target, Nxi, Nt, dt):
    # Time loop
    qs = np.zeros((Nxi, Nt))
    qs[:, 0] = q

    for n in range(1, Nt):
        qs[:, n] = rk4_FOM_targ(RHS_primal_target, qs[:, n - 1], dt, Grad, v_x_target[n - 1])

    return qs


# -------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #

@njit
def IC_primal_kdvb(X, Lx, offset, variance):
    q = np.exp(-((X - Lx / offset) ** 2) / variance)
    return q


def RHS_primal_kdvb_expl(q, f, A_p, D1, D2, D3, psi, omega, gamma, nu):
    return A_p @ q - omega * 6.0 * q * (D1 @ q) - gamma * D3 @ q + nu * D2 @ q + psi @ f


def TI_primal_kdvb_expl(q, f, A_p, D1, D2, D3, psi, Nx, Nt, dt, omega=1.0, gamma=1e-4, nu=1e-4):
    # Time loop
    qs = np.zeros((Nx, Nt))
    qs[:, 0] = q
    for n in range(1, Nt):
        qs[:, n] = rk4_FOM_kdvb(RHS_primal_kdvb_expl, qs[:, n - 1], f[:, n - 1], f[:, n], dt,
                                A_p, D1, D2, D3, psi, omega, gamma, nu)
    return qs


# Nonlinear Jacobian for KdV steepening term
def J_nl_primal_kdv(q: np.ndarray,
                    D1: csc_matrix,
                    omega: float,
                    dt: float) -> csc_matrix:
    # Returns sparse matrix diag(D q) + D.T*diag(q) scaled by 3*dt
    D1q = D1.dot(q)
    return 3.0 * omega * dt * (diags(D1q, 0, format='csc') + D1.T.multiply(diags(q, 0, format='csc')))


# Right-hand side for KdV-Burgers-advection
def RHS_primal_kdvb_impl(q: np.ndarray,
                         u: np.ndarray,
                         A: csc_matrix,
                         D1: csc_matrix,
                         D2: csc_matrix,
                         D3: csc_matrix,
                         B: csc_matrix,
                         omega: float,
                         gamma: float,
                         nu: float) -> np.ndarray:
    return (A.dot(q)
            - omega * 6.0 * q * (D1.dot(q))
            - gamma * D3.dot(q)
            + nu * D2.dot(q)
            + B.dot(u))


def TI_primal_kdvb_impl(q, f, J_l, Nx, Nt, dt, **kwargs):
    # Time loop
    qs = np.zeros((Nx, Nt))
    qs[:, 0] = q
    for n in range(1, Nt):
        qs[:, n] = implicit_midpoint_FOM_primal_kdvb(RHS_primal_kdvb_impl,
                                                     J_nl_primal_kdv,
                                                     qs[:, n - 1],
                                                     f[:, n - 1],
                                                     f[:, n],
                                                     J_l,
                                                     dt,
                                                     **kwargs)
    print('Implicit midpoint primal finished')
    return qs


@njit
def IC_adjoint_kdvb(X):
    q_adj = np.zeros_like(X)
    return q_adj


def RHS_adjoint_kdvb_expl(q_adj, q, q_tar, A, D1, D2, D3, CTC, dx, omega, gamma, nu):
    out = -A @ q_adj + omega * 6.0 * (
            scipy.sparse.spdiags(D1 @ q, 0, CTC.size, CTC.size) +
            D1.T @ scipy.sparse.spdiags(q, 0, CTC.size, CTC.size)).T @ q_adj + \
          gamma * D3.T @ q_adj - nu * D2.T @ q_adj
    out[CTC] -= dx * (q - q_tar)[CTC]
    return out


def TI_adjoint_kdvb_expl(q0_adj, qs, qs_target, A_f, D1, D2, D3, CTC, Nx, dx, Nt, dt, omega, gamma, nu):
    # Time loop
    qs_adj = np.zeros((Nx, Nt))
    qs_adj[:, -1] = q0_adj

    for n in range(1, Nt):
        qs_adj[:, -(n + 1)] = rk4_FOM_adj_kdvb(RHS_adjoint_kdvb_expl, qs_adj[:, -n], qs[:, -n], qs[:, -(n + 1)],
                                               qs_target[:, -n], qs_target[:, -(n + 1)], - dt, A_f, D1, D2, D3, CTC, dx,
                                               omega, gamma, nu)

    return qs_adj


def RHS_adjoint_kdvb_impl(q_adj: np.ndarray,
                          q: np.ndarray,
                          q_tar: np.ndarray,
                          dx: float,
                          A: csc_matrix,
                          D1: csc_matrix,
                          D2: csc_matrix,
                          D3: csc_matrix,
                          CTC: np.ndarray,
                          omega: float,
                          gamma: float,
                          nu: float) -> np.ndarray:
    out = (- A.T +
           gamma * D3.T -
           nu * D2.T +
           omega * 6.0 * (diags(D1.dot(q), 0, format='csc') + D1.T.multiply(diags(q, 0, format='csc')))) @ q_adj
    out[CTC] -= dx * (q - q_tar)[CTC]

    return out


def TI_adjoint_kdvb_impl(q0_adj, qs, qs_target, J_l, Nx, Nt, dx, dt, **kwargs):
    # Time loop
    qs_adj = np.zeros((Nx, Nt))
    qs_adj[:, -1] = q0_adj

    for n in range(1, Nt):
        qs_adj[:, -(n + 1)] = implicit_midpoint_FOM_adjoint_kdvb(RHS_adjoint_kdvb_impl,
                                                                 J_nl_primal_kdv,
                                                                 qs_adj[:, -n],
                                                                 qs[:, -n],
                                                                 qs[:, -(n + 1)],
                                                                 qs_target[:, -n],
                                                                 qs_target[:, -(n + 1)],
                                                                 J_l,
                                                                 dx,
                                                                 dt,
                                                                 **kwargs)
    print('Implicit midpoint adjoint finished')

    return qs_adj
