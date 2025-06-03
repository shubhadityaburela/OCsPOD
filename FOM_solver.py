import numpy as np
from numba import njit
from scipy import sparse

from TI_schemes import rk4_FOM_adj, rk4_FOM, rk4_FOM_targ, implicit_midpoint_FOM_adj, DIRK_FOM_adj, \
    poly_interp_FOM_adj, bdf4_FOM_adj, bdf2_FOM_adj, bdf3_FOM_adj


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
