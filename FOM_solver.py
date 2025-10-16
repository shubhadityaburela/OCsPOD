import numpy as np
import scipy
from matplotlib import pyplot as plt
from numba import njit
from scipy import sparse
from scipy.sparse import csc_matrix, diags
from tqdm import tqdm

# from tqdm import tqdm

from TI_schemes import rk4_FOM_adj, rk4_FOM, rk4_FOM_targ, implicit_midpoint_FOM_adj, DIRK_FOM_adj, \
    poly_interp_FOM_adj, bdf4_FOM_adj, bdf2_FOM_adj, bdf3_FOM_adj, rk4_FOM_kdvb, rk4_FOM_adj_kdvb, \
    implicit_midpoint_FOM_primal_kdvb, implicit_midpoint_FOM_adjoint_kdvb


def IC_primal(X, Lx, offset, variance, type_of_problem):
    if type_of_problem == "Constant_shift":
        A = 8
        mu = (Lx - 0) / offset
        x_t = np.mod(X - 0, Lx - 0) + 0 - mu
        q = A / np.cosh(np.sqrt(3 * A) / 6 * x_t) ** 2
    elif type_of_problem == "Shifting":
        q = np.exp(-((X - Lx / offset) ** 2) / variance)
    else:
        print("Choose a proper initial condition...")
        exit()

    return q


def RHS_primal(q, f, A, psi):
    return A @ q + psi @ f


def TI_primal(q, f, A, psi, Nx, Nt, dt):
    # Time loop
    qs = np.zeros((Nx, Nt))
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


def RHS_primal_target(q, Grad, v_x, nu):
    DT = v_x * Grad
    qdot = - DT.dot(q) + nu * (Grad @ Grad).dot(q)
    return qdot


def TI_primal_target(q, Grad, v_x_target, Nx, Nt, dt, nu=0.0):
    # Time loop
    qs = np.zeros((Nx, Nt))
    qs[:, 0] = q

    for n in range(1, Nt):
        qs[:, n] = rk4_FOM_targ(RHS_primal_target, qs[:, n - 1], dt, Grad, v_x_target[n - 1], nu)

    return qs


# -------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #

@njit
def IC_primal_kdv(X, Lx, c, offset):
    A = 12 * c
    mu = (Lx - 0) / offset
    x_t = np.mod(X - 0, Lx - 0) + 0 - mu
    q = A / np.cosh(np.sqrt(3 * A) / 6 * x_t) ** 2
    return q


def RHS_primal_kdv_expl(q, u, D1, D2, D3, B, L, c, alpha, omega, gamma, nu):
    return (L.dot(q)
            - (omega / 3) * D1.dot(q ** 2)
            - (omega / 3) * (q * D1.dot(q))
            + B.dot(u))


def TI_primal_kdv_expl(q, f, D1, D2, D3, B, L, Nx, Nt, dt, c, alpha=1.0, omega=1.0, gamma=1e-4, nu=1e-4):
    # Time loop
    qs = np.zeros((Nx, Nt))
    qs[:, 0] = q
    for n in tqdm(range(1, Nt), desc="Primal Working"):
        qs[:, n] = rk4_FOM_kdvb(RHS_primal_kdv_expl, qs[:, n - 1], f[:, n - 1], f[:, n], dt,
                                D1, D2, D3, B, L, c, alpha, omega, gamma, nu)
    return qs


# Nonlinear Jacobian for KdV steepening term
def J_nl_primal_kdv(q: np.ndarray,
                    D1: csc_matrix,
                    omega: float,
                    dt: float) -> csc_matrix:
    D1q = D1.dot(q)
    return omega * dt * ((1 / 2) * diags(D1q, 0, format='csc')
                         + (1 / 6) * (diags(q, 0, format='csc') @ D1))


# Right-hand side for KdV-Burgers-advection
def RHS_primal_kdv_impl(q: np.ndarray,
                        u: np.ndarray,
                        D1: csc_matrix,
                        D2: csc_matrix,
                        D3: csc_matrix,
                        B: csc_matrix,
                        L: csc_matrix,
                        c: float,
                        alpha: float,
                        omega: float,
                        gamma: float,
                        nu: float) -> np.ndarray:
    return (L.dot(q)
            - (omega / 3) * D1.dot(q ** 2)
            - (omega / 3) * (q * D1.dot(q))
            + B.dot(u))


def TI_primal_kdv_impl(q, f, J_l, Nx, Nt, dt, **kwargs):
    # Time loop
    qs = np.zeros((Nx, Nt))
    qs[:, 0] = q
    for n in range(1, Nt):
        qs[:, n] = implicit_midpoint_FOM_primal_kdvb(RHS_primal_kdv_impl,
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
def IC_adjoint_kdv(X):
    q_adj = np.zeros_like(X)
    return q_adj


def RHS_adjoint_kdv_expl(q_adj, q, q_tar, D1, D2, D3, CTC, L, dx, c, alpha, omega, gamma, nu):
    out = L.dot(q_adj) - \
          omega * q * D1.dot(q_adj)
    out[CTC] -= dx * (q - q_tar)[CTC]
    return out


def TI_adjoint_kdv_expl(q0_adj, qs, qs_target, D1, D2, D3, CTC, L, Nx, dx, Nt, dt, c, alpha, omega, gamma, nu):
    # Time loop
    qs_adj = np.zeros((Nx, Nt))
    qs_adj[:, -1] = q0_adj

    for n in tqdm(range(1, Nt), desc="Adjoint working"):
        qs_adj[:, -(n + 1)] = rk4_FOM_adj_kdvb(RHS_adjoint_kdv_expl, qs_adj[:, -n], qs[:, -n], qs[:, -(n + 1)],
                                               qs_target[:, -n], qs_target[:, -(n + 1)], - dt, D1, D2, D3, CTC, L, dx,
                                               c, alpha, omega, gamma, nu)

    return qs_adj


def J_nl_adjoint_kdv(q: np.ndarray,
                     D1: csc_matrix,
                     omega: float,
                     dt: float) -> csc_matrix:
    return - ((omega * dt) / 2) * (diags(q, 0, format='csc') @ D1)


def RHS_adjoint_kdv_impl(q_adj: np.ndarray,
                         q: np.ndarray,
                         q_tar: np.ndarray,
                         dx: float,
                         D1: csc_matrix,
                         D2: csc_matrix,
                         D3: csc_matrix,
                         CTC: np.ndarray,
                         L: csc_matrix,
                         c: float,
                         alpha: float,
                         omega: float,
                         gamma: float,
                         nu: float) -> np.ndarray:
    out = L.dot(q_adj) \
          - omega * q * D1.dot(q_adj)
    out[CTC] -= dx * (q - q_tar)[CTC]
    return out


def TI_adjoint_kdv_impl(q0_adj, qs, qs_target, J_l, Nx, Nt, dx, dt, **kwargs):
    # Time loop
    qs_adj = np.zeros((Nx, Nt))
    qs_adj[:, -1] = q0_adj

    for n in range(1, Nt):
        qs_adj[:, -(n + 1)] = implicit_midpoint_FOM_adjoint_kdvb(RHS_adjoint_kdv_impl,
                                                                 J_nl_adjoint_kdv,
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

# if n % 100 == 0:
#     qmid = 0.5 * (qs[:, n - 1] + qs[:, n])
#     J_k = J_l + J_nl_primal_kdv(qmid, kwargs['D1'], kwargs['omega'], dt)
#
#     J_phys = - J_k.todense()
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
#     R_abs = np.abs(1 + Z + Z ** 2 / 2 + Z ** 3 / 6 + Z ** 4 / 24)
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


# eps_list = [1e-8, 1e-6, 1e-4]  # avoid astronomically small eps
# for eps in eps_list:
#     q_mid = (qs[:, 10] + qs[:, 11]) / 2  # pick a midpoint state from a forward trajectory
#     u_mid = (f[:, 10] + f[:, 11]) / 2
#     v = np.random.randn(*q_mid.shape)
#
#     R_plus = RHS_primal_kdv_impl(q_mid + eps * v, u_mid, **kwargs)
#     R = RHS_primal_kdv_impl(q_mid, u_mid, **kwargs)
#     finite_diff = (R_plus - R) / eps  # vector
#
#     # Compute the action of Jacobian on v
#     J_l = (kwargs['A'] - kwargs['gamma'] * kwargs['D3'] + kwargs['nu'] * kwargs['D2'])
#     J_nl = - J_nl_primal_kdv(q_mid, kwargs['D1'], kwargs['omega'], dt) * 2 / dt
#     J_k = (J_l + J_nl).dot(v)
#
#     print(eps)
#     print("||finite_diff - JF_action|| = ", np.linalg.norm(finite_diff - J_k))
#     print("||finite_diff|| = ", np.linalg.norm(finite_diff))
#     print("||finite_diff - JF_action|| / ||finite_diff|| = ", np.linalg.norm(finite_diff - J_k) / np.linalg.norm(finite_diff))




# eps_list = [1e-8, 1e-6, 1e-4]  # avoid astronomically small eps
    # for eps in eps_list:
    #     p_mid = (qs_adj[:, 10] + qs_adj[:, 11]) / 2  # pick a midpoint state from a forward trajectory
    #     a_mid = (qs[:, 10] + qs[:, 11]) / 2
    #     b_mid = (qs_target[:, 10] + qs_target[:, 11]) / 2
    #     w = np.random.randn(*p_mid.shape)
    #
    #     R_plus = RHS_adjoint_kdv_impl(p_mid + eps * w, a_mid, b_mid, dx, **kwargs)
    #     R = RHS_adjoint_kdv_impl(p_mid, a_mid, b_mid, dx, **kwargs)
    #     finite_diff = (R_plus - R) / eps  # vector
    #
    #     # Compute the action of Jacobian on w
    #     J_l = (- kwargs['A'].T + kwargs['gamma'] * kwargs['D3'].T - kwargs['nu'] * kwargs['D2'].T)
    #     J_nl = J_nl_primal_kdv(a_mid, kwargs['D1'], kwargs['omega'], dt).T * 2 / dt
    #     J_k = (J_l + J_nl).dot(w)
    #
    #     print(eps)
    #     print("||finite_diff - JF_action|| = ", np.linalg.norm(finite_diff - J_k))
    #     print("||finite_diff|| = ", np.linalg.norm(finite_diff))
    #     print("||finite_diff - JF_action|| / ||finite_diff|| = ", np.linalg.norm(finite_diff - J_k) / np.linalg.norm(finite_diff))