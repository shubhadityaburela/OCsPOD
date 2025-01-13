import numpy as np
from numba import njit


@njit
def E11(N, A1, z_dot, r):
    return A1 - N @ np.diag(np.repeat(z_dot, r))


@njit
def E12(M2, N, A2, D, WTB, a_dot, z_dot, a_s, u, r):

    mat1_comp = (a_dot[None, :] @ N.T).reshape(-1)
    mat3_comp = N.T @ a_dot
    mat4_comp = M2 @ (D @ z_dot)
    mat6_comp = A2 @ a_s
    mat8_comp = WTB @ u

    mat5_comp = M2 @ np.diag(np.repeat(z_dot, r))

    return (mat1_comp - mat3_comp - mat4_comp + mat6_comp + mat8_comp)[None, :] + D.T @ (- mat5_comp + A2)


@njit
def E21(N, VTdashB, a_dot, u):
    mat2 = N @ a_dot[:, None]
    mat6 = VTdashB @ u[:, None]
    return mat2 + mat6


@njit
def E22(M2, D, WTdashB, a_dot, u):
    mat1 = a_dot[None, :] @ (M2 @ D)
    mat3_comp = M2 @ a_dot[:, None]
    mat7_comp = WTdashB @ u[:, None]

    return mat1 + D.T @ (mat3_comp + mat7_comp)


@njit
def C1(M1, VT, as_p, qs_tar, dx):        # We make use of the fact that VTV in the presence of constant dx is just M1
    return dx * (M1 @ as_p - VT @ qs_tar)


@njit
def C2(N, WT, as_p, qs_tar, dx):   # We make use of the fact that WTV in the presence of constant dx is just NT
    return dx * as_p[None, :] @ (N.T @ as_p - WT @ qs_tar)


################################################################################
@njit
def D_dash(z, r):
    arr = np.repeat(z, r)
    return np.diag(arr)


@njit
def DT_dash(arr):
    return arr[None, :]


def dv_dt(Dfd, V, dz_dt):
    return (Dfd @ V) * dz_dt


def dvT_dt(Dfd, V, dz_dt):
    return (Dfd @ V).transpose() * dz_dt


def dw_dt(Dfd, W, dz_dt):
    return (Dfd @ W) * dz_dt


def dwT_dt(Dfd, W, dz_dt):
    return (Dfd @ W).transpose() * dz_dt


@njit
def dD_dt(a_dot):
    return a_dot[:, None]


@njit
def dDT_dt(a_dot):
    return a_dot[None, :]


def dN_dt(Dfd, V, W, dz_dt):
    return dvT_dt(Dfd, V, dz_dt) @ W + V.transpose() @ dw_dt(Dfd, W, dz_dt)


def dNT_dt(Dfd, V, W, dz_dt):
    return dwT_dt(Dfd, W, dz_dt) @ V + W.transpose() @ dv_dt(Dfd, V, dz_dt)


def dM1_dt(Dfd, V, dz_dt):
    return dvT_dt(Dfd, V, dz_dt) @ V + V.transpose() @ dv_dt(Dfd, V, dz_dt)


def dM2_dt(Dfd, W, dz_dt):
    return dwT_dt(Dfd, W, dz_dt) @ W + W.transpose() @ dw_dt(Dfd, W, dz_dt)


def V_dash(Dfd, V):
    return Dfd @ V


def VT_dash(Dfd, V):  # We have assumed that (V')^T = (V^T)'
    return V_dash(Dfd, V).transpose()


def W_dash(Dfd, W):
    return Dfd @ W


def WT_dash(Dfd, W):
    return W_dash(Dfd, W).transpose()


def N_dash(Dfd, V, W):
    return VT_dash(Dfd, V) @ W + V.transpose() @ W_dash(Dfd, W)


def NT_dash(Dfd, V, W):
    return WT_dash(Dfd, W) @ V + W.transpose() @ V_dash(Dfd, V)


def M1_dash(Dfd, V):
    return VT_dash(Dfd, V) @ V + V.transpose() @ V_dash(Dfd, V)


def M2_dash(Dfd, W):
    return WT_dash(Dfd, W) @ W + W.transpose() @ W_dash(Dfd, W)


def A1_dash(Dfd, V, A):
    return VT_dash(Dfd, V) @ (A @ V) + (V.transpose() @ A) @ V_dash(Dfd, V)


def A2_dash(Dfd, V, W, A):
    return WT_dash(Dfd, W) @ (A @ V) + (W.transpose() @ A) @ V_dash(Dfd, V)


def dv_dash_dt(Dfd, V, dz_dt):
    return (Dfd @ V_dash(Dfd, V)) * dz_dt


def dw_dash_dt(Dfd, W, dz_dt):
    return (Dfd @ W_dash(Dfd, W)) * dz_dt


def dvT_dash_dt(Dfd, V, dz_dt):
    return (Dfd @ VT_dash(Dfd, V).transpose()).transpose() * dz_dt


def dwT_dash_dt(Dfd, W, dz_dt):
    return (Dfd @ WT_dash(Dfd, W).transpose()).transpose() * dz_dt


def dM1_dash_dt(Dfd, V, dz_dt):
    return dvT_dash_dt(Dfd, V, dz_dt) @ V + VT_dash(Dfd, V) @ dv_dt(Dfd, V, dz_dt) \
        + dvT_dt(Dfd, V, dz_dt) @ V_dash(Dfd, V) + V.transpose() @ dv_dash_dt(Dfd, V, dz_dt)


def dM2_dash_dt(Dfd, W, dz_dt):
    return dwT_dash_dt(Dfd, W, dz_dt) @ W + WT_dash(Dfd, W) @ dw_dt(Dfd, W, dz_dt) \
        + dwT_dt(Dfd, W, dz_dt) @ W_dash(Dfd, W) + W.transpose() @ dw_dash_dt(Dfd, W, dz_dt)


def dN_dash_dt(Dfd, V, W, dz_dt):
    return dvT_dash_dt(Dfd, V, dz_dt) @ W + VT_dash(Dfd, V) @ dw_dt(Dfd, W, dz_dt) \
        + dvT_dt(Dfd, V, dz_dt) @ W_dash(Dfd, W) + V.transpose() @ dw_dash_dt(Dfd, W, dz_dt)
