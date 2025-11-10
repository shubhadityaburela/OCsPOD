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
def C1(VTV, as_p, VTqs_tar, dx): # We make use of the fact that VTV in the presence of constant dx and identity CTC is just M1
    return dx * (VTV @ as_p - VTqs_tar)


@njit
def C2(WTV, as_p, WTqs_tar, dx):   # We make use of the fact that WTV in the presence of constant dx and identity CTC is just NT
    return dx * as_p[None, :] @ (WTV @ as_p - WTqs_tar)


################################################################################
@njit
def E11_kdvb(M1_dash, N, A1, nl_1, as_kron_Jac, z_dot, r):
    return M1_dash * z_dot + A1 - N @ np.diag(np.repeat(z_dot, r)) + nl_1 @ as_kron_Jac


@njit
def E12_kdvb(M2, N, N_dash, A2, D, WTB, nl_2, as_kron, as_kron_Jac, a_dot, z_dot, a_s, u, r):

    mat1_comp = a_dot[None, :] @ N.T
    mat2_comp = D.T @ (N_dash.T * z_dot)
    mat3_comp = (N.T.dot(a_dot))[None, :]
    mat4_comp = (M2 @ (D.dot(z_dot)))[None, :]
    mat5_comp = D.T @ (M2 @ np.diag(np.repeat(z_dot, r)))
    mat6_comp = (A2.dot(a_s))[None, :]
    mat7_comp = D.T @ A2
    mat8_comp = (WTB.dot(u))[None, :]
    mat9_comp = (nl_2.dot(as_kron))[None, :]
    mat10_comp = D.T @ (nl_2 @ as_kron_Jac)

    return mat1_comp + mat2_comp - mat3_comp - mat4_comp - mat5_comp \
        + mat6_comp + mat7_comp + mat8_comp + mat9_comp + mat10_comp


@njit
def E21_kdvb(N, D, M1_dash, N_dash, A1_dash, VTdashB, a_dot, z_dot, a_s, u):
    mat1_comp = (N_dash * z_dot) @ D
    mat2_comp = N @ a_dot[:, None]
    mat3_comp = M1_dash.dot(a_dot)[:, None]
    mat4_comp = N_dash @ (D.dot(z_dot))[:, None]
    mat5_comp = A1_dash.dot(a_s)[:, None]
    mat6_comp = (VTdashB.dot(u))[:, None]

    return mat1_comp + mat2_comp - mat3_comp - mat4_comp + mat5_comp + mat6_comp


@njit
def E22_kdvb(M2, M2_dash, N_dash, A2_dash, D, WTdashB, a_dot, z_dot, a_s, u):
    mat1_comp = a_dot[None, :] @ (M2 @ D)
    mat2_comp = D.T @ ((M2_dash * z_dot) @ D)
    mat3_comp = D.T @ (M2 @ a_dot[:, None])
    mat4_comp = D.T @ (N_dash.T @ a_dot[:, None])
    mat5_comp = D.T @ (M2_dash @ (D.dot(z_dot))[:, None])
    mat6_comp = D.T @ (A2_dash.dot(a_s))[:, None]
    mat7_comp = D.T @ (WTdashB.dot(u))[:, None]

    return mat1_comp + mat2_comp + mat3_comp - mat4_comp - mat5_comp + mat6_comp + mat7_comp


# ################################################################################
#
# @njit
# def D_dash(z, r):
#     arr = np.repeat(z, r)
#     return np.diag(arr)
#
#
# @njit
# def DT_dash(arr):
#     return arr[None, :]
#
#
# def dv_dt(Dfd, V, dz_dt):
#     return (Dfd @ V) * dz_dt
#
#
# def dvT_dt(Dfd, V, dz_dt):
#     return (Dfd @ V).transpose() * dz_dt
#
#
# def dw_dt(Dfd, W, dz_dt):
#     return (Dfd @ W) * dz_dt
#
#
# def dwT_dt(Dfd, W, dz_dt):
#     return (Dfd @ W).transpose() * dz_dt
#
#
# @njit
# def dD_dt(a_dot):
#     return a_dot[:, None]
#
#
# @njit
# def dDT_dt(a_dot):
#     return a_dot[None, :]
#
#
# def dN_dt(Dfd, V, W, dz_dt):
#     return dvT_dt(Dfd, V, dz_dt) @ W + V.transpose() @ dw_dt(Dfd, W, dz_dt)
#
#
# def dNT_dt(Dfd, V, W, dz_dt):
#     return dwT_dt(Dfd, W, dz_dt) @ V + W.transpose() @ dv_dt(Dfd, V, dz_dt)
#
#
# def dM1_dt(Dfd, V, dz_dt):
#     return dvT_dt(Dfd, V, dz_dt) @ V + V.transpose() @ dv_dt(Dfd, V, dz_dt)
#
#
# def dM2_dt(Dfd, W, dz_dt):
#     return dwT_dt(Dfd, W, dz_dt) @ W + W.transpose() @ dw_dt(Dfd, W, dz_dt)
#
#
# def V_dash(Dfd, V):
#     return Dfd @ V
#
#
# def VT_dash(Dfd, V):  # We have assumed that (V')^T = (V^T)'
#     return V_dash(Dfd, V).transpose()
#
#
# def W_dash(Dfd, W):
#     return Dfd @ W
#
#
# def WT_dash(Dfd, W):
#     return W_dash(Dfd, W).transpose()
#
#
# def N_dash(Dfd, V, W):
#     return VT_dash(Dfd, V) @ W + V.transpose() @ W_dash(Dfd, W)
#
#
# def NT_dash(Dfd, V, W):
#     return WT_dash(Dfd, W) @ V + W.transpose() @ V_dash(Dfd, V)
#
#
# def M1_dash(Dfd, V):
#     return VT_dash(Dfd, V) @ V + V.transpose() @ V_dash(Dfd, V)
#
#
# def M2_dash(Dfd, W):
#     return WT_dash(Dfd, W) @ W + W.transpose() @ W_dash(Dfd, W)
#
#
# def A1_dash(Dfd, V, A):
#     return VT_dash(Dfd, V) @ (A @ V) + (V.transpose() @ A) @ V_dash(Dfd, V)
#
#
# def A2_dash(Dfd, V, W, A):
#     return WT_dash(Dfd, W) @ (A @ V) + (W.transpose() @ A) @ V_dash(Dfd, V)
#
#
# def dv_dash_dt(Dfd, V, dz_dt):
#     return (Dfd @ V_dash(Dfd, V)) * dz_dt
#
#
# def dw_dash_dt(Dfd, W, dz_dt):
#     return (Dfd @ W_dash(Dfd, W)) * dz_dt
#
#
# def dvT_dash_dt(Dfd, V, dz_dt):
#     return (Dfd @ VT_dash(Dfd, V).transpose()).transpose() * dz_dt
#
#
# def dwT_dash_dt(Dfd, W, dz_dt):
#     return (Dfd @ WT_dash(Dfd, W).transpose()).transpose() * dz_dt
#
#
# def dM1_dash_dt(Dfd, V, dz_dt):
#     return dvT_dash_dt(Dfd, V, dz_dt) @ V + VT_dash(Dfd, V) @ dv_dt(Dfd, V, dz_dt) \
#         + dvT_dt(Dfd, V, dz_dt) @ V_dash(Dfd, V) + V.transpose() @ dv_dash_dt(Dfd, V, dz_dt)
#
#
# def dM2_dash_dt(Dfd, W, dz_dt):
#     return dwT_dash_dt(Dfd, W, dz_dt) @ W + WT_dash(Dfd, W) @ dw_dt(Dfd, W, dz_dt) \
#         + dwT_dt(Dfd, W, dz_dt) @ W_dash(Dfd, W) + W.transpose() @ dw_dash_dt(Dfd, W, dz_dt)
#
#
# def dN_dash_dt(Dfd, V, W, dz_dt):
#     return dvT_dash_dt(Dfd, V, dz_dt) @ W + VT_dash(Dfd, V) @ dw_dt(Dfd, W, dz_dt) \
#         + dvT_dt(Dfd, V, dz_dt) @ W_dash(Dfd, W) + V.transpose() @ dw_dash_dt(Dfd, W, dz_dt)
