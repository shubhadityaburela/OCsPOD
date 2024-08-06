import numpy as np


def Q11(M1):
    return M1


def Q12(N, D):
    return D.transpose() @ N.transpose()


def Q21(N, D):
    return N @ D


def Q22(M2, D):
    return D.transpose() @ (M2 @ D)


def B11(N, A1, z_dot, r):
    return A1 - N @ D_dash(z_dot, r)


def B12(M2, N, A2, D, WTB, a_dot, z_dot, a_s, u, r):
    mat1 = dDT_dt(a_dot) @ N.transpose()
    mat3 = DT_dash(N.transpose() @ a_dot)
    mat4 = DT_dash(M2 @ (D @ z_dot))
    mat5 = D.transpose() @ (M2 @ D_dash(z_dot, r))
    mat6 = DT_dash(A2 @ a_s)
    mat7 = D.transpose() @ A2
    mat8 = DT_dash(WTB @ u)

    return mat1 - mat3 - mat4 - mat5 + mat6 + mat7 + mat8


def B21(Dfd, N, VTdashB, a_dot, u):
    mat2 = N @ dD_dt(a_dot)
    mat6 = VTdashB @ u
    return mat2 + mat6


def B22(Dfd, M2, D, WTdashB, a_dot, u):
    mat1 = dDT_dt(a_dot) @ (M2 @ D)
    mat3 = D.transpose() @ (M2 @ dD_dt(a_dot))
    mat7 = D.transpose() @ (WTdashB @ u)

    return mat1 + mat3 + mat7


def C1(V, qs_target, a_s):
    return a_s - V.transpose() @ qs_target


def C2(Dfd, V, qs_target, T, a_s):
    return np.atleast_1d((a_s.transpose() @ T) @ a_s - ((a_s.transpose() @ V_dash(Dfd, V).transpose()) @ qs_target))


def Z11(M1):
    return M1


def Z12(N, D):
    return D.transpose() @ N.transpose()


def Z21(N, D):
    return N @ D


def Z22(M2, D):
    return D.transpose() @ (M2 @ D)


#################
def D_dash(z, r):
    arr = np.repeat(z, r)
    return np.diag(arr)


def DT_dash(arr):
    return np.atleast_2d(arr).T


def dv_dt(Dfd, V, dz_dt):
    return (Dfd @ V) * dz_dt


def dvT_dt(Dfd, V, dz_dt):
    return (Dfd @ V).transpose() * dz_dt


def dw_dt(Dfd, W, dz_dt):
    return (Dfd @ W) * dz_dt


def dwT_dt(Dfd, W, dz_dt):
    return (Dfd @ W).transpose() * dz_dt


def dD_dt(a_dot):
    return np.atleast_2d(a_dot)


def dDT_dt(a_dot):
    return dD_dt(a_dot).transpose()


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
