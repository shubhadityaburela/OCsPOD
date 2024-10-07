import numpy as np
from scipy import sparse
import sys

from Helper_sPODG_FRTO import Q11, Q12, Q21, Q22, B11, B12, B21, B22, C1, C2, Z11, Z12, Z21, Z22, VT_dash, WT_dash, \
    V_dash

sys.path.append('./sPOD/lib/')

########################################################################################################################
# sPOD Galerkin helper functions
from transforms import Transform
from numba import njit


# General functions
def central_FDMatrix(order, Nx, dx):
    if order == 2:
        pass
    elif order == 4:
        pass
    elif order == 6:
        Coeffs = np.array([-1, 9, -45, 0, 45, -9, 1]) / 60
        diagonalLow = int(-(len(Coeffs) - 1) / 2)
        diagonalUp = int(-diagonalLow)

        D_1 = sparse.csr_matrix(np.zeros((Nx, Nx), dtype=float))

        for k in range(diagonalLow, diagonalUp + 1):
            D_1 = D_1 + Coeffs[k - diagonalLow] * sparse.csr_matrix(np.diag(np.ones(Nx - abs(k)), k))
            if k < 0:
                D_1 = D_1 + Coeffs[k - diagonalLow] * sparse.csr_matrix(
                    np.diag(np.ones(abs(k)), Nx + k))
            if k > 0:
                D_1 = D_1 + Coeffs[k - diagonalLow] * sparse.csr_matrix(
                    np.diag(np.ones(abs(k)), -Nx + k))

    return D_1 * (1 / dx)


def subsample(X, num_sample):
    active_subspace_factor = -1

    # sampling points for the shifts (The shift values can range from 0 to X/2 and then is a mirror image for X/2 to X)
    delta_samples = np.linspace(0, X[-1], num_sample)

    delta_sampled = [active_subspace_factor * delta_samples,
                     np.zeros_like(delta_samples),
                     delta_samples]

    return np.array(delta_sampled)


def get_T(delta_s, X, t):
    Nx = len(X)
    Nt = len(t)

    data_shape = [Nx, 1, 1, Nt]
    dx = X[1] - X[0]
    L = [X[-1]]

    # Create the transformations
    trafo_1 = Transform(data_shape, L, shifts=delta_s[0],
                         dx=[dx],
                         use_scipy_transform=False,
                         interp_order=5)

    return trafo_1.shifts_pos, trafo_1


def make_V_W_delta(U, T_delta, D, num_sample):

    V_delta = [T_delta[it] @ U for it in range(num_sample)]
    W_delta = [D @ (T_delta[it] @ U) for it in range(num_sample)]

    return V_delta, W_delta


@njit
def findIntervalAndGiveInterpolationWeight_1D(xPoints, xStar):
    intervalBool_arr = np.where(xStar >= xPoints, 1, 0)
    mixed = intervalBool_arr[:-1] * (1 - intervalBool_arr)[1:]
    index = np.sum(mixed * np.arange(0, mixed.shape[0]))

    intervalIdx = index
    alpha = (xPoints[intervalIdx + 1] - xStar) / (
            xPoints[intervalIdx + 1] - xPoints[intervalIdx])

    return intervalIdx, alpha


def make_Da(a):
    D_a = np.copy(a[:len(a) - 1])

    return D_a[:, np.newaxis]


def get_online_state(T_trafo, V, a, X, t):
    Nx = len(X)
    Nt = len(t)
    qs_online = np.zeros((Nx, Nt))
    q = V @ a

    qs_online += T_trafo[0].apply(q)

    return qs_online


@njit
def findIntervals(delta_s, delta):
    Nt = len(delta)
    intIds = []
    weights = []

    for i in range(Nt):
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(delta_s[2], -delta[i])
        intIds.append(intervalIdx)
        weights.append(weight)

    return intIds, weights


######################################### FRTO sPOD functions #########################################
def LHS_offline_primal_FRTO(V_delta, W_delta):
    # D(a) matrices are dynamic in nature thus need to be included in the time integration part
    LHS11 = V_delta[0].transpose() @ V_delta[0]
    LHS12 = V_delta[0].transpose() @ W_delta[0]
    LHS22 = W_delta[0].transpose() @ W_delta[0]

    LHS_mat = [LHS11, LHS12, LHS22]

    return LHS_mat


def RHS_offline_primal_FRTO(V_delta, W_delta, A):
    A_1 = (V_delta[0].transpose() @ A) @ V_delta[0]
    A_2 = (W_delta[0].transpose() @ A) @ V_delta[0]

    RHS_mat = [A_1, A_2]

    return RHS_mat


def Control_offline_primal_FRTO(V_delta, W_delta, psi, D):
    C_mat = []

    for it in range(len(V_delta)):
        C_1 = (V_delta[it].transpose() @ psi)
        C_2 = (W_delta[it].transpose() @ psi)
        C_3 = VT_dash(D, V_delta[it]) @ psi
        C_4 = WT_dash(D, W_delta[it]) @ psi

        C_mat.append([C_1, C_2, C_3, C_4])

    return C_mat


def Target_offline_adjoint_FRTO(D, V_delta):
    T_mat = []

    for it in range(len(V_delta)):
        T_mat.append(V_dash(D, V_delta[it]).transpose() @ V_delta[it])

    return T_mat


def LHS_online_primal_FRTO(LHS_matrix, Da):
    M11 = np.copy(LHS_matrix[0])
    M12 = LHS_matrix[1] @ Da
    M21 = M12.transpose()
    M22 = (Da.transpose() @ LHS_matrix[2]) @ Da

    M = np.block([
        [M11, M12],
        [M21, M22]
    ])

    return M


def RHS_online_primal_FRTO(RHS_matrix, Da):
    A11 = np.copy(RHS_matrix[0])
    A21 = Da.transpose() @ RHS_matrix[1]

    A = np.block([
        [A11, np.zeros((A11.shape[0], Da.shape[1]))],
        [A21, np.zeros((A21.shape[0], Da.shape[1]))]
    ])

    return A


def Control_online_primal_FRTO(f, C, Da, intervalIdx, weight):
    C1 = (weight * C[intervalIdx][0] + (1 - weight) * C[intervalIdx + 1][0]) @ f
    C2 = Da.transpose() @ ((weight * C[intervalIdx][1] + (1 - weight) * C[intervalIdx + 1][1]) @ f)

    C = np.concatenate((C1, C2))

    return C


def LHS_online_adjoint_FRTO(LHS_matrix, Da):
    M1 = np.copy(LHS_matrix[0])
    N = np.copy(LHS_matrix[1])
    M2 = np.copy(LHS_matrix[2])

    Q1_1_red = Q11(M1)
    Q1_2_red = Q12(N, Da)
    Q2_1_red = Q21(N, Da)
    Q2_2_red = Q22(M2, Da)

    LHS = np.block([
        [Q1_1_red.transpose(), Q1_2_red.transpose()],
        [Q2_1_red.transpose(), Q2_2_red.transpose()]
    ])

    return LHS


def RHS_online_adjoint_FRTO(RHS_matrix, LHS_matrix, C_matrix, psi, a_dot, z_dot, Da, a_, Dfd, Vdp, Wdp, u,
                            intervalIdx, weight):
    r = len(a_) - 1
    a_s = np.atleast_2d(a_[:-1]).T
    z = np.atleast_2d(a_[-1:]).T
    a_dot = np.atleast_2d(a_dot).T
    z_dot = np.atleast_2d(z_dot).T
    u = np.atleast_2d(u).T

    WTB = weight * C_matrix[intervalIdx][1] + (1 - weight) * C_matrix[intervalIdx + 1][1]
    VTdashB = weight * C_matrix[intervalIdx][2] + (1 - weight) * C_matrix[intervalIdx + 1][2]
    WTdashB = weight * C_matrix[intervalIdx][3] + (1 - weight) * C_matrix[intervalIdx + 1][3]

    N = LHS_matrix[1]
    M2 = LHS_matrix[2]
    A1 = RHS_matrix[0]
    A2 = RHS_matrix[1]

    B1_1_red = B11(N, A1, z_dot, r)
    B1_2_red = B12(M2, N, A2, Da, WTB, a_dot, z_dot, a_s, u, r)
    B2_1_red = B21(Dfd, N, VTdashB, a_dot, u)
    B2_2_red = B22(Dfd, M2, Da, WTdashB, a_dot, u)

    RHS = np.block([
        [B1_1_red.transpose(), B1_2_red.transpose()],
        [B2_1_red.transpose(), B2_2_red.transpose()]
    ])

    return RHS


def Target_online_adjoint_FRTO(a_, Dfd, Vdp, qs_target, Tp, intervalIdx, weight):
    V = weight * Vdp[intervalIdx] + (1 - weight) * Vdp[intervalIdx + 1]
    T = weight * Tp[intervalIdx] + (1 - weight) * Tp[intervalIdx + 1]
    C1_1 = C1(V, qs_target, a_[:-1])
    C2_1 = C2(Dfd, V, qs_target, T, a_[:-1])

    C = np.concatenate((C1_1, C2_1))

    return C


def check_invertability_FRTO(LHS_matrix, a_):
    M1 = np.copy(LHS_matrix[0])
    N = np.copy(LHS_matrix[1])
    M2 = np.copy(LHS_matrix[2])
    D = make_Da(a_)

    Z1_1_red = Z11(M1)
    Z1_2_red = Z12(N, D)
    Z2_1_red = Z21(N, D)
    Z2_2_red = Z22(M2, D)

    Z = np.block([
        [Z1_1_red.transpose(), Z1_2_red.transpose()],
        [Z2_1_red.transpose(), Z2_2_red.transpose()]
    ])

    if np.linalg.cond(Z) < 1 / sys.float_info.epsilon:
        return True
    else:
        return False


######################################### FOTR sPOD functions #########################################

def LHS_offline_primal_FOTR(V_delta, W_delta):
    # D(a) matrices are dynamic in nature thus need to be included in the time integration part
    LHS11 = V_delta[0].transpose() @ V_delta[0]
    LHS12 = V_delta[0].transpose() @ W_delta[0]
    LHS22 = W_delta[0].transpose() @ W_delta[0]

    LHS_mat = [LHS11, LHS12, LHS22]

    return LHS_mat


def RHS_offline_primal_FOTR(V_delta, W_delta, A):
    A_1 = (V_delta[0].transpose() @ A) @ V_delta[0]
    A_2 = (W_delta[0].transpose() @ A) @ V_delta[0]

    RHS_mat = [A_1, A_2]

    return RHS_mat


def Control_offline_primal_FOTR(V_delta, W_delta, psi):
    C_mat = []
    Ns = len(V_delta)

    for it in range(Ns):
        C_1 = (V_delta[it].transpose() @ psi)
        C_2 = (W_delta[it].transpose() @ psi)

        C_mat.append([C_1, C_2])

    # print(timeit.timeit(tmp1, number=1000))
    # print(timeit.timeit(tmp2, number=1000))

    return C_mat



def LHS_online_primal_FOTR(LHS_matrix, Da):
    M11 = np.copy(LHS_matrix[0])
    M12 = LHS_matrix[1] @ Da
    M21 = M12.transpose()
    M22 = (Da.transpose() @ LHS_matrix[2]) @ Da

    M = np.block([
        [M11, M12],
        [M21, M22]
    ])

    return M


def RHS_online_primal_FOTR(RHS_matrix, Da):
    A11 = np.copy(RHS_matrix[0])
    A21 = Da.transpose() @ RHS_matrix[1]

    A = np.block([
        [A11, np.zeros((A11.shape[0], 1))],
        [A21, np.zeros((A21.shape[0], 1))]
    ])

    return A



def Control_online_primal_FOTR(f, C, Da, intervalIdx, weight):
    C1 = (weight * C[intervalIdx][0] + (1 - weight) * C[intervalIdx + 1][0]) @ f
    C2 = Da.transpose() @ ((weight * C[intervalIdx][1] + (1 - weight) * C[intervalIdx + 1][1]) @ f)

    C = np.concatenate((C1, C2))

    return C