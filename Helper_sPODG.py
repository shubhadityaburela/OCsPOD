import line_profiler
import numpy as np
from scipy import sparse
import sys
import opt_einsum as oe

from Helper import is_contiguous

sys.path.append('./sPOD/lib/')

########################################################################################################################
# sPOD Galerkin helper functions
from transforms import Transform
from numba import njit, prange


@njit
def binary_search_interval(xPoints, xStar):
    # Perform a binary search to find the interval index
    left, right = 0, len(xPoints) - 1
    while left <= right:
        mid = (left + right) // 2
        if xPoints[mid] <= xStar < xPoints[mid + 1]:
            return mid
        elif xPoints[mid] < xStar:
            left = mid + 1
        else:
            right = mid - 1
    # Clamp to valid interval if xStar is out of bounds
    return max(0, min(left - 1, len(xPoints) - 2))


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


@njit
def subsample(X, num_sample):
    active_subspace_factor = -1
    delta_sampled = np.zeros((3, num_sample))

    delta_samples = np.linspace(0, X[-1], num_sample)

    delta_sampled[0, :] = active_subspace_factor * delta_samples
    delta_sampled[2, :] = delta_samples

    return np.ascontiguousarray(delta_sampled)


def get_T(delta_s, X, t, interp_order):
    Nx = len(X)
    Nt = len(t)

    data_shape = [Nx, 1, 1, Nt]
    dx = X[1] - X[0]
    L = [X[-1]]

    # Create the transformations
    trafo_1 = Transform(data_shape, L, shifts=delta_s[0],
                        dx=[dx],
                        use_scipy_transform=False,
                        interp_order=interp_order)

    return trafo_1.shifts_pos, trafo_1


def make_V_W_delta(U, T_delta, D, num_sample, Nx, modes):
    V_delta = np.zeros((num_sample, Nx, modes))
    W_delta = np.zeros((num_sample, Nx, modes))
    for it in range(num_sample):
        V_delta[it] = T_delta[it] @ U
        W_delta[it] = D @ V_delta[it]

    return np.ascontiguousarray(V_delta), np.ascontiguousarray(W_delta)


@njit
def findIntervalAndGiveInterpolationWeight_1D(xPoints, xStar):
    # # Find the interval index using the optimized binary search
    # intervalIdx = binary_search_interval(xPoints, xStar)
    #
    # # Calculate interpolation weight
    # x1, x2 = xPoints[intervalIdx], xPoints[intervalIdx + 1]
    # alpha = (x2 - xStar) / (x2 - x1)

    # Use binary search to find the interval index
    intervalIdx = np.searchsorted(xPoints, xStar) - 1

    # Ensure intervalIdx is within valid range
    intervalIdx = max(0, min(intervalIdx, len(xPoints) - 2))

    # Compute interpolation weight alpha
    x1, x2 = xPoints[intervalIdx], xPoints[intervalIdx + 1]
    alpha = (x2 - xStar) / (x2 - x1)

    return intervalIdx, alpha


@njit(parallel=True)
def findIntervals(delta_s, delta):
    Nt = len(delta)
    intIds = np.zeros(Nt, dtype=np.int32)
    weights = np.zeros(Nt)

    for i in prange(Nt):
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(delta_s[2], -delta[i])
        intIds[i] = intervalIdx
        weights[i] = weight

    return intIds, weights


######################################### FOTR sPOD functions #########################################

def LHS_offline_primal_FOTR(V_delta, W_delta, modes):
    # D(a) matrices are dynamic in nature thus need to be included in the time integration part
    LHS_mat = np.zeros((3, modes, modes))
    LHS_mat[0, ...] = oe.contract('ij,jk->ik', V_delta[0].T, V_delta[0])
    LHS_mat[1, ...] = oe.contract('ij,jk->ik', V_delta[0].T, W_delta[0])
    LHS_mat[2, ...] = oe.contract('ij,jk->ik', W_delta[0].T, W_delta[0])

    return np.ascontiguousarray(LHS_mat)


def RHS_offline_primal_FOTR(V_delta, W_delta, A, modes):
    RHS_mat = np.zeros((2, modes, modes))
    RHS_mat[0, ...] = V_delta[0].T @ A @ V_delta[0]
    RHS_mat[1, ...] = W_delta[0].T @ A @ V_delta[0]

    return np.ascontiguousarray(RHS_mat)


@njit(parallel=True)
def Control_offline_primal_FOTR(V_delta, W_delta, psi, samples, modes):
    # # Nice alternative and is equally faster
    # VW_delta = np.stack((V_delta, W_delta), axis=0)
    # C_mat = oe.contract("zabc,bd->zacd", VW_delta, psi)

    C_mat = np.zeros((2, samples, modes, psi.shape[1]), dtype=V_delta.dtype)

    for i in prange(samples):
        C_mat[0, i, :, :] = V_delta[i].T @ psi
        C_mat[1, i, :, :] = W_delta[i].T @ psi

    return np.ascontiguousarray(C_mat)


@njit
def Matrices_online_primal_FOTR(LHS_matrix, RHS_matrix, C, f, a, ds, modes):
    M = np.empty((modes + 1, modes + 1), dtype=LHS_matrix[0].dtype)
    A = np.empty(modes + 1)
    as_ = a[:-1]
    z = a[-1]

    # Compute the interpolation weight and the interval in which the shift lies
    intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], -z)

    Da = as_.reshape(-1, 1)

    M[:modes, :modes] = LHS_matrix[0]
    M[:modes, modes:] = LHS_matrix[1] @ Da
    M[modes:, :modes] = M[:modes, modes:].T
    M[modes:, modes:] = Da.T @ (LHS_matrix[2] @ Da)

    A[:modes] = RHS_matrix[0] @ as_ + np.add(weight * C[0, intervalIdx],
                                             (1 - weight) * C[0, intervalIdx + 1]) @ f
    A[modes:] = Da.T @ (RHS_matrix[1] @ as_ + np.add(weight * C[1, intervalIdx],
                                                     (1 - weight) * C[1, intervalIdx + 1]) @ f)

    return np.ascontiguousarray(M), np.ascontiguousarray(A), intervalIdx, weight


@njit
def solve_lin_system(M, A):
    return np.linalg.solve(M, A)


######################################### FRTO sPOD functions #########################################
def LHS_offline_primal_FRTO(V_delta, W_delta, modes):
    # D(a) matrices are dynamic in nature thus need to be included in the time integration part
    LHS_mat = np.zeros((3, modes, modes))
    LHS_mat[0, ...] = oe.contract('ij,jk->ik', V_delta[0].T, V_delta[0])
    LHS_mat[1, ...] = oe.contract('ij,jk->ik', V_delta[0].T, W_delta[0])
    LHS_mat[2, ...] = oe.contract('ij,jk->ik', W_delta[0].T, W_delta[0])

    return np.ascontiguousarray(LHS_mat)


def RHS_offline_primal_FRTO(V_delta, W_delta, A, modes):
    RHS_mat = np.zeros((2, modes, modes))
    RHS_mat[0, ...] = V_delta[0].T @ A @ V_delta[0]
    RHS_mat[1, ...] = W_delta[0].T @ A @ V_delta[0]

    return np.ascontiguousarray(RHS_mat)


@njit(parallel=True)
def Control_offline_primal_FRTO(V_delta, W_delta, psi, samples, modes):
    # # Nice alternative and is equally faster
    # VW_delta = np.stack((V_delta, W_delta), axis=0)
    # C_mat = oe.contract("zabc,bd->zacd", VW_delta, psi)

    C_mat = np.zeros((2, samples, modes, psi.shape[1]), dtype=V_delta.dtype)

    for i in prange(samples):
        C_mat[0, i, :, :] = V_delta[i].T @ psi
        C_mat[1, i, :, :] = W_delta[i].T @ psi

    return np.ascontiguousarray(C_mat)


@njit
def Matrices_online_primal_FRTO(LHS_matrix, RHS_matrix, C, f, a, ds, modes):
    M = np.empty((modes + 1, modes + 1), dtype=LHS_matrix[0].dtype)
    A = np.empty(modes + 1)
    as_ = a[:-1]
    z = a[-1]

    # Compute the interpolation weight and the interval in which the shift lies
    intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], -z)

    Da = as_.reshape(-1, 1)

    M[:modes, :modes] = LHS_matrix[0]
    M[:modes, modes:] = LHS_matrix[1] @ Da
    M[modes:, :modes] = M[:modes, modes:].T
    M[modes:, modes:] = Da.T @ (LHS_matrix[2] @ Da)

    A[:modes] = RHS_matrix[0] @ as_ + np.add(weight * C[0, intervalIdx],
                                             (1 - weight) * C[0, intervalIdx + 1]) @ f
    A[modes:] = Da.T @ (RHS_matrix[1] @ as_ + np.add(weight * C[1, intervalIdx],
                                                     (1 - weight) * C[1, intervalIdx + 1]) @ f)

    return np.ascontiguousarray(M), np.ascontiguousarray(A), intervalIdx, weight
