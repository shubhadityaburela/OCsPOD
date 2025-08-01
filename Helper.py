import numpy as np
from matplotlib import pyplot as plt
from numba import njit, prange
from scipy.signal import savgol_filter
from sklearn.utils.extmath import randomized_svd
from scipy import optimize
from scipy.linalg import svd, eig, pinv


@njit
def BarzilaiBorwein(itr, dt, fNew, fOld, gNew, gOld):
    # Computes the step size according to Barzilai-Borwein method
    SY = L2inner_prod(fNew - fOld, gNew - gOld, dt)

    if itr % 2 == 0:  # Even number of iterations
        SS = L2inner_prod(fNew - fOld, fNew - fOld, dt)
        return SY / SS
    else:  # Odd number of iterations
        YY = L2inner_prod(gNew - gOld, gNew - gOld, dt)
        return YY / SY


@njit
def L2norm_FOM(qq, dx, dt):
    # Directly modify the input array if allowed
    q = qq.copy()  # Copy only if you need to keep qq unchanged

    # Scale the first and last column
    q[:, 0] /= np.sqrt(2.0)
    q[:, -1] /= np.sqrt(2.0)

    # Calculate the squared L2 norm
    norm = np.sum(q ** 2)

    return norm * dx * dt


@njit
def L2norm_ROM(qq, dt):
    # Directly modify the input array if allowed
    q = qq.copy()  # Copy only if you need to keep qq unchanged

    # Scale the first and last column
    q[:, 0] /= np.sqrt(2.0)
    q[:, -1] /= np.sqrt(2.0)

    # Calculate the squared L2 norm
    norm = np.sum(q ** 2)

    return norm * dt


@njit
def L2norm_ROM_1D(qq, dt):
    # Directly modify the input array if allowed
    q = qq.copy()  # Copy only if you need to keep qq unchanged

    # Scale the first and last column
    q[0] /= np.sqrt(2.0)
    q[-1] /= np.sqrt(2.0)

    # Calculate the squared L2 norm
    norm = np.sum(q ** 2)

    return norm * dt


@njit
def L2inner_prod(qq1, qq2, dt):
    # Directly modify the input array if allowed
    q1 = qq1.copy()  # Copy only if you need to keep uu unchanged
    q2 = qq2.copy()

    # Scale the first and last column
    q1[:, 0] /= np.sqrt(2.0)
    q1[:, -1] /= np.sqrt(2.0)

    q2[:, 0] /= np.sqrt(2.0)
    q2[:, -1] /= np.sqrt(2.0)

    prod = np.sum(q1 * q2)
    return prod * dt


# Other Helper functions
def ControlSelectionMatrix(wf, n_c, Gaussian=False, trim_first_n=0, gaussian_mask_sigma=1,
                           start_controlling_from=0):
    if trim_first_n >= n_c:
        print("Number of controls should always be more than the number which you want to trim out. "
              "Set it accordingly. Exiting !!!!!!!")
        exit()
    psi = np.zeros((wf.Nxi, n_c - trim_first_n), order="F")
    if Gaussian:
        for i in range(n_c - trim_first_n):
            psi[:, i] = func(wf.X - wf.Lxi / n_c - (trim_first_n + i) * wf.Lxi / n_c,
                             sigma=gaussian_mask_sigma)  # Could also divide the middle quantity by 2 for similar non-overlapping gaussians
    else:
        control_index = np.array_split(np.arange(start_controlling_from, wf.Nxi), n_c)
        for i in range(n_c - trim_first_n):
            psi[control_index[trim_first_n + i], i] = 1.0

    return np.ascontiguousarray(psi)


def ControlSelectionMatrix_kdvb(wf, n_c, Gaussian=False, trim_first_n=0, gaussian_mask_sigma=1,
                                start_controlling_from=0):
    if trim_first_n >= n_c:
        print("Number of controls should always be more than the number which you want to trim out. "
              "Set it accordingly. Exiting !!!!!!!")
        exit()
    psi = np.zeros((wf.Nx, n_c - trim_first_n), order="F")
    if Gaussian:
        for i in range(n_c - trim_first_n):
            psi[:, i] = func(wf.X - wf.Lx / n_c - (trim_first_n + i) * wf.Lx / n_c,
                             sigma=gaussian_mask_sigma)  # Could also divide the middle quantity by 2 for similar non-overlapping gaussians
    else:
        control_index = np.array_split(np.arange(start_controlling_from, wf.Nx), n_c)
        for i in range(n_c - trim_first_n):
            psi[control_index[trim_first_n + i], i] = 1.0

    return np.ascontiguousarray(psi)


def func(x, sigma):
    return np.exp(-x ** 2 / sigma ** 2)


def objective(c, X, qs_0, qs_frame):
    interpolated_f = np.interp(X + c, X, qs_frame, period=X[-1])
    error = interpolated_f - qs_0

    squared_error = np.linalg.norm(error)
    return squared_error


def calc_shift(qs, qs_0, X, t):
    # Initial guess for c
    Nt = len(t)
    initial_guess_c = 0
    optimal_c = np.zeros((1, Nt))

    for i in range(Nt):
        qs_frame = qs[:, i]
        # Minimize the objective function with respect to c
        result = optimize.minimize(objective, np.asarray([initial_guess_c]), args=(X, qs_0, qs_frame,),
                                   method="Nelder-mead")
        # Extract the optimal value of c
        optimal_c[:, i] = -result.x[0]
        initial_guess_c = result.x[0]

    return optimal_c


def compute_red_basis(qs, equation="primal", **kwargs):
    if kwargs['threshold']:
        U, S, VT = np.linalg.svd(qs, full_matrices=False)
        indices = np.where(S / S[0] > kwargs['base_tol'])[0]
        return np.ascontiguousarray(U[:, :indices[-1] + 1]), np.ascontiguousarray(U[:, :indices[-1] + 1].dot(
            np.diag(S[:indices[-1] + 1]).dot(VT[:indices[-1] + 1, :])))
    else:
        if equation == "primal":
            U, S, VT = randomized_svd(qs, n_components=kwargs['Nm_p'], random_state=42)
            return np.ascontiguousarray(U), np.ascontiguousarray(U @ np.diag(S) @ VT)
        else:
            U, S, VT = randomized_svd(qs, n_components=kwargs['Nm_a'], random_state=42)
            return np.ascontiguousarray(U), np.ascontiguousarray(U @ np.diag(S) @ VT)


def compute_deim_basis(qs, equation="primal", **kwargs):
    if kwargs['threshold']:
        U, S, VT = np.linalg.svd(qs, full_matrices=False)
        indices = np.where(S / S[0] > kwargs['deim_tol'])[0]
        return np.ascontiguousarray(U[:, :indices[-1] + 1]), np.ascontiguousarray(U[:, :indices[-1] + 1].dot(
            np.diag(S[:indices[-1] + 1]).dot(VT[:indices[-1] + 1, :])))
    else:
        if equation == "primal":
            U, S, VT = randomized_svd(qs, n_components=kwargs['Nm_deim_p'], random_state=42)
            return np.ascontiguousarray(U), np.ascontiguousarray(U @ np.diag(S) @ VT)
        else:
            U, S, VT = randomized_svd(qs, n_components=kwargs['Nm_deim_a'], random_state=42)
            return np.ascontiguousarray(U), np.ascontiguousarray(U @ np.diag(S) @ VT)


def is_contiguous(array):
    return array.flags['C_CONTIGUOUS'] or array.flags['F_CONTIGUOUS']


@njit
def L1norm_ROM(qq, dt):
    # Directly modify the input array if allowed
    q = qq.copy()  # Copy only if you need to keep qq unchanged

    # Scale the first and last column
    q[:, 0] /= 2.0
    q[:, -1] /= 2.0

    # Calculate the absolute L1 norm
    norm = np.sum(np.abs(q))

    return norm * dt


@njit
def L1norm_FOM(qq, dx, dt, adjust):
    # Directly modify the input array if allowed
    q = qq.copy()  # Copy only if you need to keep qq unchanged

    # Scale the first and last column
    q[:, 0] /= 2.0
    q[:, -1] /= 2.0

    # Calculate the L1 norm
    norm = np.sum(np.abs(q))

    return norm * dx * dt / adjust


def check_weak_divergence(J_opt_FOM_list, window=100, margin=0.0):
    """
    Check for “weaker divergence” by comparing the mean of the last `window` entries
    against the mean of the preceding `window` entries in J_opt_FOM_list.

    Returns (is_diverging, avg_prev, avg_last), where:
      - is_diverging is True if avg_last > avg_prev * (1 + margin)
      - avg_prev is the mean of J_opt_FOM_list[-2*window : -window]
      - avg_last is the mean of J_opt_FOM_list[-window :]
      - If len(J_opt_FOM_list) < 2*window, returns (False, None, None).
    """
    n = len(J_opt_FOM_list)
    if n < 2 * window:
        return False, None, None

    prev_window = J_opt_FOM_list[-2 * window:-window]
    last_window = J_opt_FOM_list[-window:]

    avg_prev = sum(prev_window) / window
    avg_last = sum(last_window) / window

    is_diverging = (avg_last > avg_prev * (1 + margin))
    return is_diverging, avg_prev, avg_last


@njit(parallel=True)
def compute_state(V, Nx, Nt, as_adj, intIds, weights):
    qs = np.zeros((Nx, Nt))

    for i in prange(Nt):
        V_idx = intIds[i]
        V_delta = weights[i] * V[V_idx] + (1 - weights[i]) * V[V_idx + 1]
        qs[:, i] = V_delta @ as_adj[:, i]

    return qs
