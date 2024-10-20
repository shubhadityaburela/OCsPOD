import line_profiler
import numpy as np
from jax import jit, jacobian
from numba import njit, prange
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from jax.scipy.optimize import minimize
from scipy.optimize import root
from scipy import integrate
import jax.numpy as jnp
import jax
from scipy import optimize
import sys


@njit
def RHS_PODG_FOTR_solve(Ar_p, a, psir_p, f):
    Ar_p_cont = np.ascontiguousarray(Ar_p)
    a_cont = np.ascontiguousarray(a)
    psir_p_cont = np.ascontiguousarray(psir_p)
    f_cont = np.ascontiguousarray(f)

    return Ar_p_cont @ a_cont + psir_p_cont @ f_cont


@njit(parallel=True)
def L2norm_FOM(qq, dx, dt):
    # Directly modify the input array if allowed
    q = qq.copy()  # Copy only if you need to keep qq unchanged

    # Scale the first and last column
    q[:, 0] /= np.sqrt(2)
    q[:, -1] /= np.sqrt(2)

    # Calculate the squared L2 norm
    # Instead of reshaping, iterate through the elements
    norm = 0.0
    rows, cols = q.shape
    for i in prange(rows):
        for j in range(cols):
            norm += q[i, j] ** 2

    return norm * dx * dt


@njit(parallel=True)
def L2norm_ROM(qq, dt):
    # Directly modify the input array if allowed
    q = qq.copy()  # Copy only if you need to keep qq unchanged

    # Scale the first and last column
    q[:, 0] /= np.sqrt(2)
    q[:, -1] /= np.sqrt(2)

    # Calculate the squared L2 norm
    # Instead of reshaping, iterate through the elements
    norm = 0.0
    rows, cols = q.shape
    for i in prange(rows):
        for j in range(cols):
            norm += q[i, j] ** 2

    return norm * dt


# Other Helper functions
def ControlSelectionMatrix_advection(wf, n_c):
    psi = np.zeros((wf.Nxi, n_c))
    for i in range(n_c):
        psi[:, i] = func(wf.X - wf.Lxi/n_c - i * wf.Lxi/n_c, sigma=1)

    # # Switching off the controls:
    # psi_small = np.zeros_like(psi)
    # psi_small[:, -15:] = psi[:, -15:]

    # print(n_c)
    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('TkAgg')
    # for i in range(n_c):
    #     plt.plot(psi_small[:, i])
    # plt.show()
    # exit()

    return psi


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
        result = optimize.minimize(objective, np.asarray([initial_guess_c]), args=(X, qs_0, qs_frame,))

        # Extract the optimal value of c
        optimal_c[:, i] = -result.x[0]
        initial_guess_c = result.x[0]

    return optimal_c


@line_profiler.profile
def compute_red_basis(qs, **kwargs):
    if kwargs['threshold']:
        U, S, VT = np.linalg.svd(qs, full_matrices=False)
        indices = np.where(S / S[0] > kwargs['base_tol'])[0]
        return U[:, :indices[-1] + 1], U[:, :indices[-1] + 1].dot(np.diag(S[:indices[-1] + 1]).dot(VT[:indices[-1] + 1, :]))
    else:
        U, S, VT = randomized_svd(qs, n_components=kwargs['Nm'], random_state=42)
        return U, U @ np.diag(S) @ VT

