import numpy as np
from jax import jit, jacobian
from sklearn.utils.extmath import randomized_svd
from jax.scipy.optimize import minimize
from scipy.optimize import root
from scipy import integrate
import jax.numpy as jnp
import jax
from scipy import optimize


def trapezoidal_integration(q, **kwargs):
    return integrate.trapezoid(integrate.trapezoid(np.square(q), axis=0, dx=kwargs['dx']), axis=0, dx=kwargs['dt'])


def trapezoidal_integration_control(q, **kwargs):
    return integrate.trapezoid(integrate.trapezoid(np.square(q), axis=0), axis=0, dx=kwargs['dt'])



def integrate_cost(q, **kwargs):
    q[:, 0] = q[:, 0] / 2
    q[:, -1] = q[:, -1] / 2
    q = q.reshape((-1))
    return np.sum(np.square(q)) * kwargs.get('dx') * kwargs.get('dt')



def integrate_cost_TA(q, **kwargs):
    q[:, 0] = q[:, 0] / 2
    q[:, -1] = q[:, -1] / 2
    q = q.reshape((-1))
    return np.sum(np.square(q)) * kwargs.get('dt')


def L2norm_ROM(q, **kwargs):
    q[:, 0] = q[:, 0] / 2
    q[:, -1] = q[:, -1] / 2
    q = q.reshape((-1))
    return np.sum(np.square(q)) * kwargs.get('dt')


# Other Helper functions
def ControlSelectionMatrix_advection(wf, n_c):
    psi = np.zeros((wf.Nxi, n_c))
    for i in range(n_c):
        psi[:, i] = func(wf.X - 0.25 - i * 0.25, sigma=2)  # wf.X - 2.5 - i * 2.5, sigma=4

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


def compute_red_basis(qs, nm):
    U, S, VT = randomized_svd(qs, n_components=nm, random_state=0)

    return U[:, :nm], U[:, :nm].dot(np.diag(S[:nm]).dot(VT[:nm, :]))

