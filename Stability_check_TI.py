import numpy as np
import matplotlib.pyplot as plt

from Coefficient_Matrix import CoefficientMatrix

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------
# BDF Methods Stability Boundaries
# ------------------------------

def bdf2_stability_boundary(theta):
    """
    Returns the stability boundary for BDF2.
    The parametric equation is:
        z(theta) = 3/2 - 2*exp(-i*theta) + 1/2*exp(-2*i*theta).
    """
    return 1.5 - 2 * np.exp(-1j * theta) + 0.5 * np.exp(-2j * theta)


def bdf3_stability_boundary(theta):
    """
    Returns the stability boundary for BDF3.
    One common parametric representation is:
        z(theta) = 11/6 - [3*exp(2*i*theta) - (3/2)*exp(i*theta) + 1/3] / exp(3*i*theta).
    """
    return (11 / 6) - (3 * np.exp(2j * theta) - (3 / 2) * np.exp(1j * theta) + 1 / 3) / np.exp(3j * theta)


def bdf4_stability_boundary(theta):
    """
    Returns the stability boundary for BDF4.
    One common parametric representation is:
        z(theta) = 25/12 - [4*exp(3*i*theta) - 3*exp(2*i*theta) + (4/3)*exp(i*theta) - 1/4] / exp(4*i*theta).
    """
    return (25 / 12) - (
                4 * np.exp(3j * theta) - 3 * np.exp(2j * theta) + (4 / 3) * np.exp(1j * theta) - 1 / 4) / np.exp(
        4j * theta)


# ------------------------------
# RK4 Stability Function
# ------------------------------

def rk4_R(z):
    """
    RK4 stability function:
        R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24.
    """
    return 1 + z + (z ** 2) / 2 + (z ** 3) / 6 + (z ** 4) / 24


# ------------------------------
# Plotting Functions
# ------------------------------

def plot_bdf_stability_region_and_eigenvalues(L, h, method="BDF2"):
    """
    Plots the stability boundary for BDF2, BDF3, or BDF4 along with the scaled eigenvalues of L.

    Parameters:
      L      : numpy array representing the discretized operator.
      h      : time step (so that z = h*λ).
      method : 'BDF2', 'BDF3', or 'BDF4'.
    """
    # Compute eigenvalues and scale them
    eigvals = np.linalg.eigvals(L)
    z_eigs = h * eigvals

    theta = np.linspace(0, 2 * np.pi, 400)
    if method.upper() == "BDF2":
        z_boundary = bdf2_stability_boundary(theta)
    elif method.upper() == "BDF3":
        z_boundary = bdf3_stability_boundary(theta)
    elif method.upper() == "BDF4":
        z_boundary = bdf4_stability_boundary(theta)
    else:
        raise ValueError("Method must be 'BDF2', 'BDF3', or 'BDF4'.")

    plt.figure(figsize=(8, 6))
    plt.plot(z_boundary.real, z_boundary.imag, 'k-', lw=2,
             label=f'{method} Stability Boundary')
    plt.scatter(z_eigs.real, z_eigs.imag, color='red', marker='o', s=50,
                label='Scaled eigenvalues (h*λ)')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title(f'{method} Stability Region (z = h*λ)')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='gray', lw=1)
    plt.axvline(0, color='gray', lw=1)
    plt.show()


def plot_rk4_stability_region_and_eigenvalues(L, h):
    """
    Plots the RK4 stability region (|R(z)|=1 contour) along with the scaled eigenvalues of L.

    Parameters:
      L : numpy array representing the discretized operator.
      h : time step (scaled eigenvalues z = h*λ).
    """
    eigvals = np.linalg.eigvals(L)
    z_eigs = h * eigvals

    # Create grid for complex plane
    re = np.linspace(-5, 5, 400)
    im = np.linspace(-5, 5, 400)
    RE, IM = np.meshgrid(re, im)
    Z = RE + 1j * IM

    R_val = np.abs(rk4_R(Z))

    plt.figure(figsize=(8, 6))
    contour = plt.contour(RE, IM, R_val, levels=[1], colors='k', linewidths=2)
    plt.clabel(contour, inline=True, fontsize=10)
    plt.scatter(z_eigs.real, z_eigs.imag, color='red', marker='o', s=50,
                label='Scaled eigenvalues (h*λ)')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title('RK4 Stability Region (|R(z)|=1) and Scaled Eigenvalues')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='gray', lw=1)
    plt.axvline(0, color='gray', lw=1)
    plt.show()


# Example usage:
# For demonstration, let's assume L is a 2x2 skew-symmetric matrix (typical of advection discretization).
Mat = CoefficientMatrix(orderDerivative="6thOrder", Nxi=3200,
                        Neta=1, periodicity='Periodic', dx=0.03125, dy=0)
# Convection matrix (Needs to be changed if the velocity is time dependent)
A_p = - 0.5 * Mat.Grad_Xi_kron
L = A_p.transpose().todense()
h = 0.04166666  # choose a time step

# Plot BDF stability regions along with scaled eigenvalues
plot_bdf_stability_region_and_eigenvalues(L, h, method="BDF2")
plot_bdf_stability_region_and_eigenvalues(L, h, method="BDF3")
plot_bdf_stability_region_and_eigenvalues(L, h, method="BDF4")

# Plot RK4 stability region along with scaled eigenvalues
plot_rk4_stability_region_and_eigenvalues(L, h)
