import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
import glob

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"]})

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def save_fig(filepath, figure=None, **kwargs):
    import tikzplotlib
    import os
    import matplotlib.pyplot as plt

    ## split extension
    fpath = os.path.splitext(filepath)[0]
    ## get figure handle
    if figure is None:
        figure = plt.gcf()
    figure.savefig(fpath + ".png", dpi=200, transparent=True)
    tikzplotlib.save(
        figure=figure,
        filepath=fpath + ".tex",
        axis_height='\\figureheight',
        axis_width='\\figurewidth',
        override_externals=True,
        **kwargs
    )


class PlotFlow:
    def __init__(self, X, t) -> None:

        self.Nx = int(np.size(X))
        self.Nt = int(np.size(t))

        self.X = X
        self.t = t

        # Prepare the space-time grid for 1D plots
        self.X_1D_grid, self.t_grid = np.meshgrid(X, t)
        self.X_1D_grid = self.X_1D_grid.T
        self.t_grid = self.t_grid.T

    def plot1D(self, Q, name, immpath):
        os.makedirs(immpath, exist_ok=True)

        # Plot the snapshot matrix for conserved variables for original model
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(111)
        im1 = ax1.pcolormesh(self.X_1D_grid, self.t_grid, Q, cmap='YlOrRd')
        ax1.axis('off')
        ax1.set_title(r"$q(x, t)$")
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='10%', pad=0.08)
        fig.colorbar(im1, cax=cax, orientation='vertical')

        fig.supylabel(r"time $t$")
        fig.supxlabel(r"space $x$")

        out_file = os.path.join(immpath, f"{name}.png")
        fig.savefig(out_file, dpi=300, transparent=True)

    def plot1D_FOM_converg(self, J, name, immpath):

        os.makedirs(immpath, exist_ok=True)

        fig1 = plt.figure(figsize=(8, 8))
        ax1 = fig1.add_subplot(111, label="1")
        ax1.semilogy(np.arange(len(J)), J, color="C0", label=r"$\mathcal{J}$")
        ax1.set_xlabel(r"$n_{\mathrm{iter}}$", color="C0")
        ax1.set_ylabel(r"$J$", color="C0")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.tick_params(axis='x', colors="C0")
        ax1.tick_params(axis='y', colors="C0")

        out_file = os.path.join(immpath, f"{name}.png")
        fig1.savefig(out_file, dpi=300, transparent=True)

    def plot1D_ROM_converg(self, J_ROM, J_FOM, name, immpath):

        os.makedirs(immpath, exist_ok=True)

        fig1 = plt.figure(figsize=(8, 8))
        ax1 = fig1.add_subplot(111, label="1")
        ax1.semilogy(np.arange(len(J_ROM)), J_ROM, color="C0", label=r"$\mathcal{J}_\mathrm{ROM}$")
        ax1.semilogy(np.arange(len(J_FOM)), J_FOM, color="C1", label=r"$\mathcal{J}_\mathrm{FOM}$")
        ax1.set_xlabel(r"$n_{\mathrm{iter}}$", color="C0")
        ax1.set_ylabel(r"$J$", color="C0")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.tick_params(axis='x', colors="C0")
        ax1.tick_params(axis='y', colors="C0")
        ax1.legend()

        out_file = os.path.join(immpath, f"{name}.png")
        fig1.savefig(out_file, dpi=300, transparent=True)


def plot_normalized_singular_values(X, sv, semilogy=False, savepath=None, name=None, id=None):
    """
    Compute singular values of snapshot matrix X, normalize by the largest,
    and plot them.

    Parameters
    ----------
    X : array_like, shape (m, n)
        Snapshot matrix (rows = DOFs, columns = snapshots) or vice-versa.
    semilogy : bool, optional
        If True, plot y-axis on a log scale (useful when values decay fast).
    savepath : str or None, optional
        If given, save the figure to this path (e.g. "sv.png").

    Returns
    -------
    s : 1D numpy array
        Singular values (not normalized), sorted descending.
    s_norm : 1D numpy array
        Singular values normalized by the largest (s / s[0]).
    """

    # compute singular values (fast: no U/V returned)
    s = np.linalg.svd(X, compute_uv=False)
    s = s[:sv]
    if s.size == 0:
        raise ValueError("No singular values found (empty input?).")

    s_norm = s / s[0]  # normalize by largest singular value

    idx = np.arange(1, len(s_norm) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    if semilogy:
        ax.semilogy(idx, s_norm,
                    color="brown",
                    marker="+",
                    linestyle='None',
                    markersize=5)
        ax.set_ylabel(r"$\sigma_{k} / \sigma_{0}$")
    else:
        ax.plot(idx, s_norm,
                color="brown",
                marker="+",
                linestyle='None',
                markersize=5)
        ax.set_ylabel(r"$\sigma_{k} / \sigma_{0}$")

    ax.set_xlabel(r"Num. of singular vals.")
    ax.set_title(rf"$n_\mathrm{{opt}} = {id}$")
    ax.grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath + name + '_' + str(id), dpi=300, bbox_inches='tight')

    return s, s_norm