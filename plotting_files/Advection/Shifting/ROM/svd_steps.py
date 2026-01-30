import os

import numpy as np
from matplotlib import pyplot as plt
import sys

import argparse

parser = argparse.ArgumentParser(description="Input the variables for running the script.")
parser.add_argument("Problem", type=str, choices=["Shifting", "Shifting_3"], help="Choose the problem")
args = parser.parse_args()

problem = args.Problem

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"]})

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


if problem == "Shifting":
    impath = "../../../Plots/results_advection/Shifting/ROM/"
    os.makedirs(impath, exist_ok=True)

    svd = np.load("/Users/shubhadityaburela/Python/Paper4_OCsPOD/OCsPOD/data/Shifting/FOM/L1=0.0_L2=0.001/n_c=9/svd.npy")
else:
    impath = "../../../Plots/results_advection/Shifting_3/ROM/"
    os.makedirs(impath, exist_ok=True)

    svd = np.load(
        "/Users/shubhadityaburela/Python/Paper4_OCsPOD/OCsPOD/data/Shifting_3/FOM/L1=0.0_L2=0.001/n_c=9/svd.npy")


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


s1 = np.array(svd)[:, 9]
s2 = np.array(svd)[:, 10]

steps = np.arange(0, int(s1.shape[0]) * 10, 10, dtype=int)


fig1 = plt.figure(figsize=(9, 6))
ax1 = fig1.add_subplot(111)
ax1.plot(steps, s1, marker="*", label=r"$\sigma_{m + 1}$")
ax1.plot(steps, s2, marker="*", label=r"$\sigma_{m + 2}$")
ax1.set_xlabel(r"$n_{\mathrm{iter}}$")
ax1.set_ylabel(r"$\frac{\sigma}{\sigma_0}$")
ax1.set_yscale('log')
ax1.grid()
ax1.legend(loc='right')

fig1.tight_layout()
fig1.savefig(impath + problem + "_svd.pdf", dpi=300, transparent=True, format="pdf")
save_fig(impath + problem + "_svd", fig1)
