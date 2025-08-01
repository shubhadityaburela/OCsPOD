"""
This file is the version with Lagrange interpolation. It can handle both the scenarios.
1. Fixed tolerance
2. Fixed modes
"""
import os
import sys
import time
import argparse
import traceback
from time import perf_counter
from ast import literal_eval

import numpy as np
import scipy
from matplotlib import pyplot as plt, animation
import matplotlib

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
matplotlib.use('TkAgg')

# Add local sPOD library to path
sys.path.append('../sPOD/lib/')

# ──────────────── Local Modules ────────────────
from Coefficient_Matrix import CoefficientMatrix
from Cubic_spline import (
    give_spline_coefficient_matrices,
    construct_spline_coeffs_multiple,
    shift_matrix_precomputed_coeffs_multiple,
)
from FOM_solver import (
    IC_primal,
    TI_primal,
    TI_primal_target,
    IC_adjoint,
    TI_adjoint, IC_primal_kdvb, TI_primal_kdvb_impl, IC_adjoint_kdvb, TI_adjoint_kdvb_impl,
)
from Grads import (
    Calc_Grad_sPODG_FRTO_smooth,
    Calc_Grad_mapping,
)
from Update import (
    Update_Control_sPODG_FRTO_TWBT,
    Update_Control_sPODG_FRTO_BB,
    get_BB_step,
)
from grid_params import advection, Korteweg_de_Vries_Burgers
from Plots import PlotFlow
from Costs import Calc_Cost_sPODG, Calc_Cost

from Helper import (
    ControlSelectionMatrix,
    compute_red_basis,
    calc_shift,
    L2norm_ROM,
    check_weak_divergence, ControlSelectionMatrix_kdvb, compute_deim_basis,
)
from Helper_sPODG import (
    subsample,
    get_T,
    central_FDMatrix,
    central_FD2Matrix,
    make_V_W_U_delta,
    make_V_W_delta_CubSpl, get_approx_state_sPODG, findIntervals,
)

from sPODG_solver import (
    IC_primal_sPODG_FRTO,
    IC_adjoint_sPODG_FRTO,
    mat_primal_sPODG_FRTO,
    TI_primal_sPODG_FRTO,
    TI_adjoint_sPODG_FRTO, IC_primal_sPODG_FRTO_kdvb, IC_adjoint_sPODG_FRTO_kdvb, mat_primal_sPODG_FRTO_kdvb,
    TI_primal_sPODG_FRTO_kdvb_expl, TI_primal_sPODG_FRTO_kdvb_impl,
)

from sPOD_algo import give_interpolation_error


# ───────────────────────────────────────────────────────────────────────
def parse_arguments():
    p = argparse.ArgumentParser(description="Input the variables for running the script.")
    p.add_argument("primal_adjoint_common_basis", type=literal_eval, choices=[True, False],
                   help="Include adjoint in basis computation? (True/False)")
    p.add_argument("N_iter", type=int, help="Number of optimization iterations")
    p.add_argument("dir_prefix", type=str, choices=[".", "/work/burela"],
                   help="Directory prefix for I/O")
    p.add_argument("reg", type=float, nargs=2, metavar=("L1", "L2"),
                   help="L1 and L2 regularization weights (e.g. 0.01 0.001)")
    p.add_argument("CTC_mask_activate", type=literal_eval, choices=[True, False],
                   help="Include CTC mask in the system? (True/False)")
    p.add_argument("--modes", type=int, nargs=2,
                   help="Enter two mode numbers: [ROM_modes DEIM_modes], e.g., --modes 10 5")
    p.add_argument("--tol", type=float, nargs=2,
                   help="Enter two tolerances: [ROM_tol DEIM_tol], e.g., --tol 1e-6 1e-4")
    return p.parse_args()


def decide_run_type(args):
    if args.modes is not None and args.tol is not None:
        print("Modes test takes precedence…")
        TYPE = "modes"
        rom_modes, deim_modes = args.modes
        modes = (rom_modes, deim_modes)
        tol = (None, None)
        threshold = False
        VAL = modes
    # Only modes provided
    elif args.modes is not None:
        print("Modes test…")
        TYPE = "modes"
        rom_modes, deim_modes = args.modes
        modes = (rom_modes, deim_modes)
        tol = (None, None)
        threshold = False
        VAL = modes
    # Only tolerance provided
    elif args.tol is not None:
        print("Tolerance test…")
        TYPE = "tol"
        rom_tol, deim_tol = args.tol
        modes = (None, None)
        tol = (rom_tol, deim_tol)
        threshold = True
        VAL = tol
    else:
        print("ERROR: Must specify either --modes or --tol.")
        sys.exit(1)

    print(f"TYPE = {TYPE!r}, VAL = {VAL}")
    return TYPE, VAL, modes, tol, threshold


def setup_kdvb():
    Nx, Nt = 1000, 2000
    kdvb = Korteweg_de_Vries_Burgers(Nx=Nx, timesteps=Nt, cfl=0.4, v_x=220.0,
                                     variance=5e6, offset=30)
    kdvb.Grid()
    return kdvb


def C_matrix(Nx, CTC_end_index, apply_CTC_mask=False):  # For now not active
    C = np.ones(Nx)
    if apply_CTC_mask:
        return C == 1
    else:
        return C == 1


# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_arguments()

    print("Type of basis computation: fixed")
    print(f"Using adjoint in basis: {args.primal_adjoint_common_basis}")
    print(f"Using CTC mask pseudo hyperreduction: {args.CTC_mask_activate}")
    print(f"L1, L2 regularization = {tuple(args.reg)}")

    # Determine run type
    TYPE, VAL, modes, tol, threshold = decide_run_type(args)

    # Unpack regularization parameters
    L1_reg, L2_reg = args.reg

    # Set up kdvb and control matrix
    kdvb = setup_kdvb()
    if L1_reg != 0 and L2_reg == 0:  # Purely L1
        n_c_init = kdvb.Nx
        psi = ControlSelectionMatrix_kdvb(kdvb, n_c_init, Gaussian=False, gaussian_mask_sigma=0.5)
        adjust = 1.0
    else:  # Mix type
        n_c_init = 100
        psi = ControlSelectionMatrix_kdvb(kdvb, n_c_init, Gaussian=False, gaussian_mask_sigma=0.5)
        adjust = kdvb.dx
    n_c = psi.shape[1]

    # Prepare kwargs
    kwargs = {
        'dx': kdvb.dx,
        'dt': kdvb.dt,
        'Nx': kdvb.Nx,
        'Nt': kdvb.Nt,
        'n_c': n_c,
        'lamda_l1': L1_reg,
        'lamda_l2': L2_reg,
        'delta_conv': 1e-4,
        'delta': 1 / 2,  # Armijo constant  USE 1.01 with L1_reg=0 for getting the older results
        'opt_iter': args.N_iter,
        'beta': 1 / 2,  # for TWBT
        'verbose': True,
        'base_tol': tol[0],
        'deim_tol': tol[1],
        'omega_cutoff': 1e-10,
        'threshold': threshold,
        'Nm_p': modes[0],
        'Nm_deim_p': modes[1],
        'trafo_interp_order': 5,  # Order of the polynomial interpolation for the transformation operators
        'shift_sample': kdvb.Nx,  # Number of samples for shift interpolation
        'common_basis': args.primal_adjoint_common_basis
    }
    f = np.zeros((n_c, kdvb.Nt))  # initial control guess
    psi = scipy.sparse.csc_matrix(psi)

    # --------------------------------------------------------------------------------- #
    # BUILDING THE FOM DATA FIRST
    # Build coefficient matrices
    Mat = CoefficientMatrix(orderDerivative=kdvb.firstderivativeOrder,
                            Nxi=kdvb.Nx, Neta=1,
                            periodicity='Periodic',
                            dx=kdvb.dx, dy=0)
    A_p = - kdvb.v_x[0] * Mat.Grad_Xi_kron
    D1 = Mat.Grad_Xi_kron
    D2 = Mat.Grad_Xi_kron @ D1
    D3 = Mat.Grad_Xi_kron @ D2

    # Calculate the CTC matrix/array  (CTC and C are exactly the same)
    C = C_matrix(kdvb.Nx, None, apply_CTC_mask=args.CTC_mask_activate)
    CTC = C.copy()

    common_params = dict(A=A_p, D1=D1, D2=D2, D3=D3)
    shared_dynamics = {**common_params, 'B': psi}
    params_primal = {**shared_dynamics, 'omega': 0, 'gamma': 1e-2, 'nu': 1e-2}
    params_adjoint = {**common_params, 'CTC': CTC, 'omega': 0, 'gamma': 1e-2, 'nu': 1e-2}

    J_l = scipy.sparse.identity(kdvb.Nx, format='csc') - 0.5 * kdvb.dt * (
            A_p - params_primal['gamma'] * D3 + params_primal['nu'] * D2)
    J_l_adjoint = scipy.sparse.identity(kdvb.Nx, format='csc') + 0.5 * kdvb.dt * (
            - A_p.T + params_primal['gamma'] * D3.T - params_primal['nu'] * D2.T)

    # Solve uncontrolled FOM once
    qs0 = IC_primal_kdvb(kdvb.X, kdvb.Lx, kdvb.offset, kdvb.variance)
    qs_org = TI_primal_kdvb_impl(qs0, f, J_l, kdvb.Nx, kdvb.Nt, kdvb.dt, **params_primal)
    q0 = np.ascontiguousarray(qs0)
    q0_adj = np.ascontiguousarray(IC_adjoint_kdvb(kdvb.X))

    # --------------------------------------------------------------------------------- #
    # Precompute full basis once (fixed‐basis approach)

    # Sample the shift values
    # Extract transformation operators based on sub-sampled delta
    delta_s = subsample(kdvb.X, num_sample=kwargs['shift_sample'])
    T_delta, _ = get_T(delta_s, kdvb.X, kdvb.t, interp_order=kwargs['trafo_interp_order'])
    qs_full = qs_org.copy()
    # Compute shifts of the primal uncontrolled profile
    z = calc_shift(qs_full, q0, kdvb.X, kdvb.t)
    _, T = get_T(z, kdvb.X, kdvb.t, interp_order=kwargs['trafo_interp_order'])
    snap_cat_p_s = T.reverse(qs_full).copy()

    # Compute reduced bases
    for i in list([[2, 2], [6, 6], [15, 15], [20, 20], [30, 30], [40, 40]]):
        kwargs['Nm_p'] = int(i[0])
        kwargs['Nm_deim_p'] = int(i[1])

        V, qs_sPOD = compute_red_basis(snap_cat_p_s, equation="primal", **kwargs)
        Nm = V.shape[1]
        err = np.linalg.norm(snap_cat_p_s - qs_sPOD) / np.linalg.norm(snap_cat_p_s)
        print(f"Primal basis: Nm_p={Nm}, err={err:.3e}")

        # DEIM TO BE SOLVED
        qs_full_deim = qs_full * (D1 @ qs_full)  # DEIM snapshot matrix
        qs_full_deim_s = T.reverse(qs_full_deim)
        V_deim, qs_sPOD_deim = compute_deim_basis(qs_full_deim_s, equation="primal", **kwargs)
        Nm_deim = V_deim.shape[1]
        err = np.linalg.norm(qs_full_deim_s - qs_sPOD_deim) / np.linalg.norm(qs_full_deim_s)
        print(f"DEIM basis: Nm_deim={Nm_deim}, err={err:.3e}")

        # Initial condition for dynamical simulation
        a_p = IC_primal_sPODG_FRTO_kdvb(q0, V)
        a_a = IC_adjoint_sPODG_FRTO_kdvb(Nm)

        # Construct the primal system matrices for the sPOD-Galerkin approach
        Vd_p, Wd_p, Ud_p = make_V_W_U_delta(V, T_delta, D1, D2, kwargs['shift_sample'], kwargs['Nx'], Nm)

        # Construct the primal and adjoint system matrices for the sPOD-Galerkin approach
        lhs_p, rhs_p, deim_p, deim_p_hl, c_p = mat_primal_sPODG_FRTO_kdvb(T_delta, Vd_p, Wd_p, Ud_p, V_deim,
                                                                          kwargs['shift_sample'],
                                                                          Nm, Nm_deim, params_primal, delta_s)

        # ───── Forward ROM: compute ROM state a_p → as_p ─────
        as_p, as_dot, intIds, weights = TI_primal_sPODG_FRTO_kdvb_impl(lhs_p, rhs_p, deim_p, deim_p_hl, c_p, a_p, f,
                                                                       delta_s, Nm, kwargs['Nt'], kwargs['dt'])

        # as_p, as_dot, intIds, weights = TI_primal_sPODG_FRTO_kdvb_expl(lhs_p, rhs_p, deim_p, c_p, a_p, f,
        #                                                 delta_s, Nm, kwargs['Nt'], kwargs['dt'])

        qqqq = get_approx_state_sPODG(Vd_p, f, as_p[:-1, :], intIds, weights, kwargs['Nx'], kwargs['Nt'])

        print(np.linalg.norm(qs_org - qqqq) / np.linalg.norm(qs_org))
