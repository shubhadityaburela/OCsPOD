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
import scipy.sparse as sp
from matplotlib import pyplot as plt

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
    TI_adjoint,
)
from Grads import (
    Calc_Grad_sPODG_FRTO_smooth,
    Calc_Grad_mapping,
)
from Update import (
    Update_Control_sPODG_FRTO_TWBT,
    get_BB_step, Update_Control_BB,
)
from grid_params import advection
from Plots import PlotFlow
from Costs import Calc_Cost_sPODG, Calc_Cost

from Helper import (
    ControlSelectionMatrix,
    compute_red_basis,
    calc_shift,
    L2norm_ROM,
    check_weak_divergence, L2inner_prod,
)
from Helper_sPODG import (
    subsample,
    get_T,
    central_FDMatrix,
    central_FD2Matrix,
    make_V_W_U_delta,
    make_V_W_delta_CubSpl, get_approx_state_sPODG,
)

from sPODG_solver import (
    IC_primal_sPODG_FRTO,
    IC_adjoint_sPODG_FRTO,
    mat_primal_sPODG_FRTO,
    TI_primal_sPODG_FRTO,
    TI_adjoint_sPODG_FRTO,
)

from sPOD_algo import give_interpolation_error


np.random.seed(0)


# ───────────────────────────────────────────────────────────────────────
def parse_arguments():
    p = argparse.ArgumentParser(description="Input the variables for running the script.")
    p.add_argument("type_of_problem", type=str, choices=["Shifting", "Constant_shift"],
                   help="Choose the problem type")
    p.add_argument("primal_adjoint_common_basis", type=literal_eval, choices=[True, False],
                   help="Include adjoint in basis computation? (True/False)")
    p.add_argument("grid", type=int, nargs=3, metavar=("Nx", "Nt", "cfl_fac"),
                   help="Enter the grid resolution and the cfl factor")
    p.add_argument("N_iter", type=int, help="Number of optimization iterations")
    p.add_argument("dir_prefix", type=str, choices=[".", "/work/burela"],
                   help="Directory prefix for I/O")
    p.add_argument("reg", type=float, nargs=2, metavar=("L1", "L2"),
                   help="L1 and L2 regularization weights (e.g. 0.01 0.001)")
    p.add_argument("CTC_mask_activate", type=literal_eval, choices=[True, False],
                   help="Include CTC mask in the system? (True/False)")
    p.add_argument("--modes", type=int, nargs=1,
                   help="Enter the modes e.g., --modes 3")
    p.add_argument("--tol", type=float, help="Tolerance level for fixed‐tol run")
    return p.parse_args()


def decide_run_type(args):
    if args.modes and args.tol is not None:
        print("Modes test takes precedence…")
        print(f"Modes provided: {args.modes}")
        TYPE, VAL = "modes", args.modes
        modes, tol, threshold = args.modes, None, False
    elif args.modes:
        print("Modes test…")
        print(f"Modes provided: {args.modes}")
        TYPE, VAL = "modes", args.modes
        modes, tol, threshold = args.modes, None, False
    elif args.tol is not None:
        print("Tolerance test…")
        print(f"Tolerance provided: {args.tol}")
        TYPE, VAL = "tol", args.tol
        modes, tol, threshold = (None, None), args.tol, True
    else:
        print("ERROR: Must specify either --modes or --tol.")
        sys.exit(1)
    return TYPE, VAL, modes, tol, threshold


def setup_advection(Nx, Nt, cfl_fac, type):
    if type == "Shifting":
        wf = advection(Lx=100, Nx=Nx, timesteps=Nt,
                       cfl=(8 / 6) / cfl_fac, tilt_from=3 * Nt // 4,
                       v_x=0.5, v_x_t=1.0,
                       variance=7, offset=12)
    elif type == "Constant_shift":
        wf = advection(Lx=80, Nx=Nx, timesteps=Nt,
                       cfl=0.0425 / cfl_fac, tilt_from=0,
                       v_x=8 / 3, v_x_t=8 / 3,
                       variance=7, offset=20)
    else:
        print("Please choose the correct problem type!!")
        exit()

    wf.Grid()
    return wf


def build_dirs(prefix, common_basis, reg_tuple, CTC_mask, TYPE, VAL):
    cb_str = "primal+adjoint_common_basis" if common_basis else "primal_basis"
    reg_str = f"L1={reg_tuple[0]}_L2={reg_tuple[1]}"
    data_dir = os.path.join(prefix, "data/sPODG_FRTO_adaptive", cb_str, reg_str, f"CTC_mask={CTC_mask}",
                            f"{TYPE}={VAL}")
    plot_dir = os.path.join(prefix, "plots/sPODG_FRTO_adaptive", cb_str, reg_str, f"CTC_mask={CTC_mask}",
                            f"{TYPE}={VAL}")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    return data_dir, plot_dir


def save_all(data_dir, **arrays):
    """
    Save all arrays/lists in `arrays` dict to data_dir
    """
    for name, arr in arrays.items():
        path = os.path.join(data_dir, f"{name}.npy")
        np.save(path, np.array(arr, dtype=object))
    print(f"→ Saved {len(arrays)} arrays.")


def write_checkpoint(data_dir, opt_step,
                     f, best_control, best_details,
                     J_opt_list, J_opt_FOM_list,
                     dL_du_norm_list, running_time,
                     trunc_modes_list, basis_update_idx_list):
    """
    Overwrite checkpoint files. All variables are written with fixed names,
    so each iteration replaces the previous checkpoint.
    """
    ckpt_dir = os.path.join(data_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Controls
    np.save(os.path.join(ckpt_dir, "f.npy"), f)
    np.save(os.path.join(ckpt_dir, "best_control.npy"), best_control)

    # Best-details
    np.save(os.path.join(ckpt_dir, "best_details.npy"),
            best_details, allow_pickle=True)

    # Histories
    np.save(os.path.join(ckpt_dir, "iter.npy"), opt_step)
    np.save(os.path.join(ckpt_dir, "J_opt_list.npy"), np.array(J_opt_list))
    np.save(os.path.join(ckpt_dir, "J_opt_FOM_list.npy"), np.array(J_opt_FOM_list))
    np.save(os.path.join(ckpt_dir, "dL_du_norm_list.npy"), np.array(dL_du_norm_list))
    np.save(os.path.join(ckpt_dir, "running_time.npy"), np.array(running_time))
    np.save(os.path.join(ckpt_dir, "trunc_modes.npy"), np.array(trunc_modes_list))
    np.save(os.path.join(ckpt_dir, "basis_update_idx_list.npy"), np.array(basis_update_idx_list))

    print(f"Checkpoint overwritten → {ckpt_dir}")


def C_matrix(Nx, CTC_end_index, apply_CTC_mask=False):
    C = np.ones(Nx)
    if apply_CTC_mask:
        C[:CTC_end_index] = 0
        return C == 1
    else:
        return C == 1


# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_arguments()

    print(f"Type of problem = {args.type_of_problem}")
    print("Type of basis computation: adaptive")
    print(f"Using adjoint in basis: {args.primal_adjoint_common_basis}")
    print(f"Using CTC mask pseudo hyperreduction: {args.CTC_mask_activate}")
    print(f"L1, L2 regularization = {tuple(args.reg)}")
    print(f"Grid = {tuple(args.grid)}")

    # Determine run type
    TYPE, VAL, modes, tol, threshold = decide_run_type(args)

    # Unpack regularization parameters
    L1_reg, L2_reg = args.reg
    Nx, Nt, cfl_fac = args.grid
    type_of_problem = args.type_of_problem

    # Set up WF and control matrix
    wf = setup_advection(Nx, Nt, cfl_fac, type_of_problem)
    if type_of_problem == "Shifting":
        if L1_reg != 0 and L2_reg == 0:  # Purely L1
            n_c_init = wf.Nx
            psi = ControlSelectionMatrix(wf, n_c_init, Gaussian=False, gaussian_mask_sigma=0.5)
            adjust = 1.0
        else:  # Mix type
            n_c_init = 40
            psi = ControlSelectionMatrix(wf, n_c_init, Gaussian=True, gaussian_mask_sigma=0.5)
            adjust = wf.dx
    elif type_of_problem == "Constant_shift":
        if L1_reg != 0 and L2_reg == 0:  # Purely L1
            n_c_init = wf.Nx
            psi = ControlSelectionMatrix(wf, n_c_init, Gaussian=False, gaussian_mask_sigma=0.5)
            adjust = 1.0
        else:  # Mix type
            n_c_init = 100
            psi = ControlSelectionMatrix(wf, n_c_init, Gaussian=False, gaussian_mask_sigma=0.5)
            adjust = wf.dx

    n_c = psi.shape[1]

    f = np.zeros((n_c, wf.Nt))  # initial control guess
    df = np.random.randn(*f.shape)
    psi = scipy.sparse.csc_matrix(psi)

    # Build coefficient matrices
    Mat = CoefficientMatrix(orderDerivative=wf.firstderivativeOrder,
                            Nxi=wf.Nx, Neta=1,
                            periodicity='Periodic',
                            dx=wf.dx, dy=0)
    A_p = - wf.v_x[0] * Mat.Grad_Xi_kron
    A_a = A_p.transpose()

    # Solve uncontrolled FOM once
    qs0 = IC_primal(wf.X, wf.Lx, wf.offset, wf.variance, type_of_problem=type_of_problem)
    qs_org = TI_primal(qs0, f, A_p, psi, wf.Nx, wf.Nt, wf.dt)
    qs_target = TI_primal_target(qs0, Mat.Grad_Xi_kron, wf.v_x_target, wf.Nx, wf.Nt,
                                 wf.dt, nu=0.1 if type_of_problem == "Constant_shift" else 0.0)
    q0 = np.ascontiguousarray(qs0)
    q0_adj = np.ascontiguousarray(IC_adjoint(wf.X))

    # Calculate the CTC matrix/array  (CTC and C are exactly the same)
    C = C_matrix(wf.Nx, wf.CTC_end_index, apply_CTC_mask=args.CTC_mask_activate)

    # Prepare directories
    data_dir, plot_dir = build_dirs(args.dir_prefix,
                                    args.primal_adjoint_common_basis,
                                    args.reg, args.CTC_mask_activate,
                                    TYPE, VAL)

    # Prepare kwargs
    kwargs = {
        'dx': wf.dx,
        'dt': wf.dt,
        'Nx': wf.Nx,
        'Nt': wf.Nt,
        'n_c': n_c,
        'lamda_l1': L1_reg,
        'lamda_l2': L2_reg,
        'delta_conv': 1e-4,  # Convergence criteria
        'delta': 1 / 2,  # Armijo constant
        'opt_iter': args.N_iter,  # Total iterations
        'shift_sample': (
            wf.Nx if type_of_problem == "Constant_shift"
            else 800 if type_of_problem == "Shifting"
            else wf.Nx
        ),
        'beta': 1 / 2,  # Beta factor for two-way backtracking line search
        'verbose': True,  # Print options
        'base_tol': tol,  # Base tolerance for selecting number of modes (main variable for truncation)
        'omega_cutoff': 1e-10,  # Below this cutoff the Armijo and Backtracking should exit the update loop
        'threshold': threshold,
        # Variable for selecting threshold based truncation or mode based. "TRUE" for threshold based
        # "FALSE" for mode based.
        'Nm_p': modes[0],  # Number of modes for truncation if threshold selected to False.
        'trafo_interp_order': 5,  # Order of the polynomial interpolation for the transformation operators
        'adjoint_scheme': "RK4",  # Time integration scheme for adjoint equation
        'common_basis': args.primal_adjoint_common_basis,  # True if primal + adjoint in basis else False
        'perform_grad_check': False,
        'offline_online_err_check': False
    }

    D = central_FDMatrix(order=6, Nx=wf.Nx, dx=wf.dx)
    D2 = central_FD2Matrix(order=6, Nx=wf.Nx, dx=wf.dx)
    delta_s = subsample(wf.X, num_sample=kwargs['shift_sample'])
    # Extract transformation operators based on sub-sampled delta
    T_delta, _ = get_T(delta_s, wf.X, wf.t, interp_order=kwargs['trafo_interp_order'])
    z = calc_shift(qs_org, q0, wf.X, wf.t)
    _, T = get_T(z, wf.X, wf.t, interp_order=kwargs['trafo_interp_order'])

    # Collector lists
    dL_du_norm_list = []
    J_opt_FOM_list = []
    J_opt_list = []
    running_time = []
    trunc_modes_list = []
    basis_update_idx_list = []
    best_control = np.zeros_like(f)
    best_details = {'J': np.inf, 'N_iter': None, 'Nm': None}
    f_last_valid = None

    start_total = time.time()
    t0 = perf_counter()

    omega_twbt = 1.0
    omega_bb = 1.0
    omega = 1.0
    stag = False
    stag_cntr = 0

    # ─────────────────────────────────────────────────────────────────────
    # Main “optimize‐step” loop wrapped in try/except/finally
    # ─────────────────────────────────────────────────────────────────────
    try:
        for opt_step in range(kwargs['opt_iter']):
            print(f"\n==============================")
            print(f"Optimization step: {opt_step}")

            if stag or (type_of_problem == "Constant_shift" and opt_step % 20 == 0) or (type_of_problem == "Shifting" and opt_step == 0):
                basis_update_idx_list.append(opt_step)

                # ───── Forward FOM: compute FOM state qs ─────
                # Compute FOM trajectories
                qs_full = TI_primal(q0, f, A_p, psi, wf.Nx, wf.Nt, wf.dt)
                qs_adj_full = TI_adjoint(q0_adj, qs_full, qs_target, None, A_a, None, C, wf.Nx, wf.dx, wf.Nt, wf.dt,
                                         scheme="RK4")

                # Compute shifts and (re)interpolate
                if type_of_problem == "Shifting":
                    z = calc_shift(qs_full, q0, wf.X, wf.t)
                    _, T = get_T(z, wf.X, wf.t, interp_order=kwargs['trafo_interp_order'])


                # Primal: reverse‐transform (and normalize if shared‐basis)
                if kwargs['common_basis']:
                    qs_norm = qs_full / np.linalg.norm(qs_full)
                    qs_adj_norm = qs_adj_full / np.linalg.norm(qs_adj_full)
                    qs_norm_s = T.reverse(qs_norm)
                    qs_adj_norm_s = T.reverse(qs_adj_norm)
                    snap_cat_p_s = np.concatenate([qs_norm_s, qs_adj_norm_s], axis=1)
                else:
                    snap_cat_p_s = T.reverse(qs_full).copy()

                # Compute reduced bases
                V, qs_sPOD = compute_red_basis(snap_cat_p_s, equation="primal", **kwargs)
                Nm = V.shape[1]
                err = np.linalg.norm(snap_cat_p_s - qs_sPOD) / np.linalg.norm(snap_cat_p_s)
                print(f"Primal basis: Nm_p={Nm}, err={err:.3e}")

                # Initial condition for dynamical simulation
                a_p = IC_primal_sPODG_FRTO(q0, V)
                a_a = IC_adjoint_sPODG_FRTO(Nm)
                trunc_modes_list.append(Nm)

                # Construct the primal system matrices for the sPOD-Galerkin approach
                Vd_p, Wd_p, Ud_p = make_V_W_U_delta(V, T_delta, D, D2, kwargs['shift_sample'], kwargs['Nx'], Nm)

                # Construct the primal and adjoint system matrices for the sPOD-Galerkin approach
                lhs_p, rhs_p, c_p, tar_a = mat_primal_sPODG_FRTO(Vd_p, Wd_p, Ud_p, C, A_p, psi,
                                                                 samples=kwargs['shift_sample'],
                                                                 modes=Nm)

            # ───── Forward ROM: compute ROM state a_p → as_p ─────
            as_p, as_dot, intIds, weights = TI_primal_sPODG_FRTO(lhs_p, rhs_p, c_p, a_p, f, delta_s, modes=Nm,
                                                                 Nt=kwargs['Nt'],
                                                                 dt=kwargs['dt'])

            # ───── Compute costs ─────
            J_s, J_ns, _ = Calc_Cost_sPODG(Vd_p, as_p[:-1], qs_target, f, C, intIds, weights,
                                           kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
            J_ROM = J_s + J_ns

            qs_opt_full = TI_primal(q0, f, A_p, psi, wf.Nx, wf.Nt, wf.dt)
            JJ_s, JJ_ns = Calc_Cost(qs_opt_full, qs_target, f, C, kwargs['dx'], kwargs['dt'],
                                    kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
            J_FOM = JJ_s + JJ_ns

            J_opt_list.append(J_ROM)
            J_opt_FOM_list.append(J_FOM)

            # Track best control
            if J_FOM < best_details['J']:
                best_details.update({'J': J_FOM, 'N_iter': opt_step, 'Nm': Nm})
                best_control = f.copy()

            # ───── Backward ROM (adjoint) ─────
            as_adj = TI_adjoint_sPODG_FRTO(a_a, f, as_p, qs_target, as_dot, lhs_p, rhs_p, c_p, tar_a, C, Vd_p, Wd_p,
                                           Nm, delta_s, Nt=kwargs['Nt'], dt=kwargs['dt'], dx=kwargs['dx'],
                                           scheme=kwargs['adjoint_scheme'])

            # ───── Compute the smooth gradient + the generalized gradient mapping ─────
            dL_du_s = Calc_Grad_sPODG_FRTO_smooth(f, c_p, as_adj, as_p, intIds, weights, kwargs['lamda_l2'])
            dL_du_g = Calc_Grad_mapping(f, dL_du_s, omega, kwargs['lamda_l1'])
            dL_du_norm = np.sqrt(L2norm_ROM(dL_du_g, kwargs['dt']))

            dL_du_norm_list.append(dL_du_norm)

            # ───── Gradient check with Finite differences ─────
            if kwargs['perform_grad_check']:
                print("-------------GRAD CHECK-----------------")
                eps = 1e-5
                f_rand = f + eps * df
                qs_rand = TI_primal(q0, f_rand, A_p, psi, wf.Nx, wf.Nt, wf.dt)
                J_s_eps, J_ns_eps = Calc_Cost(qs_rand, qs_target, f_rand, C, kwargs['dx'], kwargs['dt'],
                                              kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
                J_FOM_eps = J_s_eps + J_ns_eps
                print("Finite difference gradient", (J_FOM_eps - J_FOM) / eps)
                print("Analytic gradient", L2inner_prod(dL_du_g, df, kwargs['dt']))

            # ───── Offline/Online error check ─────
            if kwargs['offline_online_err_check']:
                print("-------------OFF/ON ERROR CHECK-----------------")
                print("The discrepancy between the offline error and the online error starts growing with increasing "
                      "optimization steps here because of the stale basis used to reconstruct the online solution, "
                      "whereas the offline error is computed from the FOM primal and adjoint at the current control")
                qs_opt_full_shifted = T.reverse(qs_opt_full)
                _, qs_sPOD_offline = compute_red_basis(qs_opt_full_shifted, equation="primal", **kwargs)
                err_offline = np.linalg.norm(qs_opt_full_shifted - qs_sPOD_offline) / np.linalg.norm(
                    qs_opt_full_shifted)
                print(f"Primal offline error: err={err_offline:.3e}")
                qs_sPOD_online = get_approx_state_sPODG(Vd_p, f, as_p[:-1, :], intIds, weights, kwargs['Nx'],
                                                        kwargs['Nt'])
                err_online = np.linalg.norm(qs_opt_full - qs_sPOD_online) / np.linalg.norm(qs_opt_full)
                print(f"Primal online error: err={err_online:.3e}")

            # ───── Step‐size: BB vs. TWBT, including Armijo‐stagnation logic ─────
            ratio = dL_du_norm / dL_du_norm_list[0]
            if ratio < 5e-3:
                print(f"BB acting.....")
                omega_bb = get_BB_step(fOld, f, dL_du_Old, dL_du_s, opt_step, **kwargs)
                if omega_bb < 0:
                    print("WARNING: BB gave negative step size thus resorting to using TWBT")
                    fNew, omega_twbt, stag = Update_Control_sPODG_FRTO_TWBT(f, lhs_p, rhs_p, c_p, Vd_p,
                                                                            a_p, qs_target, delta_s, J_s,
                                                                            omega_twbt, Nm, dL_du_s, C, adjust,
                                                                            **kwargs)
                    omega = omega_twbt
                else:
                    fNew = Update_Control_BB(f, dL_du_s, omega_bb, kwargs['lamda_l1'])
                    stag = False
                    omega = omega_bb
            else:
                print("TWBT acting…")
                fNew, omega_twbt, stag = Update_Control_sPODG_FRTO_TWBT(f, lhs_p, rhs_p, c_p, Vd_p,
                                                                        a_p, qs_target, delta_s, J_s,
                                                                        omega_twbt, Nm, dL_du_s, C, adjust,
                                                                        **kwargs)
                omega = omega_twbt

            t1 = perf_counter()
            running_time.append(t1 - t0)

            # Saving previous controls for Barzilai Borwein step
            fOld = f.copy()
            f = fNew.copy()
            dL_du_Old = dL_du_s.copy()

            print(
                f"J_ROM: {J_ROM:.3e}, J_FOM: {J_FOM:.3e}, ||dL_du|| = {dL_du_norm:.3e}, ||dL_du||_{opt_step} / "
                f"||dL_du||_0 = {ratio:.3e}"
            )

            # Convergence Criteria
            if opt_step == kwargs['opt_iter'] - 1:
                print("\n\n-------------------------------")
                print(
                    f"WARNING... maximal number of steps reached, "
                    f"J_ROM: {J_ROM:.3e}, J_FOM: {J_FOM:.3e}, ||dL_du|| = {dL_du_norm:.3e}, ||dL_du||_{opt_step} / "
                    f"||dL_du||_0 = {ratio:.3e}"
                )
                f_last_valid = fNew.copy()
                qs_opt_full = TI_primal(q0, f_last_valid, A_p, psi, wf.Nx, wf.Nt, wf.dt)
                JJ_s, JJ_ns = Calc_Cost(qs_opt_full, qs_target, f_last_valid, C, kwargs['dx'], kwargs['dt'],
                                        kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
                J_FOM = JJ_s + JJ_ns
                if J_FOM < best_details['J']:
                    best_details.update({'J': J_FOM})
                    best_control = f_last_valid.copy()
                break
            elif ratio < kwargs['delta_conv']:
                print("\n\n-------------------------------")
                print(
                    f"Optimization converged with, "
                    f"J_ROM: {J_ROM:.3e}, J_FOM: {J_FOM:.3e}, ||dL_du|| = {dL_du_norm:.3e}, ||dL_du||_{opt_step} / "
                    f"||dL_du||_0 = {ratio:.3e}"
                )
                f_last_valid = fNew.copy()
                qs_opt_full = TI_primal(q0, f_last_valid, A_p, psi, wf.Nx, wf.Nt, wf.dt)
                JJ_s, JJ_ns = Calc_Cost(qs_opt_full, qs_target, f_last_valid, C, kwargs['dx'], kwargs['dt'],
                                        kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
                J_FOM = JJ_s + JJ_ns
                if J_FOM < best_details['J']:
                    best_details.update({'J': J_FOM})
                    best_control = f_last_valid.copy()
                break
            else:
                if opt_step == 0:
                    if stag:
                        print("\n\n-------------------------------")
                        print(
                            f"Armijo Stagnated !!!!!! due to the step length being too low thus refining the basis at itr: {opt_step + 1} with "
                            f"J_ROM: {J_ROM:.3e}, J_FOM: {J_FOM:.3e}, ||dL_du|| = {dL_du_norm:.3e}, ||dL_du||_{opt_step} /"
                            f"||dL_du||_0 = {ratio:.3e}")
                        f = best_control.copy()
                    else:
                        stag_cntr = 0
                else:
                    dJ = (J_opt_list[-1] - J_opt_list[-2]) / J_opt_list[0]
                    if abs(dJ) == 0:
                        print(f"WARNING: dJ ~ 0 → stopping"
                              f"J_ROM: {J_ROM:.3e}, J_FOM: {J_FOM:.3e}, ||dL_du|| = {dL_du_norm:.3e}, ||dL_du||_{opt_step} /"
                              f"||dL_du||_0 = {ratio:.3e}"
                              )
                        f_last_valid = fOld.copy()
                        break
                    if stag:
                        stag_cntr = stag_cntr + 1
                        if stag_cntr >= 2:
                            print("\n-------------------------------")
                            print(
                                f"TWBT Armijo Stagnated !!!!!! even after 2 consecutive basis updates thus exiting at itr: {opt_step} with "
                                f"J_ROM: {J_ROM:.3e}, J_FOM: {J_FOM:.3e}, ||dL_du|| = {dL_du_norm:.3e}, ||dL_du||_{opt_step} /"
                                f"||dL_du||_0 = {ratio:.3e}")
                            f_last_valid = fOld.copy()
                            break
                        print("\n-------------------------------")
                        print(
                            f"Armijo Stagnated !!!!!! due to the step length being too low thus updating the basis at itr: {opt_step + 1}")
                        f = best_control.copy()
                    else:
                        stag_cntr = 0
                    # Convergence criteria for BB
                    if J_FOM > 1e6 or abs(omega_bb) < kwargs['omega_cutoff']:
                        print("\n\n-------------------------------")
                        print(
                            f"Barzilai Borwein acceleration failed!!!!!! J_FOM increased to unrealistic values or the "
                            f"omega went below cutoff, thus exiting"
                            f"at itr: {opt_step} with "
                            f"J_ROM: {J_ROM:.3e}, J_FOM: {J_FOM:.3e}, ||dL_du|| = {dL_du_norm:.3e}, ||dL_du||_{opt_step} /"
                            f"||dL_du||_0 = {ratio:.3e}"
                        )
                        f_last_valid = fOld.copy()
                        break

            if opt_step % 100 == 0:
                write_checkpoint(
                    data_dir,
                    opt_step=opt_step,
                    f=f,
                    best_control=best_control,
                    best_details=best_details,
                    J_opt_list=J_opt_list,
                    J_opt_FOM_list=J_opt_FOM_list,
                    dL_du_norm_list=dL_du_norm_list,
                    running_time=running_time,
                    trunc_modes_list=trunc_modes_list,
                    basis_update_idx_list=basis_update_idx_list
                )

                # Call the helper to check for weak divergence
                diverging, avg_prev, avg_last = check_weak_divergence(J_opt_FOM_list, window=100, margin=0.0)
                if diverging:
                    print(
                        "\n*** ROM is no longer accurate thus exiting !!!!: "
                        f"avg(J_FOM[-100:]) = {avg_last:.3e}  > "
                        f"avg(J_FOM[-200:-100]) = {avg_prev:.3e} → exiting ***"
                    )

                    # store last valid control and possibly update best_details
                    f_last_valid = f.copy()
                    qs_cand = TI_primal(q0, f_last_valid, A_p, psi, wf.Nx, wf.Nt, wf.dt)
                    JJ_s_cand, JJ_ns_cand = Calc_Cost(
                        qs_cand, qs_target, f_last_valid, C,
                        kwargs['dx'], kwargs['dt'],
                        kwargs['lamda_l1'], kwargs['lamda_l2'], adjust
                    )
                    J_FOM_cand = JJ_s_cand + JJ_ns_cand
                    if J_FOM_cand < best_details['J']:
                        best_details['J'] = J_FOM_cand
                        best_control = f_last_valid.copy()
                    break

    except Exception as e:

        print("\n*** EXCEPTION: ***")
        # Print the full traceback to stderr (includes file and line number)
        traceback.print_exc()

        print("\nSaving data up to crash…")
        to_save = {
            "J_opt_list_at_crash": J_opt_list,
            "J_opt_FOM_list_at_crash": J_opt_FOM_list,
            "running_time_at_crash": running_time,
            "dL_du_norm_list_at_crash": dL_du_norm_list,
            "trunc_modes_list_at_crash": trunc_modes_list,
            "basis_update_idx_list_at_crash": basis_update_idx_list
        }
        if f_last_valid is not None:
            to_save["last_valid_control_at_crash"] = f_last_valid
        to_save["best_control_at_crash"] = best_control
        to_save["best_details_at_crash"] = best_details

        save_all(data_dir, **to_save)
        sys.exit(1)

    finally:
        print("\nFinal save…")
        to_save_final = {"J_opt_list_final": J_opt_list, "J_opt_FOM_list_final": J_opt_FOM_list,
                         "running_time_final": running_time, "dL_du_norm_list_final": dL_du_norm_list,
                         "trunc_modes_list_final": trunc_modes_list,
                         "basis_update_idx_list_final": basis_update_idx_list,
                         "best_control_final": best_control, "best_details_final": best_details,
                         "last_valid_control_final": f_last_valid}

        save_all(data_dir, **to_save_final)

    # ─────────────────────────────────────────────────────────────────────
    # Compute best control based cost
    qs_opt_full = TI_primal(q0, best_control, A_p, psi, wf.Nx, wf.Nt, wf.dt)
    qs_adj_opt = TI_adjoint(q0_adj, qs_opt_full, qs_target, None, A_a, None, C, wf.Nx, wf.dx, wf.Nt, wf.dt, scheme="RK4")
    f_opt = psi @ best_control
    J_s_f, J_ns_f = Calc_Cost(qs_opt_full, qs_target, best_control, C, kwargs['dx'], kwargs['dt'],
                              kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
    J_final = J_s_f + J_ns_f

    # Compute last valid control based cost
    qs_opt_full__ = TI_primal(q0, f_last_valid, A_p, psi, wf.Nx, wf.Nt, wf.dt)
    qs_adj_opt__ = TI_adjoint(q0_adj, qs_opt_full__, qs_target, None, A_a, None, C, wf.Nx, wf.dx, wf.Nt, wf.dt,
                              scheme="RK4")
    f_opt__ = psi @ f_last_valid
    J_s_f__, J_ns_f__ = Calc_Cost(qs_opt_full__, qs_target, f_last_valid, C, kwargs['dx'], kwargs['dt'],
                                  kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
    J_final__ = J_s_f__ + J_ns_f__

    print(f"\nBest FOM cost: {J_final:.3e}")
    print(f"\nLast valid FOM cost: {J_final__:.3e}")
    print(f"\nTotal elapsed time: {time.time() - start_total:.3f}s")

    # # Save convergence lists and final states:
    # np.save(os.path.join(data_dir, "qs_opt_final.npy"), qs_opt_full)
    # np.save(os.path.join(data_dir, "qs_adj_opt_final.npy"), qs_adj_opt)

    # Plot results
    pf = PlotFlow(wf.X, wf.t)
    pf.plot1D(qs_opt_full, name="qs_opt", immpath=plot_dir)
    pf.plot1D(qs_adj_opt, name="qs_adj_opt", immpath=plot_dir)
    pf.plot1D(f_opt, name="f_opt", immpath=plot_dir)
    pf.plot1D_ROM_converg(J_opt_list, J_opt_FOM_list, name="J", immpath=plot_dir)
