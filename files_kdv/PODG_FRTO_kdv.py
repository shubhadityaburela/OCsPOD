"""
This file is the version with ROM adjoint. It can handle both the scenarios.
1. Fixed tolerance
2. Fixed modes
"""
import os
import sys
import time
import argparse
import traceback

import matplotlib.pyplot as plt
import numpy as np
import scipy
from time import perf_counter
from ast import literal_eval
import faulthandler

from Coefficient_Matrix import CoefficientMatrix
from Costs import Calc_Cost_PODG, Calc_Cost
from FOM_solver import IC_primal_kdv, TI_primal_kdv_impl, IC_adjoint_kdv, \
    TI_adjoint_kdv_impl
from Grads import Calc_Grad_PODG_smooth, Calc_Grad_mapping
from Helper import compute_red_basis, L2norm_ROM, check_weak_divergence, \
    ControlSelectionMatrix_kdvb, compute_deim_basis, L2inner_prod
from PODG_solver import (
    IC_primal_PODG_FRTO_kdv, IC_adjoint_PODG_FRTO_kdv, mat_primal_PODG_FRTO_kdv, mat_adjoint_PODG_FRTO_kdv,
    TI_primal_PODG_FRTO_kdv_expl, TI_adjoint_PODG_FRTO_kdv_impl, TI_adjoint_PODG_FRTO_kdv_expl
)
from Update import get_BB_step, \
    Update_Control_PODG_FRTO_TWBT_kdv, Update_Control_BB_kdv
from grid_params import Korteweg_de_Vries
from Plots import PlotFlow


np.random.seed(0)


# ───────────────────────────────────────────────────────────────────────
def parse_arguments():
    p = argparse.ArgumentParser(description="Input the variables for running the script.")
    p.add_argument("fully_nonlinear", type=literal_eval, choices=[True, False],
                   help="Select True for Fully nonlinear else False? (True/False)")
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


def setup_kdv(Nx, Nt, cfl_fac):
    kdv = Korteweg_de_Vries(Nx=Nx, timesteps=Nt, cfl=0.17 / cfl_fac, v_x=8 / 3, offset=20)
    kdv.Grid()
    return kdv


def build_dirs(prefix, fully_nonlinear, common_basis, reg_tuple, CTC_mask, TYPE, VAL):
    cb_str = "primal+adjoint_common_basis" if common_basis else "primal_basis"
    reg_str = f"L1={reg_tuple[0]}_L2={reg_tuple[1]}"
    fnl_str = f"fully_nonlinear={fully_nonlinear}"
    data_dir = os.path.join(prefix, "data/PODG_FRTO_kdv", fnl_str, cb_str, reg_str, f"CTC_mask={CTC_mask}",
                            f"{TYPE}={VAL}")
    plot_dir = os.path.join(prefix, "plots/PODG_FRTO_kdv", fnl_str, cb_str, reg_str, f"CTC_mask={CTC_mask}",
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
                     trunc_modes_list, trunc_deim_modes_list):
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
    np.save(os.path.join(ckpt_dir, "trunc_deim_modes.npy"), np.array(trunc_deim_modes_list))

    print(f"Checkpoint overwritten → {ckpt_dir}")


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
    print(f"Problem type (full nonlinearity)= {args.fully_nonlinear}")
    print(f"Using adjoint in basis: {args.primal_adjoint_common_basis}")
    print(f"L1, L2 regularization = {tuple(args.reg)}")
    print(f"Using CTC mask pseudo hyperreduction: {args.CTC_mask_activate}")
    print(f"Grid = {tuple(args.grid)}")

    # Determine run type
    TYPE, VAL, modes, tol, threshold = decide_run_type(args)

    # Unpack regularization parameters
    L1_reg, L2_reg = args.reg
    Nx, Nt, cfl_fac = args.grid

    # Set up kdvb and control matrix
    kdv = setup_kdv(Nx, Nt, cfl_fac)

    # Set up kdvb and control matrix
    if L1_reg != 0 and L2_reg == 0:  # Purely L1
        n_c_init = kdv.Nx
        psi = ControlSelectionMatrix_kdvb(kdv, n_c_init, Gaussian=False, gaussian_mask_sigma=0.5)
        adjust = 1.0
    else:  # Mix type
        n_c_init = 100
        psi = ControlSelectionMatrix_kdvb(kdv, n_c_init, Gaussian=False, gaussian_mask_sigma=0.5)
        adjust = kdv.dx
    n_c = psi.shape[1]

    # Prepare kwargs
    kwargs = {
        'dx': kdv.dx,
        'dt': kdv.dt,
        'Nx': kdv.Nx,
        'Nt': kdv.Nt,
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
        'common_basis': args.primal_adjoint_common_basis,
        'perform_grad_check': False,
        'offline_online_err_check': False
    }
    f = np.zeros((n_c, kdv.Nt))  # initial control guess
    df = np.random.randn(*f.shape)
    psi = scipy.sparse.csc_matrix(psi)

    # --------------------------------------------------------------------------------- #
    # BUILDING THE FOM DATA FIRST
    # Build coefficient matrices
    Mat = CoefficientMatrix(orderDerivative=kdv.firstderivativeOrder,
                            Nxi=kdv.Nx, Neta=1,
                            periodicity='Periodic',
                            dx=kdv.dx, dy=0)
    D1 = Mat.Grad_Xi_kron
    D2 = Mat.Grad_Xi_kron @ D1
    D3 = Mat.Grad_Xi_kron @ D2

    # Calculate the CTC matrix/array  (CTC and C are exactly the same)
    C = C_matrix(kdv.Nx, None, apply_CTC_mask=args.CTC_mask_activate)
    CTC = C.copy()

    common_params = dict(D1=D1, D2=D2, D3=D3)
    shared_dynamics = {**common_params, 'B': psi}

    if args.fully_nonlinear:
        # Nonlinear
        target_params = {'c': kdv.v_x[0], 'alpha': 0.0, 'omega': 1.4, 'gamma': 1.4, 'nu': 0.06}
        shared_params = {'c': kdv.v_x[0], 'alpha': 0.0, 'omega': 1.0, 'gamma': 1.0, 'nu': 0.0}
    else:
        # Nearly linear
        target_params = {'c': kdv.v_x[0], 'alpha': 1.0, 'omega': 0.0, 'gamma': 0.0, 'nu': 0.1}
        shared_params = {'c': kdv.v_x[0], 'alpha': 1.0, 'omega': 0.0, 'gamma': 0.0, 'nu': 0.0}

    # Nonlinearity less prevalent
    params_primal = {**shared_dynamics, **shared_params}
    params_target = {**shared_dynamics, **target_params}
    params_adjoint = {**common_params, 'CTC': CTC, **shared_params}

    J_l = scipy.sparse.identity(kdv.Nx, format='csc') - 0.5 * kdv.dt * (
            params_primal['alpha'] * (- params_primal['c']) * D1
            - params_primal['gamma'] * D3 + params_primal['nu'] * D2)
    J_l_adjoint = scipy.sparse.identity(kdv.Nx, format='csc') + 0.5 * kdv.dt * (
            params_adjoint['alpha'] * params_adjoint['c'] * D1.T
            + params_adjoint['gamma'] * D3.T - params_adjoint['nu'] * D2.T)
    J_l_target = scipy.sparse.identity(kdv.Nx, format='csc') - 0.5 * kdv.dt * (
            params_target['alpha'] * (- params_target['c']) * D1
            - params_target['gamma'] * D3 + params_target['nu'] * D2)

    # Solve uncontrolled FOM once
    qs0 = IC_primal_kdv(kdv.X, kdv.Lx, kdv.c, kdv.offset)
    qs_org = TI_primal_kdv_impl(qs0, f, J_l, kdv.Nx, kdv.Nt, kdv.dt, **params_primal)
    qs_target = TI_primal_kdv_impl(qs0, np.zeros_like(f), J_l_target, kdv.Nx, kdv.Nt, kdv.dt, **params_target)
    q0 = np.ascontiguousarray(qs0)
    q0_adj = np.ascontiguousarray(IC_adjoint_kdv(kdv.X))

    # Prepare directories
    data_dir, plot_dir = build_dirs(args.dir_prefix,
                                    args.fully_nonlinear,
                                    args.primal_adjoint_common_basis,
                                    args.reg, args.CTC_mask_activate,
                                    TYPE, VAL)

    # --------------------------------------------------------------------------------- #
    # Precompute full basis once (fixed‐basis approach)
    qs_full = qs_org.copy()
    qs_adj_full = TI_adjoint_kdv_impl(q0_adj, qs_full, qs_target, J_l_adjoint, kdv.Nx, kdv.Nt, kdv.dx, kdv.dt,
                                      **params_adjoint)

    # Concatenate if common_basis, else keep separately
    qs_norm = qs_full / np.linalg.norm(qs_full)
    qs_adj_norm = qs_adj_full / np.linalg.norm(qs_adj_full)
    if kwargs['common_basis']:
        snap_cat_p = np.concatenate([qs_norm, qs_adj_norm], axis=1)
    else:
        snap_cat_p = qs_full.copy()

    # Compute reduced bases
    V, qsPOD = compute_red_basis(snap_cat_p, equation="primal", **kwargs)
    Nm = V.shape[1]
    err = np.linalg.norm(snap_cat_p - qsPOD) / np.linalg.norm(snap_cat_p)
    print(f"Primal basis: Nm_p={Nm}, err={err:.3e}")

    # Compute the DEIM basis
    qs_full_deim = D1 @ (qs_full ** 2) + qs_full * (D1 @ qs_full)  # DEIM snapshot matrix
    V_deim, qsPOD_deim = compute_deim_basis(qs_full_deim, equation="primal", **kwargs)
    Nm_deim = V_deim.shape[1]
    err = np.linalg.norm(qs_full_deim - qsPOD_deim) / np.linalg.norm(qs_full_deim)
    print(f"DEIM basis: Nm_deim={Nm_deim}, err={err:.3e}")

    # Initial conditions for ROM
    a_p = IC_primal_PODG_FRTO_kdv(V, q0)
    a_a = IC_adjoint_PODG_FRTO_kdv(Nm)

    # Build ROM system matrices
    primal_mat = mat_primal_PODG_FRTO_kdv(V, V_deim, **params_primal)
    adjoint_mat = mat_adjoint_PODG_FRTO_kdv(V, CTC, qs_target)

    # Prepare the linear parts of the Jacobian for the reduced order models
    J_l_ROM_primal = np.eye(Nm) - 0.5 * kdv.dt * (
            - params_primal['alpha'] * params_primal['c'] * primal_mat.D_1r
            - params_primal['gamma'] * primal_mat.D_3r + params_primal['nu'] * primal_mat.D_2r)
    J_l_ROM_adjoint = np.eye(Nm) + 0.5 * kdv.dt * (
            params_primal['alpha'] * params_primal['c'] * primal_mat.D_1r.T
            + params_primal['gamma'] * primal_mat.D_3r.T - params_primal['nu'] * primal_mat.D_2r.T)

    # Collector lists
    dL_du_norm_list = []
    J_opt_FOM_list = []
    J_opt_list = []
    running_time = []
    trunc_modes_list = [Nm]  # fixed basis, just one value
    trunc_deim_modes_list = [Nm_deim]  # fixed basis, just one value
    best_control = np.zeros_like(f)
    best_details = {'J': np.inf, 'N_iter': None, 'Nm': Nm, 'Nm_deim': Nm_deim}
    f_last_valid = None

    start_total = time.time()
    t0 = perf_counter()

    omega_twbt = 1
    omega_bb = 1
    omega = 1
    stag = False
    stag_cntr = 0

    # ─────────────────────────────────────────────────────────────────────
    # Main “optimize‐step” loop wrapped in try/except/finally
    # ─────────────────────────────────────────────────────────────────────
    try:
        for opt_step in range(kwargs['opt_iter']):
            print(f"\n==============================")
            print(f"Optimization step: {opt_step}")

            # ───── Forward ROM: compute ROM state a_p → as_p ─────
            as_p = TI_primal_PODG_FRTO_kdv_expl(a_p, f, primal_mat.D_1r, primal_mat.D_2r, primal_mat.D_3r,
                                                primal_mat.prefactor, primal_mat.ST_V, primal_mat.ST_D1V,
                                                primal_mat.B_r, params_primal['c'], params_primal['alpha'],
                                                params_primal['omega'], params_primal['gamma'],
                                                params_primal['nu'], kwargs['Nt'], kwargs['dt'])

            # ───── Compute costs ─────
            J_s, J_ns = Calc_Cost_PODG(V, as_p, qs_target, f, C, kwargs['dx'], kwargs['dt'],
                                       kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
            J_ROM = J_s + J_ns

            qs_opt_full = TI_primal_kdv_impl(q0, f, J_l, kdv.Nx, kdv.Nt, kdv.dt, **params_primal)
            JJ_s, JJ_ns = Calc_Cost(qs_opt_full, qs_target, f, C, kwargs['dx'], kwargs['dt'],
                                    kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
            J_FOM = JJ_s + JJ_ns

            J_opt_list.append(J_ROM)
            J_opt_FOM_list.append(J_FOM)

            # Track best control
            if J_FOM < best_details['J']:
                best_details.update({'J': J_FOM, 'N_iter': opt_step, 'Nm': Nm, 'Nm_deim': Nm_deim})
                best_control = f.copy()

            # ───── Backward ROM (adjoint) ─────
            as_adj = TI_adjoint_PODG_FRTO_kdv_impl(a_a, as_p, adjoint_mat, primal_mat,
                                                   J_l_ROM_adjoint, kwargs['Nx'], kwargs['Nt'],
                                                   kwargs['dx'], kwargs['dt'], **params_primal)

            # ───── Compute the smooth gradient + the generalized gradient mapping ─────
            dL_du_s = Calc_Grad_PODG_smooth(primal_mat.B_r, f, as_adj, kwargs['lamda_l2'])
            dL_du_g = Calc_Grad_mapping(f, dL_du_s, omega, kwargs['lamda_l1'])
            dL_du_norm = np.sqrt(L2norm_ROM(dL_du_g, kwargs['dt']))

            dL_du_norm_list.append(dL_du_norm)

            # ───── Gradient check with Finite differences ─────
            if kwargs['perform_grad_check']:
                print("-------------GRAD CHECK-----------------")
                eps = 1e-5
                f_rand = f + eps * df
                qs_rand = TI_primal_kdv_impl(q0, f_rand, J_l, kdv.Nx, kdv.Nt, kdv.dt, **params_primal)
                JJ_s_eps, JJ_ns_eps = Calc_Cost(qs_rand, qs_target, f_rand, C, kwargs['dx'], kwargs['dt'],
                                                kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
                J_FOM_eps = JJ_s_eps + JJ_ns_eps
                print("Finite difference gradient", (J_FOM_eps - J_FOM) / eps)
                print("Analytic gradient", L2inner_prod(dL_du_g, df, kwargs['dt']))

            # ───── Offline/Online error check ─────
            if kwargs['offline_online_err_check']:
                print("-------------OFF/ON ERROR CHECK-----------------")
                print("The discrepancy between the offline error and the online error starts growing with increasing "
                      "optimization steps here because of the stale basis used to reconstruct the online solution, "
                      "whereas the offline error is computed from the FOM primal and adjoint at the current control")
                _, qs_POD_offline = compute_red_basis(qs_opt_full, equation="primal", **kwargs)
                err_offline = np.linalg.norm(qs_opt_full - qs_POD_offline) / np.linalg.norm(qs_opt_full)
                print(f"Primal offline error: err={err_offline:.3e}")
                qs_POD_online = V @ as_p
                err_online = np.linalg.norm(qs_opt_full - qs_POD_online) / np.linalg.norm(qs_opt_full)
                print(f"Primal online error: err={err_online:.3e}")

            # ───── Step‐size: BB vs. TWBT, including Armijo‐stagnation logic ─────
            ratio = dL_du_norm / dL_du_norm_list[0]
            if ratio < 5e-3:
                print(f"BB acting.....")
                omega_bb = get_BB_step(fOld, f, dL_du_Old, dL_du_s, opt_step, **kwargs)
                if omega_bb < 0:
                    print("WARNING: BB gave negative step size thus resorting to using TWBT")
                    fNew, omega_twbt, stag = Update_Control_PODG_FRTO_TWBT_kdv(f, primal_mat, J_l_ROM_primal, V, a_p,
                                                                               qs_target, J_s, omega_twbt, dL_du_s, C,
                                                                               adjust, params_primal,
                                                                               **kwargs)
                    omega = omega_twbt
                else:
                    fNew = Update_Control_BB_kdv(f, dL_du_s, omega_bb, kwargs['lamda_l1'])
                    stag = False
                    omega = omega_bb
            else:
                print("TWBT acting…")
                fNew, omega_twbt, stag = Update_Control_PODG_FRTO_TWBT_kdv(f, primal_mat, J_l_ROM_primal, V, a_p,
                                                                           qs_target, J_s, omega_twbt, dL_du_s, C,
                                                                           adjust, params_primal,
                                                                           **kwargs)
                omega = omega_twbt

            t1 = perf_counter()
            running_time.append(t1 - t0)
            t0 = t1

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
                qs_opt_full = TI_primal_kdv_impl(q0, f_last_valid, J_l, kdv.Nx, kdv.Nt, kdv.dt, **params_primal)
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
                qs_opt_full = TI_primal_kdv_impl(q0, f_last_valid, J_l, kdv.Nx, kdv.Nt, kdv.dt, **params_primal)
                JJ_s, JJ_ns = Calc_Cost(qs_opt_full, qs_target, f_last_valid, C, kwargs['dx'], kwargs['dt'],
                                        kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
                J_FOM = JJ_s + JJ_ns
                if J_FOM < best_details['J']:
                    best_details.update({'J': J_FOM})
                    best_control = f_last_valid.copy()
                break
            else:
                f_last_valid = fOld.copy()
                if opt_step == 0:
                    if stag:
                        print("\n\n-------------------------------")
                        print(
                            f"Armijo Stagnated !!!!!! due to the step length being too low thus exiting at itr: {opt_step} with "
                            f"J_ROM: {J_ROM:.3e}, J_FOM: {J_FOM:.3e}, ||dL_du|| = {dL_du_norm:.3e}, ||dL_du||_{opt_step} /"
                            f"||dL_du||_0 = {ratio:.3e}")
                        break
                else:
                    dJ = (J_opt_list[-1] - J_opt_list[-2]) / J_opt_list[0]
                    if abs(dJ) == 0:
                        print(f"WARNING: dJ ~ 0 → stopping"
                              f"J_ROM: {J_ROM:.3e}, J_FOM: {J_FOM:.3e}, ||dL_du|| = {dL_du_norm:.3e}, ||dL_du||_{opt_step} /"
                              f"||dL_du||_0 = {ratio:.3e}"
                              )
                        break
                    if stag:
                        print("\n-------------------------------")
                        print(
                            f"TWBT Armijo Stagnated !!!!!! due to the step length being too low thus exiting at itr: {opt_step} with "
                            f"J_ROM: {J_ROM:.3e}, J_FOM: {J_FOM:.3e}, ||dL_du|| = {dL_du_norm:.3e}, ||dL_du||_{opt_step} /"
                            f"||dL_du||_0 = {ratio:.3e}"
                        )
                        break
                    if J_FOM > 1e10 or abs(omega_bb) < kwargs['omega_cutoff']:
                        print("\n\n-------------------------------")
                        print(
                            f"Barzilai Borwein acceleration failed!!!!!! J_FOM increased to unrealistic values or the "
                            f"omega went below cutoff, thus exiting"
                            f"at itr: {opt_step} with "
                            f"J_ROM: {J_ROM:.3e}, J_FOM: {J_FOM:.3e}, ||dL_du|| = {dL_du_norm:.3e}, ||dL_du||_{opt_step} /"
                            f"||dL_du||_0 = {ratio:.3e}"
                        )
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
                    trunc_deim_modes_list=trunc_deim_modes_list,
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
                    qs_cand = TI_primal_kdv_impl(q0, f_last_valid, J_l, kdv.Nx, kdv.Nt, kdv.dt, **params_primal)
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
            "trunc_deim_modes_list_at_crash": trunc_modes_list,
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
                         "trunc_deim_modes_list_final": trunc_deim_modes_list,
                         "best_control_final": best_control, "best_details_final": best_details,
                         "last_valid_control_final": f_last_valid}

        save_all(data_dir, **to_save_final)

    # ─────────────────────────────────────────────────────────────────────
    # Compute best control based cost
    qs_opt_full = TI_primal_kdv_impl(q0, best_control, J_l, kdv.Nx, kdv.Nt, kdv.dt, **params_primal)
    qs_adj_opt = TI_adjoint_kdv_impl(q0_adj, qs_opt_full, qs_target, J_l_adjoint,
                                     kdv.Nx, kdv.Nt, kdv.dx, kdv.dt, **params_adjoint)
    f_opt = psi @ best_control
    J_s_f, J_ns_f = Calc_Cost(qs_opt_full, qs_target, best_control, C, kwargs['dx'], kwargs['dt'],
                              kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
    J_final = J_s_f + J_ns_f

    # Compute last valid control based cost
    qs_opt_full__ = TI_primal_kdv_impl(q0, f_last_valid, J_l, kdv.Nx, kdv.Nt, kdv.dt, **params_primal)
    qs_adj_opt__ = TI_adjoint_kdv_impl(q0_adj, qs_opt_full__, qs_target, J_l_adjoint,
                                       kdv.Nx, kdv.Nt, kdv.dx, kdv.dt, **params_adjoint)
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
    pf = PlotFlow(kdv.X, kdv.t)
    pf.plot1D(qs_org, name="qs_org", immpath=plot_dir)
    pf.plot1D(qs_target, name="qs_target", immpath=plot_dir)
    pf.plot1D(qs_opt_full, name="qs_opt", immpath=plot_dir)
    pf.plot1D(qs_adj_opt, name="qs_adj_opt", immpath=plot_dir)
    pf.plot1D(f_opt, name="f_opt", immpath=plot_dir)
    pf.plot1D_ROM_converg(J_opt_list, J_opt_FOM_list, name="J", immpath=plot_dir)
