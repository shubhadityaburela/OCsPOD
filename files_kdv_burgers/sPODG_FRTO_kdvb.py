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

# Add local sPOD library to path
sys.path.append('../sPOD/lib/')

# ──────────────── Local Modules ────────────────
from Coefficient_Matrix import CoefficientMatrix

from FOM_solver import (
    IC_primal_kdvb, TI_primal_kdvb_impl, IC_adjoint_kdvb, TI_adjoint_kdvb_impl,
)
from Grads import (
    Calc_Grad_sPODG_FRTO_smooth,
    Calc_Grad_mapping,
)
from Update import (
    get_BB_step, Update_Control_BB_kdvb, Update_Control_sPODG_FRTO_TWBT_kdvb,
)
from grid_params import Korteweg_de_Vries_Burgers
from Plots import PlotFlow
from Costs import Calc_Cost_sPODG, Calc_Cost

from Helper import (
    compute_red_basis,
    calc_shift,
    L2norm_ROM,
    check_weak_divergence, ControlSelectionMatrix_kdvb, compute_deim_basis,
)
from Helper_sPODG import (
    subsample,
    get_T,
    make_V_W_U_delta,
)

from sPODG_solver import (
    IC_primal_sPODG_FRTO_kdvb, IC_adjoint_sPODG_FRTO_kdvb, mat_primal_sPODG_FRTO_kdvb,
    TI_primal_sPODG_FRTO_kdvb_expl, TI_adjoint_sPODG_FRTO_kdvb_impl,
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
    Nx, Nt = 2000, 4000
    kdvb = Korteweg_de_Vries_Burgers(Nx=Nx, timesteps=Nt, cfl=0.4, v_x=220.0,
                                     variance=5e6, offset=30)
    kdvb.Grid()
    return kdvb


def build_dirs(prefix, common_basis, reg_tuple, CTC_mask, TYPE, VAL):
    cb_str = "primal+adjoint_common_basis" if common_basis else "primal_basis"
    reg_str = f"L1={reg_tuple[0]}_L2={reg_tuple[1]}"
    data_dir = os.path.join(prefix, "data/sPODG_FRTO_kdvb", cb_str, reg_str, f"CTC_mask={CTC_mask}",
                            f"{TYPE}={VAL}")
    plot_dir = os.path.join(prefix, "plots/sPODG_FRTO_kdvb", cb_str, reg_str, f"CTC_mask={CTC_mask}",
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
    params_primal = {**shared_dynamics, 'omega': 4e-1, 'gamma': 1e-2, 'nu': 1e-2}
    params_target = {**shared_dynamics, 'omega': 1e0, 'gamma': 1e6, 'nu': 1e5}
    params_adjoint = {**common_params, 'CTC': CTC, 'omega': 4e-1, 'gamma': 1e-2, 'nu': 1e-2}

    J_l = scipy.sparse.identity(kdvb.Nx, format='csc') - 0.5 * kdvb.dt * (
            A_p - params_primal['gamma'] * D3 + params_primal['nu'] * D2)
    J_l_adjoint = scipy.sparse.identity(kdvb.Nx, format='csc') + 0.5 * kdvb.dt * (
            - A_p.T + params_primal['gamma'] * D3.T - params_primal['nu'] * D2.T)
    J_l_target = scipy.sparse.identity(kdvb.Nx, format='csc') - 0.5 * kdvb.dt * (
            A_p - params_target['gamma'] * D3 + params_target['nu'] * D2)

    # Solve uncontrolled FOM once
    qs0 = IC_primal_kdvb(kdvb.X, kdvb.Lx, kdvb.offset, kdvb.variance)
    qs_org = TI_primal_kdvb_impl(qs0, f, J_l, kdvb.Nx, kdvb.Nt, kdvb.dt, **params_primal)
    qs_target = TI_primal_kdvb_impl(qs0, f, J_l_target, kdvb.Nx, kdvb.Nt, kdvb.dt, **params_target)
    q0 = np.ascontiguousarray(qs0)
    q0_adj = np.ascontiguousarray(IC_adjoint_kdvb(kdvb.X))

    # Prepare directories
    data_dir, plot_dir = build_dirs(args.dir_prefix,
                                    args.primal_adjoint_common_basis,
                                    args.reg, args.CTC_mask_activate,
                                    TYPE, VAL)

    # --------------------------------------------------------------------------------- #
    # Precompute full basis once (fixed‐basis approach)

    # Sample the shift values
    # Extract transformation operators based on sub-sampled delta
    delta_s = subsample(kdvb.X, num_sample=kwargs['shift_sample'])
    T_delta, _ = get_T(delta_s, kdvb.X, kdvb.t, interp_order=kwargs['trafo_interp_order'])

    qs_full = qs_org.copy()
    qs_adj_full = TI_adjoint_kdvb_impl(q0_adj, qs_full, qs_target, J_l_adjoint,
                                       kdvb.Nx, kdvb.Nt, kdvb.dx, kdvb.dt, **params_adjoint)

    # Compute shifts of the primal uncontrolled profile
    z = calc_shift(qs_full, q0, kdvb.X, kdvb.t)
    _, T = get_T(z, kdvb.X, kdvb.t, interp_order=kwargs['trafo_interp_order'])
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
    lhs_p, rhs_p, deim_p, deim_p_hl, c_p, tar_a = mat_primal_sPODG_FRTO_kdvb(T_delta, Vd_p, Wd_p, Ud_p, V_deim,
                                                                             kwargs['shift_sample'],
                                                                             Nm, Nm_deim, params_primal, params_adjoint,
                                                                             delta_s)

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

    omega_twbt = 1e-8
    omega_bb = 1e-8
    omega = 1e-8
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
            as_p, as_dot, intIds, weights = TI_primal_sPODG_FRTO_kdvb_expl(lhs_p, rhs_p, deim_p, c_p, a_p, f,
                                                                           delta_s, Nm, kwargs['Nt'], kwargs['dt'])

            # ───── Compute costs ─────
            J_s, J_ns, _ = Calc_Cost_sPODG(Vd_p, as_p[:-1], qs_target, f, C, intIds, weights,
                                           kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
            J_ROM = J_s + J_ns

            qs_opt_full = TI_primal_kdvb_impl(q0, f, J_l, kdvb.Nx, kdvb.Nt, kdvb.dt, **params_primal)
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
            as_adj = TI_adjoint_sPODG_FRTO_kdvb_impl(a_a, f, as_p, qs_target, as_dot, lhs_p, rhs_p, deim_p, deim_p_hl,
                                                     c_p, tar_a, Vd_p, Wd_p,
                                                     delta_s, Nm, kwargs['Nt'], kwargs['dt'], kwargs['dx'],
                                                     params_adjoint)

            # ───── Compute the smooth gradient + the generalized gradient mapping ─────
            dL_du_s = Calc_Grad_sPODG_FRTO_smooth(f, c_p, as_adj, as_p, intIds, weights, kwargs['lamda_l2'])
            dL_du_g = Calc_Grad_mapping(f, dL_du_s, omega, kwargs['lamda_l1'])
            dL_du_norm = np.sqrt(L2norm_ROM(dL_du_g, kwargs['dt']))

            dL_du_norm_list.append(dL_du_norm)

            # ───── Step‐size: BB vs. TWBT, including Armijo‐stagnation logic ─────
            ratio = dL_du_norm / dL_du_norm_list[0]
            if ratio < 5e-3:
                print(f"BB acting.....")
                omega_bb = get_BB_step(fOld, f, dL_du_Old, dL_du_s, opt_step, **kwargs)
                if omega_bb < 0:
                    print("WARNING: BB gave negative step size thus resorting to using TWBT")
                    fNew, omega_twbt, stag = Update_Control_sPODG_FRTO_TWBT_kdvb(f, lhs_p, rhs_p, deim_p, deim_p_hl,
                                                                                 c_p, Vd_p,
                                                                                 a_p, qs_target, delta_s, J_s,
                                                                                 omega_twbt, Nm, dL_du_s, C, adjust,
                                                                                 **kwargs)
                    omega = omega_twbt
                else:
                    fNew = Update_Control_BB_kdvb(f, dL_du_s, omega_bb, kwargs['lamda_l1'])
                    stag = False
                    omega = omega_bb
            else:
                print("TWBT acting…")
                fNew, omega_twbt, stag = Update_Control_sPODG_FRTO_TWBT_kdvb(f, lhs_p, rhs_p, deim_p, deim_p_hl,
                                                                             c_p, Vd_p,
                                                                             a_p, qs_target, delta_s, J_s,
                                                                             omega_twbt, Nm, dL_du_s, C, adjust,
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
                qs_opt_full = TI_primal_kdvb_impl(q0, f_last_valid, J_l, kdvb.Nx, kdvb.Nt, kdvb.dt, **params_primal)
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
                qs_opt_full = TI_primal_kdvb_impl(q0, f_last_valid, J_l, kdvb.Nx, kdvb.Nt, kdvb.dt, **params_primal)
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
                    qs_cand = TI_primal_kdvb_impl(q0, f_last_valid, J_l, kdvb.Nx, kdvb.Nt, kdvb.dt, **params_primal)
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
                         "trunc_modes_list_final": trunc_modes_list, "trunc_deim_modes_list_final": trunc_deim_modes_list,
                         "best_control_final": best_control, "best_details_final": best_details,
                         "last_valid_control_final": f_last_valid}

        save_all(data_dir, **to_save_final)

    # ─────────────────────────────────────────────────────────────────────
    # Compute best control based cost
    qs_opt_full = TI_primal_kdvb_impl(q0, best_control, J_l, kdvb.Nx, kdvb.Nt, kdvb.dt, **params_primal)
    qs_adj_opt = TI_adjoint_kdvb_impl(q0_adj, qs_opt_full, qs_target, J_l_adjoint,
                                      kdvb.Nx, kdvb.Nt, kdvb.dx, kdvb.dt, **params_adjoint)
    f_opt = psi @ best_control
    J_s_f, J_ns_f = Calc_Cost(qs_opt_full, qs_target, best_control, C, kwargs['dx'], kwargs['dt'],
                              kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
    J_final = J_s_f + J_ns_f

    # Compute last valid control based cost
    qs_opt_full__ = TI_primal_kdvb_impl(q0, f_last_valid, J_l, kdvb.Nx, kdvb.Nt, kdvb.dt, **params_primal)
    qs_adj_opt__ = TI_adjoint_kdvb_impl(q0_adj, qs_opt_full__, qs_target, J_l_adjoint,
                                        kdvb.Nx, kdvb.Nt, kdvb.dx, kdvb.dt, **params_adjoint)
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
    pf = PlotFlow(kdvb.X, kdvb.t)
    pf.plot1D(qs_opt_full, name="qs_opt", immpath=plot_dir)
    pf.plot1D(qs_adj_opt, name="qs_adj_opt", immpath=plot_dir)
    pf.plot1D(f_opt, name="f_opt", immpath=plot_dir)
    pf.plot1D_ROM_converg(J_opt_list, J_opt_FOM_list, name="J", immpath=plot_dir)

# qqqq = get_approx_state_sPODG(Vd_p, f, as_p[:-1, :], intIds, weights, kwargs['Nx'], kwargs['Nt'])
# print(np.linalg.norm(qs_org - qqqq) / np.linalg.norm(qs_org))
# plt.pcolormesh(qqqq.T)
# plt.show()
#
# # Faster animation by subsampling frames and overlaying two curves
# skip = 20  # show every 20th time step
# frames = range(0, len(kdvb.t), skip)
# fig, ax = plt.subplots(figsize=(8, 4))
# # plot both on the same axes, with labels
# line1, = ax.plot(kdvb.X, qs_org[:, 0], lw=2, label='Original')
# line2, = ax.plot(kdvb.X, qqqq[:, 0], lw=2, label='implicit_midpoint')
# ax.set_xlim(kdvb.X.min(), kdvb.X.max())
# ax.set_ylim(-0.1, 1.3)
# ax.set_xlabel('x')
# ax.set_ylabel('u(x,t)')
# ax.legend()
# title = ax.set_title('')
#
#
# def update(frame_index):
#     idx = frames[frame_index]
#     line1.set_ydata(qs_org[:, idx])
#     line2.set_ydata(qqqq[:, idx])
#     title.set_text(f"t = {kdvb.t[idx]:.2f}")
#     return line1, line2, title
#
#
# ani = animation.FuncAnimation(
#     fig, update, frames=len(frames), interval=30, blit=False
# )
# plt.show()
#
# exit()
