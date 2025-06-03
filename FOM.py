import argparse
import os
import sys
import time
import traceback
from ast import literal_eval
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from Coefficient_Matrix import CoefficientMatrix
from Costs import Calc_Cost
from FOM_solver import IC_primal, TI_primal, TI_primal_target, IC_adjoint, TI_adjoint
from Grads import Calc_Grad_smooth, Calc_Grad_mapping
from Helper import ControlSelectionMatrix_advection, L2norm_ROM, check_weak_divergence
from TI_schemes import DF_start_FOM
from Update import get_BB_step, Update_Control_BB, Update_Control_TWBT
from grid_params import advection
from Plots import PlotFlow


# ───────────────────────────────────────────────────────────────────────
def parse_arguments():
    p = argparse.ArgumentParser(description="Input the variables for running the script.")
    p.add_argument("problem", type=int, choices=[1, 2, 3],
                   help="Problem number (1, 2, or 3)")
    p.add_argument("N_iter", type=int, help="Number of optimization iterations")
    p.add_argument("dir_prefix", type=str, choices=[".", "/work/burela"],
                   help="Directory prefix for I/O")
    p.add_argument("CTC_mask_activate", type=literal_eval, choices=[True, False],
                   help="Include CTC mask in the system? (True/False)")
    p.add_argument("reg", type=float, nargs=2, metavar=("L1", "L2"),
                   help="L1 and L2 regularization weights (e.g. 0.01 0.001)")
    return p.parse_args()


def setup_advection(problem):
    Nxi, Nt = 3200, 3360
    if problem == 1:
        wf = advection(Nxi=Nxi, timesteps=Nt,
                       cfl=8 / 6, tilt_from=3 * Nt // 4,
                       v_x=0.5, v_x_t=1.0,
                       variance=7, offset=12)
    elif problem == 2:
        wf = advection(Nxi=Nxi, timesteps=Nt,
                       cfl=8 / 6, tilt_from=3 * Nt // 4,
                       v_x=0.55, v_x_t=1.0,
                       variance=0.5, offset=30)
    else:
        wf = advection(Nxi=Nxi, timesteps=Nt,
                       cfl=8 / 6, tilt_from=9 * Nt // 10,
                       v_x=0.6, v_x_t=1.3,
                       variance=0.5, offset=30)
    wf.Grid()
    return wf


def build_dirs(prefix, reg_tuple, CTC_mask, problem):
    reg_str = f"L1={reg_tuple[0]}_L2={reg_tuple[1]}"
    data_dir = os.path.join(prefix, "data/FOM", reg_str, f"CTC_mask={CTC_mask}", f"problem={problem}")
    plot_dir = os.path.join(prefix, "plots/FOM", reg_str, f"CTC_mask={CTC_mask}", f"problem={problem}")
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
                     J_opt_list, dL_du_norm_list, running_time):
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
    np.save(os.path.join(ckpt_dir, "dL_du_norm_list.npy"), np.array(dL_du_norm_list))
    np.save(os.path.join(ckpt_dir, "running_time.npy"), np.array(running_time))

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

    print(f"\nSolving problem: {args.problem}")
    print(f"L1, L2 regularization = {tuple(args.reg)}")

    # Unpack regularization parameters
    L1_reg, L2_reg = args.reg

    # Set up WF and control matrix
    wf = setup_advection(args.problem)
    if L1_reg != 0 and L2_reg == 0:  # Purely L1
        n_c_init = wf.Nxi
        psi = ControlSelectionMatrix_advection(wf, n_c_init, Gaussian=False, gaussian_mask_sigma=0.5)
        adjust = 1.0
    else:  # Mix type
        n_c_init = 40
        psi = ControlSelectionMatrix_advection(wf, n_c_init, Gaussian=True, gaussian_mask_sigma=0.5)
        adjust = wf.dx
    n_c = psi.shape[1]
    f = np.zeros((n_c, wf.Nt))  # initial control guess

    # Build coefficient matrices
    Mat = CoefficientMatrix(orderDerivative=wf.firstderivativeOrder,
                            Nxi=wf.Nxi, Neta=1,
                            periodicity='Periodic',
                            dx=wf.dx, dy=0)
    A_p = - wf.v_x[0] * Mat.Grad_Xi_kron
    A_a = A_p.transpose()

    # Solve uncontrolled FOM once
    qs0 = IC_primal(wf.X, wf.Lxi, wf.offset, wf.variance)
    qs_org = TI_primal(qs0, f, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
    qs_target = TI_primal_target(qs0, Mat.Grad_Xi_kron, wf.v_x_target, wf.Nxi, wf.Nt, wf.dt)
    q0 = np.ascontiguousarray(qs0)
    q0_adj = np.ascontiguousarray(IC_adjoint(wf.X))

    # Calculate the CTC matrix/array  (CTC and C are exactly the same)
    C = C_matrix(wf.Nxi, wf.CTC_end_index, apply_CTC_mask=args.CTC_mask_activate)

    # Prepare directories
    data_dir, plot_dir = build_dirs(args.dir_prefix, args.reg, args.CTC_mask_activate, args.problem)

    # Prepare kwargs
    kwargs = {
        'dx': wf.dx,
        'dt': wf.dt,
        'Nx': wf.Nxi,
        'Nt': wf.Nt,
        'n_c': n_c,
        'lamda_l1': L1_reg,
        'lamda_l2': L2_reg,
        'delta_conv': 1e-4,
        'delta': 1 / 2,  # Armijo constant  USE 1.01 with L1_reg=0 for getting the older results
        'opt_iter': args.N_iter,
        'beta': 1 / 2,  # for TWBT
        'verbose': True,
        'omega_cutoff': 1e-10,
        'adjoint_scheme': "RK4",
    }

    # Select the LU pre-factors for the inverse of mass matrix for linear solve of adjoint equation
    if kwargs['adjoint_scheme'] == "RK4":
        M_f = None
        A_f = A_a.copy()
        LU_M_f = None
        Df = None
    elif kwargs['adjoint_scheme'] == "implicit_midpoint":
        M_f = sparse.eye(kwargs['Nx'], format="csc") + (- kwargs['dt']) / 2 * A_a
        A_f = sparse.eye(kwargs['Nx'], format="csc") - (- kwargs['dt']) / 2 * A_a
        LU_M_f = splu(M_f)
        Df = None
    elif kwargs['adjoint_scheme'] == "DIRK":
        M_f = sparse.eye(kwargs['Nx'], format="csc") + (- kwargs['dt']) / 4 * A_a
        A_f = A_a.copy()
        LU_M_f = splu(M_f)
        Df = None
    elif kwargs['adjoint_scheme'] == "BDF2":
        M_f = 3.0 * sparse.eye(kwargs['Nx'], format="csc") + 2.0 * (- kwargs['dt']) * A_a
        A_f = A_a.copy()
        LU_M_f = splu(M_f)
        Df = None
    elif kwargs['adjoint_scheme'] == "BDF3":
        M_f = 11.0 * sparse.eye(kwargs['Nx'], format="csc") + 6.0 * (- kwargs['dt']) * A_a
        A_f = A_a.copy()
        LU_M_f = splu(M_f)
        Df = csc_matrix(DF_start_FOM(A_a.todense(), kwargs['Nx'], - kwargs['dt']))
    elif kwargs['adjoint_scheme'] == "BDF4":
        M_f = 25.0 * sparse.eye(kwargs['Nx'], format="csc") + 12.0 * (- kwargs['dt']) * A_a
        A_f = A_a.copy()
        LU_M_f = splu(M_f)
        Df = csc_matrix(DF_start_FOM(A_a.todense(), kwargs['Nx'], - kwargs['dt']).tocsc())
    else:
        kwargs['adjoint_scheme'] = "RK4"
        M_f = None
        A_f = A_a.copy()
        LU_M_f = None
        Df = None

    # Collector lists
    dL_du_norm_list = []
    J_opt_list = []
    running_time = []
    best_control = np.zeros_like(f)
    best_details = {'J': np.inf, 'N_iter': None}
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

            # ───── Forward FOM:  ─────
            qs = TI_primal(q0, f, A_p, psi, wf.Nxi, wf.Nt, wf.dt)

            if opt_step % 5 == 0:
                plt.pcolormesh((psi @ f).T)
                plt.show()

            # ───── Compute costs ─────
            J_s, J_ns = Calc_Cost(qs, qs_target, f, C, kwargs['dx'], kwargs['dt'],
                                  kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
            J_ROM = J_s + J_ns

            qs_opt_full = TI_primal(q0, f, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
            JJ_s, JJ_ns = Calc_Cost(qs_opt_full, qs_target, f, C, kwargs['dx'], kwargs['dt'],
                                    kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
            J_FOM = JJ_s + JJ_ns

            J_opt_list.append(J_ROM)

            # Track best control
            if J_FOM < best_details['J']:
                best_details.update({'J': J_FOM, 'N_iter': opt_step})
                best_control = f.copy()

            # ───── Backward FOM (adjoint) ─────
            qs_adj = TI_adjoint(q0_adj, qs, qs_target, M_f, A_f, LU_M_f, C, wf.Nxi, wf.dx, wf.Nt, wf.dt,
                                scheme=kwargs['adjoint_scheme'], opt_poly_jacobian=Df)

            # ───── Compute the smooth gradient + the generalized gradient mapping ─────
            dL_du_s = Calc_Grad_smooth(psi, f, qs_adj, kwargs['lamda_l2'])
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
                    fNew, omega_twbt, stag = Update_Control_TWBT(f, q0, qs_target, psi, A_p, J_s, omega_twbt, dL_du_s,
                                                                 C, adjust, **kwargs)
                    omega = omega_twbt
                else:
                    fNew = Update_Control_BB(f, dL_du_s, omega_bb, kwargs['lamda_l1'])
                    stag = False
                    omega = omega_bb
            else:
                print("TWBT acting…")
                fNew, omega_twbt, stag = Update_Control_TWBT(f, q0, qs_target, psi, A_p, J_s, omega_twbt, dL_du_s,
                                                             C, adjust, **kwargs)
                omega = omega_twbt

            t1 = perf_counter()
            running_time.append(t1 - t0)
            t0 = t1

            # Saving previous controls for Barzilai Borwein step
            fOld = f.copy()
            f = fNew.copy()
            dL_du_Old = dL_du_s.copy()

            print(
                f"J_FOM: {J_FOM:.3e}, ||dL_du|| = {dL_du_norm:.3e}, ||dL_du||_{opt_step} / "
                f"||dL_du||_0 = {ratio:.3e}"
            )

            # Convergence Criteria
            if opt_step == kwargs['opt_iter'] - 1:
                print("\n\n-------------------------------")
                print(
                    f"WARNING... maximal number of steps reached, "
                    f"J_FOM: {J_FOM:.3e}, ||dL_du|| = {dL_du_norm:.3e}, ||dL_du||_{opt_step} / "
                    f"||dL_du||_0 = {ratio:.3e}"
                )
                f_last_valid = fNew.copy()
                qs_opt_full = TI_primal(q0, f_last_valid, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
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
                    f"J_FOM: {J_FOM:.3e}, ||dL_du|| = {dL_du_norm:.3e}, ||dL_du||_{opt_step} / "
                    f"||dL_du||_0 = {ratio:.3e}"
                )
                f_last_valid = fNew.copy()
                qs_opt_full = TI_primal(q0, f_last_valid, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
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
                            f"J_FOM: {J_FOM:.3e}, ||dL_du|| = {dL_du_norm:.3e}, ||dL_du||_{opt_step} /"
                            f"||dL_du||_0 = {ratio:.3e}")
                        break
                else:
                    dJ = (J_opt_list[-1] - J_opt_list[-2]) / J_opt_list[0]
                    if abs(dJ) == 0:
                        print(f"WARNING: dJ ~ 0 → stopping"
                              f"J_FOM: {J_FOM:.3e}, ||dL_du|| = {dL_du_norm:.3e}, ||dL_du||_{opt_step} /"
                              f"||dL_du||_0 = {ratio:.3e}"
                              )
                        break
                    if stag:
                        print("\n-------------------------------")
                        print(
                            f"TWBT Armijo Stagnated !!!!!! due to the step length being too low thus exiting at itr: {opt_step} with "
                            f"J_FOM: {J_FOM:.3e}, ||dL_du|| = {dL_du_norm:.3e}, ||dL_du||_{opt_step} /"
                            f"||dL_du||_0 = {ratio:.3e}"
                        )
                        break
                    if J_FOM > 1e6 or abs(omega_bb) < kwargs['omega_cutoff']:
                        print("\n\n-------------------------------")
                        print(
                            f"Barzilai Borwein acceleration failed!!!!!! J_FOM increased to unrealistic values or the "
                            f"omega went below cutoff, thus exiting"
                            f"at itr: {opt_step} with "
                            f"J_FOM: {J_FOM:.3e}, ||dL_du|| = {dL_du_norm:.3e}, ||dL_du||_{opt_step} /"
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
                    dL_du_norm_list=dL_du_norm_list,
                    running_time=running_time,
                )

                # Call the helper to check for weak divergence
                diverging, avg_prev, avg_last = check_weak_divergence(J_opt_list, window=100, margin=0.0)
                if diverging:
                    print(
                        "\n*** ROM is no longer accurate thus exiting !!!!: "
                        f"avg(J_FOM[-100:]) = {avg_last:.3e}  > "
                        f"avg(J_FOM[-200:-100]) = {avg_prev:.3e} → exiting ***"
                    )

                    # store last valid control and possibly update best_details
                    f_last_valid = f.copy()
                    qs_cand = TI_primal(q0, f_last_valid, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
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
            "running_time_at_crash": running_time,
            "dL_du_norm_list_at_crash": dL_du_norm_list,
        }
        if f_last_valid is not None:
            to_save["last_valid_control_at_crash"] = f_last_valid
        to_save["best_control_at_crash"] = best_control
        to_save["best_details_at_crash"] = best_details

        save_all(data_dir, **to_save)
        sys.exit(1)

    finally:
        print("\nFinal save…")
        to_save_final = {"J_opt_list_final": J_opt_list,
                         "running_time_final": running_time, "dL_du_norm_list_final": dL_du_norm_list,
                         "best_control_final": best_control, "best_details_final": best_details,
                         "last_valid_control_final": f_last_valid}

        save_all(data_dir, **to_save_final)

    # ─────────────────────────────────────────────────────────────────────
    # Compute best control based cost
    qs_opt_full = TI_primal(q0, best_control, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
    qs_adj_opt = TI_adjoint(q0_adj, qs_opt_full, qs_target, None, A_a, None, C, wf.Nxi, wf.dx, wf.Nt, wf.dt, scheme="RK4")
    f_opt = psi @ best_control
    J_s_f, J_ns_f = Calc_Cost(qs_opt_full, qs_target, best_control, C, kwargs['dx'], kwargs['dt'],
                              kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
    J_final = J_s_f + J_ns_f

    # Compute last valid control based cost
    qs_opt_full__ = TI_primal(q0, f_last_valid, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
    qs_adj_opt__ = TI_adjoint(q0_adj, qs_opt_full__, qs_target, None, A_a, None, C, wf.Nxi, wf.dx, wf.Nt, wf.dt,
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
    pf.plot1D_FOM_converg(J_opt_list, name="J", immpath=plot_dir)
