import argparse
import os
import sys
import time
import traceback
from ast import literal_eval
from time import perf_counter

import numpy as np
import scipy
from matplotlib import animation, pyplot as plt

from Coefficient_Matrix import CoefficientMatrix
from Costs import Calc_Cost
from FOM_solver import IC_primal_kdv, TI_primal_kdv_impl, TI_adjoint_kdv_impl, IC_adjoint_kdv, TI_adjoint_kdv_expl
from Grads import Calc_Grad_smooth, Calc_Grad_mapping
from Helper import ControlSelectionMatrix_kdvb, L2norm_ROM, check_weak_divergence, L2inner_prod, calc_shift, \
    compute_red_basis
from Helper_sPODG import get_T
from Update import get_BB_step, Update_Control_TWBT_kdv, Update_Control_BB_kdv
from grid_params import Korteweg_de_Vries
from Plots import PlotFlow

np.random.seed(0)


# ───────────────────────────────────────────────────────────────────────
def parse_arguments():
    p = argparse.ArgumentParser(description="Input the variables for running the script.")
    p.add_argument("fully_nonlinear", type=literal_eval, choices=[True, False],
                   help="Select True for fully nonlinear else False? (True/False)")
    p.add_argument("grid", type=int, nargs=3, metavar=("Nx", "Nt", "cfl_fac"),
                   help="Enter the grid resolution and the cfl factor")
    p.add_argument("N_iter", type=int, help="Number of optimization iterations")
    p.add_argument("dir_prefix", type=str, choices=[".", "/work/burela"],
                   help="Directory prefix for I/O")
    p.add_argument("CTC_mask_activate", type=literal_eval, choices=[True, False],
                   help="Include CTC mask in the system? (True/False)")
    p.add_argument("reg", type=float, nargs=2, metavar=("L1", "L2"),
                   help="L1 and L2 regularization weights (e.g. 0.01 0.001)")
    return p.parse_args()


def setup_kdv(Nx, Nt, cfl_fac):
    kdv = Korteweg_de_Vries(Nx=Nx, timesteps=Nt, cfl=0.0425 / cfl_fac, v_x=8 / 3, offset=20)
    kdv.Grid()
    return kdv


def build_dirs(prefix, fully_nonlinear, reg_tuple, CTC_mask):
    reg_str = f"L1={reg_tuple[0]}_L2={reg_tuple[1]}"
    fnl_str = f"fully_nonlinear={fully_nonlinear}"
    data_dir = os.path.join(prefix, "data/FOM_kdv", fnl_str, reg_str, f"CTC_mask={CTC_mask}")
    plot_dir = os.path.join(prefix, "plots/FOM_kdv", fnl_str, reg_str, f"CTC_mask={CTC_mask}")
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


def C_matrix(Nx, CTC_end_index, apply_CTC_mask=False):  # For now not active
    C = np.ones(Nx)
    if apply_CTC_mask:
        return C == 1
    else:
        return C == 1


# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_arguments()

    print(f"Problem type (full nonlinearity)= {args.fully_nonlinear}")
    print(f"L1, L2 regularization = {tuple(args.reg)}")
    print(f"Grid = {tuple(args.grid)}")

    # Unpack regularization parameters
    L1_reg, L2_reg = args.reg
    Nx, Nt, cfl_fac = args.grid

    # Set up kdvb and control matrix
    kdv = setup_kdv(Nx, Nt, cfl_fac)

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
        'delta': 1 / 2,
        'opt_iter': args.N_iter,
        'beta': 1 / 2,  # for TWBT
        'verbose': True,
        'omega_cutoff': 1e-10,
        'perform_grad_check': True,
    }
    f = np.zeros((n_c, kdv.Nt))  # initial control guess
    df = np.random.randn(*f.shape)
    psi = scipy.sparse.csc_matrix(psi)

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

    L_p = - shared_params['alpha'] * shared_params['c'] * D1 - shared_params['gamma'] * D3 + shared_params['nu'] * D2
    L_t = - target_params['alpha'] * target_params['c'] * D1 - target_params['gamma'] * D3 + target_params['nu'] * D2
    L_a = shared_params['alpha'] * shared_params['c'] * D1.T + shared_params['gamma'] * D3.T - shared_params['nu'] * D2.T

    params_primal = {**shared_dynamics, 'L': L_p, **shared_params}
    params_target = {**shared_dynamics, 'L': L_t, **target_params}
    params_adjoint = {**common_params, 'CTC': CTC, 'L': L_a, **shared_params}

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
    data_dir, plot_dir = build_dirs(args.dir_prefix, args.fully_nonlinear, args.reg, args.CTC_mask_activate)

    # Collector lists
    dL_du_norm_list = []
    J_opt_list = []
    running_time = []
    best_control = np.zeros_like(f)
    best_details = {'J': np.inf, 'N_iter': None}
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

            # ───── Forward FOM:  ─────
            qs = TI_primal_kdv_impl(q0, f, J_l, kdv.Nx, kdv.Nt, kdv.dt, **params_primal)

            # ───── Compute costs ─────
            J_s, J_ns = Calc_Cost(qs, qs_target, f, C, kwargs['dx'], kwargs['dt'],
                                  kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)

            J_FOM = J_s + J_ns
            J_opt_list.append(J_FOM)

            # Track best control
            if J_FOM < best_details['J']:
                best_details.update({'J': J_FOM, 'N_iter': opt_step})
                best_control = f.copy()

            # ───── Backward FOM (adjoint) ─────
            qs_adj = TI_adjoint_kdv_impl(q0_adj, qs, qs_target, J_l_adjoint, kdv.Nx, kdv.Nt, kdv.dx, kdv.dt,
                                         **params_adjoint)

            # ───── Compute the smooth gradient + the generalized gradient mapping ─────
            dL_du_s = Calc_Grad_smooth(psi, f, qs_adj, kwargs['lamda_l2'])
            dL_du_g = Calc_Grad_mapping(f, dL_du_s, omega, kwargs['lamda_l1'])
            dL_du_norm = np.sqrt(L2norm_ROM(dL_du_g, kwargs['dt']))

            dL_du_norm_list.append(dL_du_norm)

            # ───── Gradient check with Finite differences ─────
            if kwargs['perform_grad_check']:
                print("-------------GRAD CHECK-----------------")
                eps = 1e-5
                f_rand = f + eps * df
                qs_rand = TI_primal_kdv_impl(q0, f_rand, J_l, kdv.Nx, kdv.Nt, kdv.dt, **params_primal)
                J_s_eps, J_ns_eps = Calc_Cost(qs_rand, qs_target, f_rand, C, kwargs['dx'], kwargs['dt'],
                                              kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
                J_FOM_eps = J_s_eps + J_ns_eps
                print("Finite difference gradient", (J_FOM_eps - J_FOM) / eps)
                print("Analytic gradient", L2inner_prod(dL_du_g, df, kwargs['dt']))

            # ───── Step‐size: BB vs. TWBT, including Armijo‐stagnation logic ─────
            ratio = dL_du_norm / dL_du_norm_list[0]
            if ratio < 5e-3:
                print(f"BB acting.....")
                omega_bb = get_BB_step(fOld, f, dL_du_Old, dL_du_s, opt_step, **kwargs)
                if omega_bb < 0:
                    print("WARNING: BB gave negative step size thus resorting to using TWBT")
                    fNew, omega_twbt, stag = Update_Control_TWBT_kdv(f, q0, qs_target, J_s, omega_twbt,
                                                                     dL_du_s,
                                                                     C, adjust, J_l, params_primal, **kwargs)
                    omega = omega_twbt
                else:
                    fNew = Update_Control_BB_kdv(f, dL_du_s, omega_bb, kwargs['lamda_l1'])
                    stag = False
                    omega = omega_bb
            else:
                print("TWBT acting…")
                fNew, omega_twbt, stag = Update_Control_TWBT_kdv(f, q0, qs_target, J_s, omega_twbt, dL_du_s,
                                                                 C, adjust, J_l, params_primal, **kwargs)
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
                qs_opt_full = TI_primal_kdv_impl(q0, f_last_valid, J_l, kwargs['Nx'], kwargs['Nt'], kwargs['dt'],
                                                 **params_primal)
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
                qs_opt_full = TI_primal_kdv_impl(q0, f_last_valid, J_l, kwargs['Nx'], kwargs['Nt'], kwargs['dt'],
                                                 **params_primal)
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
                    if J_FOM > 1e10 or abs(omega_bb) < kwargs['omega_cutoff']:
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
                    qs_cand = TI_primal_kdv_impl(q0, f_last_valid, J_l, kwargs['Nx'], kwargs['Nt'], kwargs['dt'],
                                                 **params_primal)
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
    qs_opt_full = TI_primal_kdv_impl(q0, best_control, J_l, kwargs['Nx'], kwargs['Nt'], kwargs['dt'],
                                     **params_primal)
    qs_adj_opt = TI_adjoint_kdv_impl(q0_adj, qs_opt_full, qs_target, J_l_adjoint,
                                     kdv.Nx, kdv.Nt, kdv.dx, kdv.dt, **params_adjoint)
    f_opt = psi @ best_control
    J_s_f, J_ns_f = Calc_Cost(qs_opt_full, qs_target, best_control, C, kwargs['dx'], kwargs['dt'],
                              kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
    J_final = J_s_f + J_ns_f

    # Compute last valid control based cost
    qs_opt_full__ = TI_primal_kdv_impl(q0, f_last_valid, J_l, kwargs['Nx'], kwargs['Nt'], kwargs['dt'],
                                       **params_primal)
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
    # np.save(os.path.join(data_dir, "qs_target.npy"), qs_target)

    # Plot results
    pf = PlotFlow(kdv.X, kdv.t)
    pf.plot1D(qs_org, name="qs_org", immpath=plot_dir)
    pf.plot1D(qs_target, name="qs_target", immpath=plot_dir)
    pf.plot1D(qs_opt_full, name="qs_opt", immpath=plot_dir)
    pf.plot1D(qs_adj_opt, name="qs_adj_opt", immpath=plot_dir)
    pf.plot1D(f_opt, name="f_opt", immpath=plot_dir)
    pf.plot1D_FOM_converg(J_opt_list, name="J", immpath=plot_dir)

    # # Faster animation by subsampling frames and overlaying two curves
    # skip = 20  # show every 20th time step
    # frames = range(0, len(kdvb.t), skip)
    # fig, ax = plt.subplots(figsize=(8, 4))
    # # plot both on the same axes, with labels
    # line1, = ax.plot(kdvb.X, qs_opt_full[:, 0], lw=2, label='Controlled')
    # line2, = ax.plot(kdvb.X, qs_target[:, 0], lw=2, label='Target')
    # ax.set_xlim(kdvb.X.min(), kdvb.X.max())
    # ax.set_ylim(-0.1, 1)
    # ax.set_xlabel('x')
    # ax.set_ylabel('u(x,t)')
    # ax.legend()
    # title = ax.set_title('')
    # def update(frame_index):
    #     idx = frames[frame_index]
    #     line1.set_ydata(qs_opt_full[:, idx])
    #     line2.set_ydata(qs_target[:, idx])
    #     title.set_text(f"t = {kdvb.t[idx]:.2f}")
    #     return line1, line2, title
    # ani = animation.FuncAnimation(
    #     fig, update, frames=len(frames), interval=30, blit=False
    # )
    # plt.show()

    # # Faster animation by subsampling frames
    # skip = 20  # show every 10th time step
    # frames = range(0, len(kdvb.t), skip)
    # fig, ax = plt.subplots(figsize=(8, 4))
    # line, = ax.plot(kdvb.X, qs_org[..., 0], lw=2)
    # ax.set_xlim(kdvb.X.min(), kdvb.X.max())
    # ax.set_ylim(-0.1, 1)
    # ax.set_xlabel('x');
    # ax.set_ylabel('u(x,t)')
    # title = ax.set_title('')
    # def update(frame_index):
    #     idx = frames[frame_index]
    #     line.set_ydata(qs_org[..., idx])
    #     title.set_text(f"t = {kdvb.t[idx]:.2f}")
    #     return line, title
    # ani = animation.FuncAnimation(
    #     fig, update, frames=len(frames), interval=30, blit=False
    # )
    # plt.show()
    #
    # exit()

    # # Faster animation by subsampling frames and overlaying two curves
    # skip = 20  # show every 20th time step
    # frames = range(0, len(kdvb.t), skip)
    # fig, ax = plt.subplots(figsize=(8, 4))
    # # plot both on the same axes, with labels
    # line1, = ax.plot(kdvb.X, qs_org[:, 0], lw=2, label='Original')
    # line2, = ax.plot(kdvb.X, qs_target[:, 0], lw=2, label='Target')
    # ax.set_xlim(kdvb.X.min(), kdvb.X.max())
    # # ax.set_ylim(-1, 2)
    # ax.set_xlabel('x')
    # ax.set_ylabel('u(x,t)')
    # ax.legend()
    # title = ax.set_title('')
    # def update(frame_index):
    #     idx = frames[frame_index]
    #     line1.set_ydata(qs_org[:, idx])
    #     line2.set_ydata(qs_target[:, idx])
    #     title.set_text(f"t = {kdvb.t[idx]:.2f}")
    #     return line1, line2, title
    # ani = animation.FuncAnimation(
    #     fig, update, frames=len(frames), interval=30, blit=False
    # )
    # plt.show()
    #
    #
    # plt.pcolormesh(qs_org.T)
    # plt.show()
    # plt.pcolormesh(qs_target.T)
    # plt.show()
    # exit()

    # # Faster animation by subsampling frames and overlaying two curves
    # skip = 20  # show every 20th time step
    # frames = range(0, len(kdv.t), skip)
    # fig, ax = plt.subplots(figsize=(8, 4))
    # # plot both on the same axes, with labels
    # line1, = ax.plot(kdv.X, qs_org[:, 0], lw=2, label='Original')
    # line2, = ax.plot(kdv.X, qs_target[:, 0], lw=2, label='Target')
    # ax.set_xlim(kdv.X.min(), kdv.X.max())
    # # ax.set_ylim(-1, 2)
    # ax.set_xlabel('x')
    # ax.set_ylabel('u(x,t)')
    # ax.legend()
    # title = ax.set_title('')
    # def update(frame_index):
    #     idx = frames[frame_index]
    #     line1.set_ydata(qs_org[:, idx])
    #     line2.set_ydata(qs_target[:, idx])
    #     title.set_text(f"t = {kdv.t[idx]:.2f}")
    #     return line1, line2, title
    # ani = animation.FuncAnimation(
    #     fig, update, frames=len(frames), interval=30, blit=False
    # )
    # plt.show()
    #
    #
    # plt.pcolormesh(qs_org.T)
    # plt.show()
    # plt.pcolormesh(qs_target.T)
    # plt.show()
    #
    # print("------------------------------\n")
    # u1 = f.copy()
    # J_s, _ = Calc_Cost(qs_org, qs_target, u1, C, kwargs['dx'], kwargs['dt'],
    #                    kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
    # v = np.random.randn(*u1.shape)
    # eps_list = [1e-5]
    # for eps in eps_list:
    #     u2 = u1 + eps * v
    #     qs2 = TI_primal_kdv_impl(q0, u2, J_l, kdv.Nx, kdv.Nt, kdv.dt, **params_primal)
    #     J_s_eps, _ = Calc_Cost(qs2, qs_target, u2, C, kwargs['dx'], kwargs['dt'],
    #                            kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
    #     print(f"epsilon: {eps}", (J_s_eps - J_s) / eps)
    #
    # qs_adj = TI_adjoint_kdv_impl(q0_adj, qs_org, qs_target, J_l_adjoint, kdv.Nx, kdv.Nt, kdv.dx, kdv.dt,
    #                              **params_adjoint)
    #
    # print(np.linalg.norm(qs_adj))
    # dL_du_s = Calc_Grad_smooth(psi, u1, qs_adj, kwargs['lamda_l2'])
    # print("Analytic gradient", L2inner_prod(dL_du_s, v, kwargs['dt']))
    #
    # exit()
