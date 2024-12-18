"""
This file is the adaptive version. It can handle both the scenarios.
1. Fixed tolerance
2. Fixed modes
"""


from Coefficient_Matrix import CoefficientMatrix
from Costs import Calc_Cost_PODG, Calc_Cost
from Helper import ControlSelectionMatrix_advection, compute_red_basis
from Update import Update_Control_PODG_FRTO_adaptive, \
    Update_Control_PODG_FRTO_adaptive_TWBT
from advection import advection
from Plots import PlotFlow
import sys
import numpy as np
import os
from time import perf_counter
import time

impath = "./data/PODG/FRTO/Nm=50,TWBT/"  # for data
immpath = "./plots/PODG/FRTO/Nm=50,TWBT/"  # for plots
os.makedirs(impath, exist_ok=True)

# Problem variables
Dimension = "1D"
Nxi = 400
Neta = 1
Nt = 700

# solver initialization along with grid initialization
wf = advection(Nxi=Nxi, Neta=Neta if Dimension == "1D" else Nxi, timesteps=Nt, cfl=0.8, tilt_from=3*Nt//4)
wf.Grid()

#%%
n_c = 40  # Number of controls
f = np.zeros((n_c, wf.Nt))  # Initial guess for the control

# Selection matrix for the control input
psi = ControlSelectionMatrix_advection(wf, n_c)

#%% Assemble the linear operators
Mat = CoefficientMatrix(orderDerivative=wf.firstderivativeOrder, Nxi=wf.Nxi,
                        Neta=wf.Neta, periodicity='Periodic', dx=wf.dx, dy=wf.dy)
# Convection matrix (Needs to be changed if the velocity is time dependent)
A_p = - (wf.v_x[0] * Mat.Grad_Xi_kron + wf.v_y[0] * Mat.Grad_Eta_kron)
A_a = A_p.transpose()

#%% Solve the uncontrolled system
qs_org = wf.TI_primal(wf.IC_primal(), f, A_p, psi)
np.save(impath + 'qs_org.npy', qs_org)

qs_target = wf.TI_primal_target(wf.IC_primal(), Mat, np.zeros((wf.Nxi * wf.Neta, wf.Nt)))
np.save(impath + 'qs_target.npy', qs_target)

# Initial conditions for both primal and adjoint are defined here as they only need to defined once.
q0 = wf.IC_primal()
q0_adj = wf.IC_adjoint()


#%% Optimal control
dL_du_list = []  # Collecting the gradient over the optimization steps
J_opt_FOM_list = []  # Collecting the FOM cost over the optimization steps
running_time = []  # Time calculated for each iteration in a running manner
J_opt_list = []  # Collecting the optimal cost functional for plotting
dL_du_ratio_list = []  # Collecting the ratio of gradients for plotting
err_list = []  # Offline error reached according to the tolerance
trunc_modes_list = []  # Number of modes needed to reach the offline error


# List of problem constants
kwargs = {
    'dx': wf.dx,
    'dy': wf.dy,
    'dt': wf.dt,
    'Nx': wf.Nxi,
    'Ny': wf.Neta,
    'Nt': wf.Nt,
    'n_c': n_c,
    'lamda': 1e-3,  # Regularization parameter
    'omega': 1,  # initial step size for gradient update
    'delta_conv': 1e-4,  # Convergence criteria
    'delta': 1e-2,  # Armijo constant
    'opt_iter': 2000,  # Total iterations
    'Armijo_iter': 20,  # Armijo iterations
    'omega_decr': 4,  # Decrease omega by a factor
    'beta': 1 / 2,  # Beta factor for two-way backtracking line search
    'verbose': True,  # Print options
    'simple_Armijo': False,  # Switch true for simple Armijo and False for two-way backtracking(Preferable option)
    'base_tol': 1e-2,  # Base tolerance for selecting number of modes (main variable for truncation)
    'omega_cutoff': 1e-10,  # Below this cutoff the Armijo and Backtracking should exit the update loop
    'threshold': False,  # Variable for selecting threshold based truncation or mode based. "TRUE" for threshold based
    # "FALSE" for mode based.
    'Nm': 100,  # Number of modes for truncation if threshold selected to False.
}

# For two-way backtracking line search
omega = 1

stag = False

start = time.time()
time_odeint_s = perf_counter()  # save running time
# %%
for opt_step in range(kwargs['opt_iter']):

    print("\n-------------------------------")
    print("Optimization step: %d" % opt_step)

    if opt_step == 0 or stag:
        time_odeint = perf_counter()  # save timing
        '''
        Forward calculation with primal FOM for basis computation and update
        '''
        qs = wf.TI_primal(q0, f, A_p, psi)

        # Compute the reduced basis
        V, qs_POD = compute_red_basis(qs, **kwargs)
        Nm = V.shape[1]
        err = np.linalg.norm(qs - qs_POD) / np.linalg.norm(qs)
        print(f"Relative error for primal: {err}, with Nm: {Nm}")

        err_list.append(err)
        trunc_modes_list.append(Nm)

        # Initial condition for dynamical simulation
        a_p = wf.IC_primal_PODG_FRTO(V, q0)
        a_a = wf.IC_adjoint_PODG_FRTO(V, q0_adj)

        # Construct the primal and adjoint system matrices for the POD-Galerkin approach
        Ar_p, psir_p = wf.mat_primal_PODG_FRTO(A_p, V, psi)
        Ar_a, Tarr_a = wf.mat_adjoint_PODG_FRTO(V, A_a, qs_target)

        time_odeint = perf_counter() - time_odeint
        if kwargs['verbose']: print("Basis refinement t_cpu = %1.3f" % time_odeint)

    '''
    Forward calculation with the reduced system
    '''
    time_odeint = perf_counter()  # save timing
    as_ = wf.TI_primal_PODG_FRTO(a_p, f, Ar_p, psir_p)
    time_odeint = perf_counter() - time_odeint
    if kwargs['verbose']: print("Forward t_cpu = %1.3f" % time_odeint)

    '''
    Objective and costs for control
    '''
    time_odeint = perf_counter()  # save timing
    J = Calc_Cost_PODG(V, as_, qs_target, f, **kwargs)
    time_odeint = perf_counter() - time_odeint
    if kwargs['verbose']: print("Calc_Cost t_cpu = %1.6f" % time_odeint)

    '''
    Backward calculation with reduced system
    '''
    time_odeint = perf_counter()  # save timing
    as_adj = wf.TI_adjoint_PODG_FRTO(a_a, as_, Ar_a, Tarr_a)
    time_odeint = perf_counter() - time_odeint
    if kwargs['verbose']: print("Backward t_cpu = %1.3f" % time_odeint)

    '''
     Update Control
    '''
    if kwargs['simple_Armijo']:
        time_odeint = perf_counter()
        f, J_opt, dL_du, stag = Update_Control_PODG_FRTO_adaptive(f, a_p, as_adj, qs_target, V, Ar_p, psir_p, psi, J,
                                                                  wf=wf, **kwargs)
        if kwargs['verbose']: print("Update Control t_cpu = %1.3f" % (perf_counter() - time_odeint))
    else:
        time_odeint = perf_counter()
        f, J_opt, dL_du, omega, stag = Update_Control_PODG_FRTO_adaptive_TWBT(f, a_p, as_adj, qs_target, V, Ar_p, psir_p,
                                                                              J, omega, wf=wf, **kwargs)
        if kwargs['verbose']: print("Update Control t_cpu = %1.3f" % (perf_counter() - time_odeint))


    running_time.append(perf_counter() - time_odeint_s)

    qs_opt_full = wf.TI_primal(q0, f, A_p, psi)
    JJ = Calc_Cost(qs_opt_full, qs_target, f, **kwargs)

    J_opt_FOM_list.append(JJ)
    J_opt_list.append(J_opt)
    dL_du_list.append(dL_du)
    dL_du_ratio_list.append(dL_du / dL_du_list[0])

    print(
        f"J_opt : {J_opt}, ||dL_du|| = {dL_du}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du / dL_du_list[0]}"
    )

    # Convergence criteria
    if opt_step == kwargs['opt_iter'] - 1:
        print("\n\n-------------------------------")
        print(
            f"WARNING... maximal number of steps reached, "
            f"J_opt : {J_opt}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du / dL_du_list[0]}"
        )
        break
    elif dL_du / dL_du_list[0] < kwargs['delta_conv']:
        print("\n\n-------------------------------")
        print(
            f"Optimization converged with, "
            f"J_opt : {J_opt}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du / dL_du_list[0]}"
        )
        break
    else:
        if opt_step == 0:
            if stag:
                print("\n\n-------------------------------")
                print(
                    f"Armijo Stagnated !!!!!! due to the step length being too low thus exiting at itr: {opt_step} with "
                    f"J_opt : {J_opt}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du / dL_du_list[0]}")
                break
        else:
            dJ = (J_opt_list[-1] - J_opt_list[-2]) / J_opt_list[0]
            if abs(dJ) == 0:
                print(f"WARNING: dJ has turned close to 0...")
                break
            if stag:
                print("\n-------------------------------")
                print(
                    f"Armijo Stagnated !!!!!! due to the step length being too low thus exiting at itr: {opt_step} with "
                    f"J_opt : {J_opt}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du / dL_du_list[0]}")
                # break


# Compute the final state
as__ = wf.TI_primal_PODG_FRTO(a_p, f, Ar_p, psir_p)
qs = V @ as__
qs_adj = V @ as_adj
f_opt = psi @ f

# Compute the cost with the optimal control
qs_opt_full = wf.TI_primal(q0, f, A_p, psi)
J = Calc_Cost(qs_opt_full, qs_target, f, **kwargs)
print("\n")
print(f"J with respect to the optimal control for FOM: {J}")


end = time.time()
print("\n")
print("Total time elapsed = %1.3f" % (end - start))

# %%
# Save the convergence lists
np.save(impath + 'J_opt_list.npy', J_opt_list)
np.save(impath + 'J_opt_FOM_list.npy', J_opt_FOM_list)
np.save(impath + 'dL_du_ratio_list.npy', dL_du_ratio_list)
np.save(impath + 'err_list.npy', err_list)
np.save(impath + 'trunc_modes_list.npy', trunc_modes_list)
np.save(impath + 'running_time.npy', running_time)

# Save the optimized solution
np.save(impath + 'qs_opt.npy', qs)
np.save(impath + 'qs_adj_opt.npy', qs_adj)
np.save(impath + 'f_opt.npy', f_opt)
np.save(impath + 'f_opt_low.npy', f)


# %%
# Load the results
qs_org = np.load(impath + 'qs_org.npy')
qs_opt = np.load(impath + 'qs_opt.npy')
qs_adj_opt = np.load(impath + 'qs_adj_opt.npy')
f_opt = np.load(impath + 'f_opt.npy')

# Plot the results
pf = PlotFlow(wf.X, wf.Y, wf.t)
if Dimension == "1D":
    pf.plot1D(qs_org, name="qs_org", immpath=immpath)
    pf.plot1D(qs_target, name="qs_target", immpath=immpath)
    pf.plot1D(qs_opt, name="qs_opt", immpath=immpath)
    pf.plot1D(qs_adj_opt, name="qs_adj_opt", immpath=immpath)
    pf.plot1D(f_opt, name="f_opt", immpath=immpath)

    pf.plot1D_ROM_converg(J_opt_list,
                          dL_du_ratio_list,
                          err_list,
                          trunc_modes_list,
                          immpath=immpath)
