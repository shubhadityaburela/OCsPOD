from Coefficient_Matrix import CoefficientMatrix
from Costs import Calc_Cost
from Helper import ControlSelectionMatrix_advection
from Update import Update_Control_TWBT
from advection import advection
from Plots import PlotFlow
import numpy as np
import os
from time import perf_counter
import time
import scipy.sparse as sp
import argparse

parser = argparse.ArgumentParser(description="Input the variables for running the script.")
parser.add_argument("problem", type=int, choices=[1, 2, 3], help="Specify the problem number (1, 2, or 3)")
args = parser.parse_args()
problem = args.problem

print("\n")
print(f"Solving problem: {args.problem}")

Nxi = 3200
Neta = 1
Nt = 3360

impath = "./data_final/FOM/problem=" + str(problem) + "/"  # for data
immpath = "./plots_final/FOM/problem=" + str(problem) + "/"  # for plots
os.makedirs(impath, exist_ok=True)

# Wildfire solver initialization along with grid initialization
# Thick wave params:                  # Sharp wave params (earlier kink):             # Sharp wave params (later kink):
# cfl = 8 / 6                         # cfl = 8 / 6                                   # cfl = 8 / 6
# tilt_from = 3 * Nt // 4             # tilt_from = 3 * Nt // 4                       # tilt_from = 9 * Nt / 10
# v_x = 0.5                           # v_x = 0.55                                    # v_x = 0.6
# v_x_t = 1.0                         # v_x_t = 1.0                                   # v_x_t = 1.3
# variance = 7                        # variance = 0.5                                # variance = 0.5
# offset = 12                         # offset = 30                                   # offset = 30


if problem == 1:    # Thick wave params
    wf = advection(Nxi=Nxi, Neta=Neta, timesteps=Nt, cfl=8 / 6,
                   tilt_from=3 * Nt // 4, v_x=0.5, v_x_t=1.0, variance=7, offset=12)
elif problem == 2:    # Sharp wave params (earlier kink):
    wf = advection(Nxi=Nxi, Neta=Neta, timesteps=Nt, cfl=8 / 6,
                   tilt_from=3 * Nt // 4, v_x=0.55, v_x_t=1.0, variance=0.5, offset=30)
elif problem == 3:    # Sharp wave params (later kink):
    wf = advection(Nxi=Nxi, Neta=Neta, timesteps=Nt, cfl=8 / 6,
                   tilt_from=9 * Nt // 10, v_x=0.6, v_x_t=1.3, variance=0.5, offset=30)
else:  # Default is problem 2
    wf = advection(Nxi=Nxi, Neta=Neta, timesteps=Nt, cfl=8 / 6,
                   tilt_from=3 * Nt // 4, v_x=0.55, v_x_t=1.0, variance=0.5, offset=30)
wf.Grid()

# %%
n_c_init = 40  # Number of initial controls

# Selection matrix for the control input
psi = ControlSelectionMatrix_advection(wf, n_c_init, Gaussian=True, gaussian_mask_sigma=0.5)  # Changing the value of
# trim_first_n should basically make the psi matrix and the number of controls to be user defined.
n_c = psi.shape[1]
f = np.zeros((n_c, wf.Nt))  # Initial guess for the control


#%% Assemble the linear operators
Mat = CoefficientMatrix(orderDerivative=wf.firstderivativeOrder, Nxi=wf.Nxi,
                        Neta=wf.Neta, periodicity='Periodic', dx=wf.dx, dy=wf.dy)
# Convection matrix (Needs to be changed if the velocity is time dependent)
A_p = - (wf.v_x[0] * Mat.Grad_Xi_kron + wf.v_y[0] * Mat.Grad_Eta_kron)
A_a = A_p.transpose()

# Grid dependent matrix for Adjoint equation correction
diagonal = np.ones(wf.Nxi) * np.sqrt(wf.dx)
diagonal[0] /= np.sqrt(2)
diagonal[-1] /= np.sqrt(2)
C = sp.diags(diagonal, format='csc')
CTC = C.T @ C

#%% Solve the uncontrolled system
qs_org = wf.TI_primal(wf.IC_primal(), f, A_p, psi)
np.save(impath + 'qs_org.npy', qs_org)

qs_target = wf.TI_primal_target(wf.IC_primal(), Mat, np.zeros((wf.Nxi * wf.Neta, wf.Nt)))
np.save(impath + 'qs_target.npy', qs_target)

# Initial conditions for both primal and adjoint are defined here as they only need to defined once.
q0 = wf.IC_primal()
q0_adj = wf.IC_adjoint()

#%% Optimal control
dL_du_norm_list = []  # Collecting the gradient over the optimization steps
J_opt_list = []  # Collecting the optimal cost functional for plotting
J_opt_FOM_list = []  # Collecting the FOM cost over the optimization steps
running_time = []  # Time calculated for each iteration in a running manner
dL_du_norm_ratio_list = []  # Collecting the ratio of gradients for plotting

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
    'omega': 1,   # initial step size for gradient update
    'delta_conv': 1e-4,  # Convergence criteria
    'delta': 1e-2,  # Armijo constant
    'opt_iter': 100000,  # Total iterations
    'beta': 1 / 2,  # Beta factor for two-way backtracking line search
    'verbose': True,  # Print options
    'omega_cutoff': 1e-10  # Below this cutoff the Armijo and Backtracking should exit the update loop
}

# For two-way backtracking line search
omega = 1
stag = False

start = time.time()
time_odeint_s = perf_counter()  # save running time
#%%
for opt_step in range(kwargs['opt_iter']):
    '''
    Forward calculation
    '''
    print("\n-------------------------------")
    print("Optimization step: %d" % opt_step)

    qs = wf.TI_primal(q0, f, A_p, psi)

    '''
    Objective and costs for control
    '''
    J = Calc_Cost(qs, qs_target, f, kwargs['dx'], kwargs['dt'], kwargs['lamda'])

    '''
    Adjoint calculation
    '''
    qs_adj = wf.TI_adjoint(q0_adj, qs, qs_target, A_a, CTC)

    '''
     Update Control
    '''
    f, J_opt, _, dL_du_norm, omega, stag = Update_Control_TWBT(f, q0, qs_adj, qs_target, psi, A_p, J, omega, wf=wf, **kwargs)

    running_time.append(perf_counter() - time_odeint_s)

    qs_opt_full = wf.TI_primal(q0, f, A_p, psi)
    JJ = Calc_Cost(qs_opt_full, qs_target, f,
                   kwargs['dx'], kwargs['dt'], kwargs['lamda'])

    # Save for plotting
    J_opt_list.append(J_opt)
    J_opt_FOM_list.append(JJ)
    dL_du_norm_list.append(dL_du_norm)
    dL_du_norm_ratio_list.append(dL_du_norm / dL_du_norm_list[0])

    print(
        f"J_opt : {J_opt}, ||dL_du|| = {dL_du_norm}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}"
    )

    # Convergence criteria
    if opt_step == kwargs['opt_iter'] - 1:
        print("\n\n-------------------------------")
        print(
            f"WARNING... maximal number of steps reached, "
            f"J_opt : {J_opt}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}"
        )
        break
    elif dL_du_norm / dL_du_norm_list[0] < kwargs['delta_conv']:
        print("\n\n-------------------------------")
        print(
            f"Optimization converged with, "
            f"J_opt : {J_opt}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}"
        )
        break
    else:
        if opt_step == 0:
            if stag:
                print("\n\n-------------------------------")
                print(f"Armijo Stagnated !!!!!! due to the step length being too low thus exiting at {opt_step} with "
                      f"J_opt : {J_opt}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}")
                break
        else:
            dJ = (J_opt_list[-1] - J_opt_list[-2]) / J_opt_list[0]
            if abs(dJ) == 0:
                print(f"WARNING: dJ has turned close to 0...")
                break
            if stag:
                print("\n-------------------------------")
                print(f"Armijo Stagnated !!!!!! due to the step length being too low thus exiting at {opt_step} with "
                      f"J_opt : {J_opt}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}")
                break

# Final state corresponding to the optimal control f
qs_opt = wf.TI_primal(q0, f, A_p, psi)
f_opt = psi @ f


# Compute the cost with the optimal control
J = Calc_Cost(qs_opt, qs_target, f, kwargs['dx'], kwargs['dt'], kwargs['lamda'])
print("\n")
print(f"J with respect to the optimal control for FOM: {J}")


end = time.time()
print("\n")
print("Total time elapsed = %1.3f" % (end - start))
#%%
# Save the convergence lists
np.save(impath + 'J_opt_list.npy', J_opt_list)
np.save(impath + 'running_time.npy', running_time)

# Save the optimized solution
np.save(impath + 'qs_opt.npy', qs_opt)
np.save(impath + 'qs_adj_opt.npy', qs_adj)
np.save(impath + 'f_opt.npy', f_opt)
# np.save(impath + 'f_opt_low.npy', f)


#%%
# Plot the results
pf = PlotFlow(wf.X, wf.Y, wf.t)

pf.plot1D(qs_org, name="qs_org", immpath=immpath)
pf.plot1D(qs_target, name="qs_target", immpath=immpath)
pf.plot1D(qs_opt, name="qs_opt", immpath=immpath)
pf.plot1D(qs_adj, name="qs_adj_opt", immpath=immpath)
pf.plot1D(f_opt, name="f_opt", immpath=immpath)
pf.plot1D_FOM_converg(J_opt_list, immpath=immpath)