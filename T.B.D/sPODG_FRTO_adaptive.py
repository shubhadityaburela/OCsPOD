"""
This file is the adaptive version. It can handle all the three scenarios.
1. Fixed tolerance
2. Adjustable tolerance
3. Advanced adjustable tolerance
"""



from Coefficient_Matrix import CoefficientMatrix
from Update import Update_Control_sPODG_FRTO
from advection import advection
from Plots import PlotFlow
from Helper import ControlSelectionMatrix_advection, compute_red_basis, calc_shift
from Helper_sPODG import subsample, findIntervals, get_T, central_FDMatrix
from Costs import Calc_Cost_sPODG, Calc_Cost
import os
from time import perf_counter
import numpy as np
import time

impath = "./data/sPODG/FRTO/adaptive/Fixtol=1e-2/"  # for data
immpath = "./plots/sPODG/FRTO/adaptive/Fixtol=1e-2/"  # for plots
os.makedirs(impath, exist_ok=True)

# Problem variables
Dimension = "1D"
Nxi = 400
Neta = 1
Nt = 700

# solver initialization along with grid initialization
wf = advection(Nxi=Nxi, Neta=Neta if Dimension == "1D" else Nxi, timesteps=Nt, cfl=0.8, tilt_from=3 * Nt // 4)
wf.Grid()

# %%
n_c = 400  # Number of controls
f = np.zeros((n_c, wf.Nt))  # Initial guess for the control

# Selection matrix for the control input
psi = ControlSelectionMatrix_advection(wf, n_c)

# %% Assemble the linear operators
Mat = CoefficientMatrix(orderDerivative=wf.firstderivativeOrder, Nxi=wf.Nxi,
                        Neta=wf.Neta, periodicity='Periodic', dx=wf.dx, dy=wf.dy)
# Convection matrix (Needs to be changed if the velocity is time dependent)
A_p = - (wf.v_x[0] * Mat.Grad_Xi_kron + wf.v_y[0] * Mat.Grad_Eta_kron)
A_a = A_p.transpose()

# %% Solve the uncontrolled system
qs_org = wf.TI_primal(wf.IC_primal(), f, A_p, psi)
np.save(impath + 'qs_org.npy', qs_org)

qs_target = wf.TI_primal_target(wf.IC_primal(), Mat, np.zeros((wf.Nxi * wf.Neta, wf.Nt)))
np.save(impath + 'qs_target.npy', qs_target)

# Initial conditions for both primal and adjoint are defined here as they only need to defined once.
q0 = wf.IC_primal()
q0_adj = wf.IC_adjoint()

# %%
dL_du_list = []  # Collecting the gradient over the optimization steps
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
    'delta_conv': 7e-4,  # Convergence criteria
    'delta': 1e-2,  # Armijo constant
    'opt_iter': 50,  # Total iterations
    'Armijo_iter': 20,  # Armijo iterations
    'shift_sample': 200,  # Number of samples for shift interpolation
    'verbose': True,  # Print options
    'base_tol': 1e-2,  # Base tolerance for selecting number of modes for the adaptivity
    'tol': 1e-2,  # Initial tolerance set for adaptivity
    'stag': False,  # Stagnation variable set as False (no need to change)
    'Adtol': False,  # Adjustable tolerance is activated if True
    'AdAdtol': False,  # Advanced adjustable tolerance is set if True.
    # (The Adtol and AdAdtol should not be used together)
}

# %% ROM Variables

D = central_FDMatrix(order=6, Nx=wf.Nxi, dx=wf.dx)

# Generate the shift samples
delta_s = subsample(wf.X, num_sample=kwargs['shift_sample'])

# Extract transformation operators based on sub-sampled delta
T_delta, _ = get_T(delta_s, wf.X, wf.t)

# Fix the shifts upfront
delta_init = calc_shift(qs_org, q0, wf.X, wf.t)
_, T = get_T(delta_init, wf.X, wf.t)

stag_cntr = 0

start = time.time()
# %%
for opt_step in range(kwargs['opt_iter']):
    Nm = 0
    err = 1e12

    print("\n-------------------------------")
    print("Optimization step: %d" % opt_step)

    time_odeint = perf_counter()  # save timing
    '''
    Forward calculation with primal FOM for basis update
    '''
    qs = wf.TI_primal(q0, f, A_p, psi)

    qs_s = T.reverse(qs)
    if kwargs['Adtol']:  # Adjustable tolerance
        if kwargs['stag']:
            kwargs['tol'] = 1 * kwargs['tol'] / 2
        else:
            kwargs['tol'] = kwargs['base_tol']
        print(f"Base tolerance: {kwargs['base_tol']}")
        print(f"Running tolerance: {kwargs['tol']}")
    elif kwargs['AdAdtol']:  # Advanced adjustable tolerance
        if kwargs['stag']:
            if stag_cntr % 100 == 0 and stag_cntr != 0:
                kwargs['base_tol'] = 3 * kwargs['base_tol'] / 4
                print(f"NOTE:......Base tolerance decreased to: {kwargs['base_tol']}")
                kwargs['tol'] = kwargs['base_tol']
            else:
                kwargs['tol'] = 3 * kwargs['tol'] / 4
        else:
            kwargs['tol'] = kwargs['base_tol']
        print(f"Base tolerance: {kwargs['base_tol']}")
        print(f"Running tolerance: {kwargs['tol']}")

    # Fixed tolerance (default)
    while err > kwargs['tol']:
        Nm = Nm + 1
        V, qs_s_POD = compute_red_basis(qs_s, Nm)
        err = np.linalg.norm(qs_s - qs_s_POD) / np.linalg.norm(qs_s)
        print(f"Relative error for shifted primal: {err}, with Nm: {Nm}")


    err_list.append(err)
    trunc_modes_list.append(Nm)

    # Construct the primal system matrices for the sPOD-Galerkin approach
    Vd_p, Wd_p, lhs_p, rhs_p, c_p, T_p = wf.mat_primal_sPODG_FRTO(T_delta, V, A_p, psi, D,
                                                                  samples=kwargs['shift_sample'])

    # Initial condition for dynamical simulation
    a_p = wf.IC_primal_sPODG_FRTO(q0, delta_s, Vd_p)
    a_a = wf.IC_adjoint_sPODG_FRTO(q0_adj, delta_s, Vd_p)

    time_odeint = perf_counter() - time_odeint
    if kwargs['verbose']: print("Basis refinement t_cpu = %1.3f" % time_odeint)

    '''
    Forward calculation with the reduced system
    '''
    time_odeint = perf_counter()  # save timing
    as_, as_dot = wf.TI_primal_sPODG_FRTO(lhs_p, rhs_p, c_p, a_p, f, delta_s)
    time_odeint = perf_counter() - time_odeint
    if kwargs['verbose']: print("Forward t_cpu = %1.3f" % time_odeint)

    '''
    Objective and costs for control
    '''
    # Compute the interpolation weight and the interval in which the shift lies corresponding to which we compute the
    # V_delta and W_delta matrices
    time_odeint = perf_counter()  # save timing
    intIds, weights = findIntervals(delta_s, as_[-1, :])
    J = Calc_Cost_sPODG(Vd_p, as_, qs_target, f, intIds, weights, **kwargs)
    time_odeint = perf_counter() - time_odeint
    if kwargs['verbose']: print("Calc_Cost t_cpu = %1.6f" % time_odeint)

    '''
    Backward calculation with the reduced system
    '''
    time_odeint = perf_counter()  # save timing
    as_adj = wf.TI_adjoint_sPODG_FRTO(a_a, f, as_, lhs_p, rhs_p, c_p, T_p, Vd_p, Wd_p, qs_target, D, A_p, psi, as_dot,
                                      delta_s)
    time_odeint = perf_counter() - time_odeint
    if kwargs['verbose']: print("Backward t_cpu = %1.3f" % time_odeint)

    '''
     Update Control
    '''
    time_odeint = perf_counter() - time_odeint
    f, J_opt, dL_du, stag = Update_Control_sPODG_FRTO(f, lhs_p, rhs_p, c_p, Vd_p, a_p, as_, as_adj, qs_target,
                                                      delta_s, J, intIds, weights, wf=wf, **kwargs)
    if kwargs['verbose']: print("Update Control t_cpu = %1.3f" % (perf_counter() - time_odeint))

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
            pass
        else:
            dJ = (J_opt_list[-1] - J_opt_list[-2]) / J_opt_list[0]
            if abs(dJ) == 0:
                print(f"WARNING: dJ has turned close to 0...")
                break
            if stag:
                stag_cntr = stag_cntr + 1
                if stag_cntr > 100:
                    print(f"WARNING: Armijo stagnated !!!!")
                    break

# Compute the final state
as__, _ = wf.TI_primal_sPODG_FRTO(lhs_p, rhs_p, c_p, a_p, f, delta_s)
as_online = as__[:Nm]
delta_online = as__[-1]
qs = np.zeros_like(qs_target)

as_adj_online = as_adj[:Nm]
qs_adj = np.zeros_like(qs_target)
intIds, weights = findIntervals(delta_s, delta_online)
for i in range(f.shape[1]):
    V_delta = weights[i] * Vd_p[intIds[i]] + (1 - weights[i]) * Vd_p[intIds[i] + 1]
    qs[:, i] = V_delta @ as_online[:, i]
    qs_adj[:, i] = V_delta @ as_adj_online[:, i]

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
np.save(impath + 'dL_du_ratio_list.npy', dL_du_ratio_list)
np.save(impath + 'err_list.npy', err_list)
np.save(impath + 'trunc_modes_list.npy', trunc_modes_list)


# Save the optimized solution
np.save(impath + 'qs_opt.npy', qs)
np.save(impath + 'qs_adj_opt.npy', qs_adj)
np.save(impath + 'f_opt.npy', f_opt)

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
