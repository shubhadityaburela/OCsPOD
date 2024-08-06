from Coefficient_Matrix import CoefficientMatrix
from Costs import Calc_Cost
from Helper import ControlSelectionMatrix_advection
from Update import Update_Control
from advection import advection
from Plots import PlotFlow
import numpy as np
import os
from time import perf_counter
import time


impath = "./data/FOM/"   # Storing data
immpath = "./plots/FOM/"  # Storing plots
os.makedirs(impath, exist_ok=True)

# Problem variables
Dimension = "1D"
Nxi = 800
Neta = 1
Nt = 1400

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

A_p = A_p.todense()
A_a = A_a.todense()

#%% Solve the uncontrolled system
qs_org = wf.TI_primal(wf.IC_primal(), f, A_p, psi)
np.save(impath + 'qs_org.npy', qs_org)

qs_target = wf.TI_primal_target(wf.IC_primal(), Mat, np.zeros((wf.Nxi * wf.Neta, wf.Nt)))
np.save(impath + 'qs_target.npy', qs_target)

# Initial conditions for both primal and adjoint are defined here as they only need to defined once.
q0 = wf.IC_primal()
q0_adj = wf.IC_adjoint()

#%% Optimal control
J_list = []  # Collecting cost functional over the optimization steps
dL_du_list = []  # Collecting the gradient over the optimization steps
J_opt_list = []  # Collecting the optimal cost functional for plotting
dL_du_ratio_list = []  # Collecting the ratio of gradients for plotting

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
    'opt_iter': 10,  # Total iterations
    'Armijo_iter': 20,  # Armijo iterations
    'verbose': True  # Print options
}


state_time = []
cost_time = []
adjoint_time = []
update_time = []

start = time.time()
#%%
for opt_step in range(kwargs['opt_iter']):
    '''
    Forward calculation
    '''
    print("\n-------------------------------")
    print("Optimization step: %d" % opt_step)

    time_odeint = perf_counter()  # save timing
    qs = wf.TI_primal(q0, f, A_p, psi)
    time_odeint = perf_counter() - time_odeint
    if kwargs['verbose']: print("Forward t_cpu = %1.3f" % time_odeint)

    state_time.append(time_odeint)

    '''
    Objective and costs for control
    '''
    time_odeint = perf_counter()  # save timing
    J = Calc_Cost(qs, qs_target, f, **kwargs)
    time_odeint = perf_counter() - time_odeint
    if kwargs['verbose']: print("Calc_Cost t_cpu = %1.6f" % time_odeint)
    if opt_step == 0:
        pass
    else:
        dJ = (J - J_list[-1]) / J_list[0]
        if abs(dJ) == 0:
            print("WARNING: dJ has turned 0...")
            break
    J_list.append(J)

    cost_time.append(time_odeint)

    '''
    Adjoint calculation
    '''
    time_odeint = perf_counter()  # save timing
    qs_adj = wf.TI_adjoint(q0_adj, f, qs, qs_target, A_a)
    time_odeint = perf_counter() - time_odeint
    if kwargs['verbose']: print("Backward t_cpu = %1.3f" % time_odeint)

    adjoint_time.append(time_odeint)


    '''
     Update Control
    '''
    time_odeint = perf_counter() - time_odeint
    f, J_opt, dL_du = Update_Control(f, q0, qs_adj, qs_target, psi, A_p, J, wf=wf, **kwargs)
    if kwargs['verbose']: print("Update Control t_cpu = %1.3f" % (perf_counter() - time_odeint))

    update_time.append(perf_counter() - time_odeint)

    # Save for plotting
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

# Final state corresponding to the optimal control f
qs_opt = wf.TI_primal(q0, f, A_p, psi)
f_opt = psi @ f



# Compute the cost with the optimal control
J = Calc_Cost(qs_opt, qs_target, f, **kwargs)
print("\n")
print(f"J with respect to the optimal control for FOM: {J}")



end = time.time()
print("\n")
print("Total time elapsed = %1.3f" % (end - start))
#%%
# Save the convergence lists
np.save(impath + 'J_opt_list.npy', J_opt_list)
np.save(impath + 'dL_du_ratio_list.npy', dL_du_ratio_list)

# Save the optimized solution
np.save(impath + 'qs_opt.npy', qs_opt)
np.save(impath + 'qs_adj_opt.npy', qs_adj)
np.save(impath + 'f_opt.npy', f_opt)


#%%
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

    pf.plot1D_FOM_converg(J_opt_list, dL_du_ratio_list, immpath=immpath)




print(sum(state_time) / kwargs['opt_iter'])
print(sum(cost_time) / kwargs['opt_iter'])
print(sum(adjoint_time) / kwargs['opt_iter'])
print(sum(update_time) / kwargs['opt_iter'])