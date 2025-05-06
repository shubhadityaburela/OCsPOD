import numpy as np
import scipy
from matplotlib import pyplot as plt
from numba import njit

from Helper_sPODG import make_V_W_delta, LHS_offline_primal_FOTR, RHS_offline_primal_FOTR, Control_offline_primal_FOTR, \
    Matrices_online_primal_FOTR, solve_lin_system, findIntervalAndGiveInterpolationWeight_1D, make_V_W_U_delta, \
    LHS_offline_primal_FRTO, RHS_offline_primal_FRTO, Control_offline_primal_FRTO, Matrices_online_primal_FRTO, \
    Matrices_online_adjoint_FRTO_expl, Matrices_online_adjoint_FRTO_impl, Target_offline_adjoint_FOTR, \
    Matrices_online_adjoint_FOTR_expl, solve_lin_system_Tikh_reg, make_V_W_delta_CubSpl
from Helper_sPODG_FRTO import E11, E12, E21, E22
from TI_schemes import rk4_sPODG_prim, rk4_sPODG_adj, implicit_midpoint_sPODG_adj, DIRK_sPODG_adj, bdf2_sPODG_adj, \
    rk4_sPODG_adj_


#############
## FOTR sPOD
#############

@njit
def IC_primal_sPODG_FOTR(q0, V):
    z = 0
    a = V.transpose() @ q0
    # Initialize the shifts with zero for online phase
    a = np.concatenate((a, np.asarray([z])))

    return a


def mat_primal_sPODG_FOTR(V_delta_primal, W_delta_primal, A_p, psi, samples, modes):

    # Construct LHS matrix
    LHS_matrix = LHS_offline_primal_FOTR(V_delta_primal, W_delta_primal, modes)

    # Construct RHS matrix
    RHS_matrix = RHS_offline_primal_FOTR(V_delta_primal, W_delta_primal, A_p, modes)

    # Construct the control matrix
    C_matrix = Control_offline_primal_FOTR(V_delta_primal, W_delta_primal, psi, samples, modes)

    return LHS_matrix, RHS_matrix, C_matrix


@njit
def RHS_primal_sPODG_FOTR(a, f, lhs, rhs, c, ds, modes):
    # Prepare the online primal matrices
    M, A, intervalIdx, weight = Matrices_online_primal_FOTR(lhs, rhs, c, f, a, ds, modes)

    # Solve the linear system of equations
    X = solve_lin_system(M, A)

    return X, intervalIdx, weight


def TI_primal_sPODG_FOTR(lhs, rhs, c, a, f0, delta_s, modes, Nt, dt):
    # Time loop
    as_ = np.zeros((a.shape[0], Nt), order="F")
    f0 = np.asfortranarray(f0)
    IntIds = np.zeros(Nt, dtype=np.int32)
    weights = np.zeros(Nt)

    as_[:, 0] = a

    for n in range(1, Nt):
        as_[:, n], _, IntIds[n - 1], weights[n - 1] = rk4_sPODG_prim(RHS_primal_sPODG_FOTR, as_[:, n - 1], f0[:, n - 1],
                                                                     f0[:, n], dt, lhs, rhs, c, delta_s, modes)

    IntIds[-1], weights[-1] = findIntervalAndGiveInterpolationWeight_1D(delta_s[2], -as_[-1, -1])

    return as_, IntIds, weights


@njit
def IC_adjoint_sPODG_FOTR(Nm_a, z):
    a = np.concatenate((np.zeros(Nm_a), np.asarray([z])))
    return a


def mat_adjoint_sPODG_FOTR(V_delta_adjoint, W_delta_adjoint, A_a, V_delta_primal, samples, modes_a, modes_p):

    # Construct LHS matrix
    LHS_matrix = LHS_offline_primal_FOTR(V_delta_adjoint, W_delta_adjoint, modes_a)

    # Construct RHS matrix
    RHS_matrix = RHS_offline_primal_FOTR(V_delta_adjoint, W_delta_adjoint, A_a, modes_a)

    # Construct the control matrix
    Tar_matrix = Target_offline_adjoint_FOTR(V_delta_primal, V_delta_adjoint, W_delta_adjoint,
                                             samples, modes_a, modes_p)

    return LHS_matrix, RHS_matrix, Tar_matrix


@njit
def RHS_adjoint_sPODG_FOTR_expl(as_adj, as_, qs_target, lhs, rhs, tar, Vda, Wda, modes_a, modes_p, delta_s, dx):
    # Prepare the online adjoint matrices
    M, A = Matrices_online_adjoint_FOTR_expl(lhs, rhs, tar, Vda, Wda, qs_target, as_adj, as_,
                                             modes_a, modes_p, delta_s, dx)

    # Solve the linear system of equations
    if np.linalg.cond(M) == np.inf:
        return solve_lin_system_Tikh_reg(M, A)
    else:
        return solve_lin_system(M, A)


def TI_adjoint_sPODG_FOTR(lhs, rhs, tar, Vda, Wda, a_a, as_, qs_target, modes_a, modes_p, delta_s, dx, Nt, dt, scheme):
    # Time loop
    as_adj = np.zeros((modes_a + 1, Nt), order="F")
    as_ = np.asfortranarray(as_)
    as_adj[:, -1] = a_a

    if scheme == "RK4":
        for n in range(1, Nt):
            as_adj[:, -(n + 1)] = rk4_sPODG_adj_(RHS_adjoint_sPODG_FOTR_expl, as_adj[:, -n],
                                                 as_[:, -n], as_[:, -(n + 1)],
                                                 qs_target[:, -n], qs_target[:, -(n + 1)], - dt, lhs, rhs, tar, Vda,
                                                 Wda, modes_a, modes_p, delta_s, dx)
    else:
        print('This is a nonlinear system of equation. It could be very hard and unnecessary to implement implicit'
              'methods for solving such an equation. Thus please choose RK4 as the preferred method......')
        exit()

    return as_adj


#############
## FRTO sPOD
#############

@njit
def IC_primal_sPODG_FRTO(q0, V):
    z = 0
    a = V.transpose() @ q0
    # Initialize the shifts with zero for online phase
    a = np.concatenate((a, np.asarray([z])))

    return a


def mat_primal_sPODG_FRTO(V_delta_primal, W_delta_primal, U_delta_primal, A_p, psi, samples, modes):

    # Construct LHS matrix
    LHS_matrix = LHS_offline_primal_FRTO(V_delta_primal, W_delta_primal, modes)

    # Construct RHS matrix
    RHS_matrix = RHS_offline_primal_FRTO(V_delta_primal, W_delta_primal, A_p, modes)

    # Construct the control matrix
    C_matrix = Control_offline_primal_FRTO(V_delta_primal, W_delta_primal, U_delta_primal, psi, samples, modes)

    return LHS_matrix, RHS_matrix, C_matrix


@njit
def RHS_primal_sPODG_FRTO(a, f, lhs, rhs, c, ds, modes):
    # Prepare the online primal matrices
    M, A, intervalIdx, weight = Matrices_online_primal_FRTO(lhs, rhs, c, f, a, ds, modes)

    X = solve_lin_system(M, A)

    return X, intervalIdx, weight


def TI_primal_sPODG_FRTO(lhs, rhs, c, a, f0, delta_s, modes, Nt, dt):
    # Time loop
    types_of_dots = 5  # derivatives to approximate
    as_ = np.zeros((a.shape[0], Nt), order="F")
    as_dot = np.zeros((types_of_dots, a.shape[0], Nt), order="F")
    f0 = np.asfortranarray(f0)
    IntIds = np.zeros(Nt, dtype=np.int32)
    weights = np.zeros(Nt)

    as_[:, 0] = a
    for n in range(1, Nt):
        as_[:, n], as_dot[..., n], IntIds[n - 1], weights[n - 1] = rk4_sPODG_prim(RHS_primal_sPODG_FRTO,
                                                                                  as_[:, n - 1],
                                                                                  f0[:, n - 1],
                                                                                  f0[:, n], dt, lhs, rhs, c,
                                                                                  delta_s,
                                                                                  modes)

    IntIds[-1], weights[-1] = findIntervalAndGiveInterpolationWeight_1D(delta_s[2], -as_[-1, -1])
    as_dot[..., 0] = as_dot[..., 1].copy()

    return as_, as_dot, IntIds, weights


@njit
def IC_adjoint_sPODG_FRTO(modes):
    z = 0
    # Initialize the shifts with zero for online phase
    a = np.concatenate((np.zeros(modes), np.asarray([z])))

    return a


@njit
def RHS_adjoint_sPODG_FRTO_expl(a, f, a_, qs_target, a_dot, M1, M2, N, A1, A2, C, Vdp, Wdp, modes, delta_s, dx):
    # Prepare the online primal matrices
    M, A = Matrices_online_adjoint_FRTO_expl(M1, M2, N, A1, A2, C, Vdp, Wdp, f, a, a_, qs_target, a_dot,
                                             modes, delta_s, dx)
    # Solve the linear system of equations
    X = solve_lin_system(M, -A)

    return X


def RHS_adjoint_sPODG_FRTO_impl(a, f, a_, qs_target, a_dot, dt, M1, M2, N, A1, A2, C, Vdp, Wdp, modes, delta_s, dx,
                                scheme):
    # Solve the linear system of equations
    M, A, T = Matrices_online_adjoint_FRTO_impl(M1, M2, N, A1, A2, C, Vdp, Wdp, f, a, a_, qs_target, a_dot,
                                                modes, delta_s, dx)
    if scheme == "implicit_midpoint":
        M_f = M + dt / 2 * A
        A_f = (M - dt / 2 * A) @ a - dt * T
        return solve_lin_system(M_f, A_f)
    elif scheme == "DIRK":
        M_f = M + dt / 4 * A
        A_f = - A @ a - T
        return solve_lin_system(M_f, A_f)
    elif scheme == "BDF2":
        M_f = 3.0 * M + 2 * dt * A
        A_f = 4.0 * M @ a[1] - 1.0 * M @ a[0] - 2 * dt * T
        return solve_lin_system(M_f, A_f)


def TI_adjoint_sPODG_FRTO(at_adj, f0, a_, qs_target, a_dot, lhsp, rhsp, C, Vdp, Wdp, modes, delta_s, Nt, dt, dx,
                          scheme):
    as_adj = np.zeros((at_adj.shape[0], Nt), order="F")
    as_adj[:, -1] = at_adj

    M1 = lhsp[0]
    N = lhsp[1]
    M2 = lhsp[2]
    A1 = rhsp[0]
    A2 = rhsp[1]

    # #################################################################################################################
    # M = np.empty((modes + 1, modes + 1), dtype=M1.dtype)
    # A = np.empty((modes + 1, modes + 1), dtype=M1.dtype)
    # as_p = a_[:-1, 0]  # Take the modes from the primal solution
    # as_p[1:] = 0.0
    # z_p = a_[-1, 0]  # Take the shifts from the primal solution
    # as_dot = a_dot[0, :-1, 0]  # Take the modes derivative from the primal
    # as_dot[:] = 0.0
    # z_dot = a_dot[0, -1:, -1]  # Take the shift derivative from the primal
    # Da = as_p.reshape(-1, 1)
    #
    # intId, weight = findIntervalAndGiveInterpolationWeight_1D(delta_s[2], -z_p)
    #
    # WTB = np.add(weight * C[1, intId], (1 - weight) * C[1, intId + 1])  # WTB and VTdashB are exactly the same quantity
    # WTdashB = np.add(weight * C[2, intId], (1 - weight) * C[2, intId + 1])
    #
    # # Assemble the mass matrix M
    # M[:modes, :modes] = M1.T
    # M[:modes, modes:] = N @ Da
    # M[modes:, :modes] = M[:modes, modes:].T
    # M[modes:, modes:] = Da.T @ (M2.T @ Da)
    #
    # # Assemble the A matrix
    # A[:modes, :modes] = E11(N, A1, z_dot, modes).T
    # A[:modes, modes:] = E12(M2, N, A2, Da, WTB, as_dot, z_dot, as_p, f0[:, -1], modes).T
    # A[modes:, :modes] = E21(N, WTB, as_dot, f0[:, -1]).T
    # A[modes:, modes:] = E22(M2, Da, WTdashB, as_dot, f0[:, -1]).T
    #
    # # Solve the linear system of equations
    # eigvals, eigvecs = scipy.linalg.eig(-A, M)
    # print("Eigenvalues:", eigvals)
    # # Check stability based on the real parts of eigenvalues
    # tol = 1e-10
    # if np.all(np.real(eigvals) < 0):
    #     print(
    #         "The system is asymptotically stable (all eigenvalues have negative real parts within tolerance).")
    # elif np.any(np.real(eigvals) > tol):
    #     print("The system is unstable (at least one eigenvalue has a positive real part beyond tolerance).")
    # else:
    #     print("The system is marginally stable (eigenvalues are near the imaginary axis within tolerance).")
    #
    # exit()
    # #################################################################################################################

    if scheme == "RK4":
        for n in range(1, Nt):
            as_adj[:, -(n + 1)] = rk4_sPODG_adj(RHS_adjoint_sPODG_FRTO_expl, as_adj[:, -n], f0[:, -n],
                                                f0[:, -(n + 1)],
                                                a_[:, -n], a_[:, -(n + 1)], qs_target[:, -n],
                                                qs_target[:, -(n + 1)],
                                                a_dot[..., -n], - dt,
                                                M1, M2, N, A1, A2, C,
                                                Vdp, Wdp, modes, delta_s, dx)

    if scheme == "implicit_midpoint":
        for n in range(1, Nt):
            as_adj[:, -(n + 1)] = implicit_midpoint_sPODG_adj(RHS_adjoint_sPODG_FRTO_impl, as_adj[:, -n], f0[:, -n],
                                                              f0[:, -(n + 1)],
                                                              a_[:, -n], a_[:, -(n + 1)], qs_target[:, -n],
                                                              qs_target[:, -(n + 1)],
                                                              a_dot[..., -n], - dt,
                                                              M1, M2, N, A1, A2, C,
                                                              Vdp, Wdp, modes, delta_s, dx, scheme)
    elif scheme == "DIRK":
        for n in range(1, Nt):
            as_adj[:, -(n + 1)] = DIRK_sPODG_adj(RHS_adjoint_sPODG_FRTO_impl, as_adj[:, -n], f0[:, -n],
                                                 f0[:, -(n + 1)],
                                                 a_[:, -n], a_[:, -(n + 1)], qs_target[:, -n],
                                                 qs_target[:, -(n + 1)],
                                                 a_dot[..., -n], - dt,
                                                 M1, M2, N, A1, A2, C,
                                                 Vdp, Wdp, modes, delta_s, dx, scheme)
    elif scheme == "BDF2":
        # last 2 steps (x_{n-1}, x_{n-2}) with RK4 (Effectively 2nd order)
        for n in range(1, 2):
            as_adj[:, -(n + 1)] = rk4_sPODG_adj(RHS_adjoint_sPODG_FRTO_expl, as_adj[:, -n], f0[:, -n],
                                                f0[:, -(n + 1)],
                                                a_[:, -n], a_[:, -(n + 1)], qs_target[:, -n],
                                                qs_target[:, -(n + 1)],
                                                a_dot[..., -n], - dt,
                                                M1, M2, N, A1, A2, C,
                                                Vdp, Wdp, modes, delta_s, dx)
        for n in range(2, Nt):
            as_adj[:, -(n + 1)] = bdf2_sPODG_adj(RHS_adjoint_sPODG_FRTO_impl, as_adj,
                                                 f0[:, -(n + 1)],
                                                 a_[:, -(n + 1)],
                                                 qs_target[:, -(n + 1)],
                                                 a_dot[..., -(n + 1)], - dt,
                                                 M1, M2, N, A1, A2, C,
                                                 Vdp, Wdp, modes, delta_s, dx, scheme, n)

    return as_adj

# # Set up the figure and axis
# fig, ax = plt.subplots(figsize=(12, 6))
# scat = ax.scatter([], [], c='red', s=50)
# ax.set_xlim(-1e5, 1e5)
# ax.set_ylim(-4, 4)
# ax.set_xlabel('Real Part')
# ax.set_ylabel('Imaginary Part')
# ax.set_title('Eigenvalues Animation')
# # Initialization function for the animation
# def init():
#     scat.set_offsets(np.empty((0, 2)))
#     return scat,
# # Update function for each frame in the animation
# def update(frame):
#     # Extract the eigenvalues for the current time step
#     eigs = eigv[frame, :]
#     # Create a (num_eigvals, 2) array with real and imaginary parts
#     points = np.column_stack((eigs.real, eigs.imag))
#     scat.set_offsets(points)
#     # Dynamically adjust the axes to the data
#     ax.relim()  # Recompute the data limits
#     ax.autoscale_view()  # Update the view limits
#     ax.set_title(f'Time step: {frame + 1}/{self.Nt}')
#     ax.grid()
#     return scat,
# # Create the animation
# ani = animation.FuncAnimation(
#     fig, update, frames=self.Nt, init_func=init, interval=20, blit=True
# )
# plt.show()
# ani.save('eigenvalues_animation.mp4', writer='ffmpeg')
