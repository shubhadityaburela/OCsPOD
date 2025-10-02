import numpy as np
import scipy
from matplotlib import pyplot as plt
from numba import njit

from Helper_sPODG import make_V_W_delta, LHS_offline_primal_FOTR, RHS_offline_primal_FOTR, Control_offline_primal_FOTR, \
    Matrices_online_primal_FOTR, solve_lin_system, findIntervalAndGiveInterpolationWeight_1D, make_V_W_U_delta, \
    LHS_offline_primal_FRTO, RHS_offline_primal_FRTO, Control_offline_primal_FRTO, Matrices_online_primal_FRTO, \
    Matrices_online_adjoint_FRTO_expl, Matrices_online_adjoint_FRTO_impl, Target_offline_adjoint_FOTR, \
    solve_lin_system_Tikh_reg, Target_online_adjoint_FRTO, \
    LHS_offline_primal_FOTR_kdv, \
    RHS_offline_primal_FOTR_kdv, DEIM_primal_FOTR_kdv, DEIM_adjoint_FOTR_kdv, Matrices_online_primal_FOTR_kdv_expl, \
    Matrices_online_adjoint_FOTR_kdv_expl, LHS_offline_primal_FRTO_kdv, RHS_offline_primal_FRTO_kdv, \
    DEIM_primal_FRTO_kdv, Matrices_online_primal_FRTO_kdv_expl, Matrices_online_adjoint_FOTR_expl
from Helper_sPODG_FRTO import E11, E12, E21, E22, E11_kdvb, E12_kdvb, E21_kdvb, E22_kdvb, C1, C2
from TI_schemes import rk4_sPODG_prim, rk4_sPODG_adj, implicit_midpoint_sPODG_adj, DIRK_sPODG_adj, bdf2_sPODG_adj, \
    rk4_sPODG_adj_, rk4_sPODG_prim_kdvb, implicit_midpoint_sPODG_FRTO_primal_kdvb, \
    implicit_midpoint_sPODG_FRTO_adjoint_kdvb, rk4_sPODG_adj_kdvb


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


def mat_adjoint_sPODG_FOTR(V_delta_adjoint, W_delta_adjoint, A_a, V_delta_primal, samples, modes_a, modes_p, CTC):
    # Construct LHS matrix
    LHS_matrix = LHS_offline_primal_FOTR(V_delta_adjoint, W_delta_adjoint, modes_a)

    # Construct RHS matrix
    RHS_matrix = RHS_offline_primal_FOTR(V_delta_adjoint, W_delta_adjoint, A_a, modes_a)

    # Construct the control matrix
    Tar_matrix = Target_offline_adjoint_FOTR(V_delta_primal, V_delta_adjoint, W_delta_adjoint,
                                             samples, modes_a, modes_p, CTC)

    return LHS_matrix, RHS_matrix, Tar_matrix


@njit
def RHS_adjoint_sPODG_FOTR_expl(as_adj, as_, qs_target, lhs, rhs, tar, CTC, Vda, Wda, modes_a, modes_p, delta_s, dx):
    # Prepare the online adjoint matrices
    M, A = Matrices_online_adjoint_FOTR_expl(lhs, rhs, tar, CTC, Vda, Wda, qs_target, as_adj, as_,
                                             modes_a, modes_p, delta_s, dx)

    # Solve the linear system of equations
    if np.linalg.cond(M) == np.inf:
        return solve_lin_system_Tikh_reg(M, A)
    else:
        return solve_lin_system(M, A)


def TI_adjoint_sPODG_FOTR(lhs, rhs, tar, CTC, Vda, Wda, a_a, as_, qs_target, modes_a, modes_p, delta_s, dx, Nt, dt,
                          scheme):
    # Time loop
    as_adj = np.zeros((modes_a + 1, Nt), order="F")
    as_ = np.asfortranarray(as_)
    as_adj[:, -1] = a_a

    if scheme == "RK4":
        for n in range(1, Nt):
            as_adj[:, -(n + 1)] = rk4_sPODG_adj_(RHS_adjoint_sPODG_FOTR_expl, as_adj[:, -n],
                                                 as_[:, -n], as_[:, -(n + 1)],
                                                 qs_target[:, -n], qs_target[:, -(n + 1)], - dt, lhs, rhs, tar, CTC,
                                                 Vda, Wda, modes_a, modes_p, delta_s, dx)
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


def mat_primal_sPODG_FRTO(V_delta_primal, W_delta_primal, U_delta_primal, CTC, A_p, psi, samples, modes):
    # Construct LHS matrix
    LHS_matrix = LHS_offline_primal_FRTO(V_delta_primal, W_delta_primal, modes)

    # Construct RHS matrix
    RHS_matrix = RHS_offline_primal_FRTO(V_delta_primal, W_delta_primal, A_p, modes)

    # Construct the control matrix
    C_matrix = Control_offline_primal_FRTO(V_delta_primal, W_delta_primal, U_delta_primal, psi, samples, modes)

    # Construct the online adjoint target matrix
    Tar_matrix = Target_online_adjoint_FRTO(V_delta_primal, W_delta_primal, CTC, samples, modes)

    return LHS_matrix, RHS_matrix, C_matrix, Tar_matrix


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
def RHS_adjoint_sPODG_FRTO_expl(a, f, a_, qs_target, a_dot, M1, M2, N, A1, A2, C, tara, CTC, Vdp, Wdp, modes, delta_s,
                                dx):
    # Prepare the online primal matrices
    M, A = Matrices_online_adjoint_FRTO_expl(M1, M2, N, A1, A2, C, tara, CTC, Vdp, Wdp, f, a, a_, qs_target, a_dot,
                                             modes, delta_s, dx)
    # Solve the linear system of equations
    X = solve_lin_system(M, -A)

    return X


def RHS_adjoint_sPODG_FRTO_impl(a, f, a_, qs_target, a_dot, dt, M1, M2, N, A1, A2, C, tara, CTC, Vdp, Wdp, modes,
                                delta_s, dx,
                                scheme):
    # Solve the linear system of equations
    M, A, T = Matrices_online_adjoint_FRTO_impl(M1, M2, N, A1, A2, C, tara, CTC, Vdp, Wdp, f, a, a_, qs_target, a_dot,
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


def TI_adjoint_sPODG_FRTO(at_adj, f0, a_, qs_target, a_dot, lhsp, rhsp, C, tara, CTC, Vdp, Wdp, modes, delta_s, Nt, dt,
                          dx,
                          scheme):
    as_adj = np.zeros((at_adj.shape[0], Nt), order="F")
    as_adj[:, -1] = at_adj

    M1 = lhsp[0]
    N = lhsp[1]
    M2 = lhsp[2]
    A1 = rhsp[0]
    A2 = rhsp[1]

    if scheme == "RK4":
        for n in range(1, Nt):
            as_adj[:, -(n + 1)] = rk4_sPODG_adj(RHS_adjoint_sPODG_FRTO_expl, as_adj[:, -n], f0[:, -n],
                                                f0[:, -(n + 1)],
                                                a_[:, -n], a_[:, -(n + 1)], qs_target[:, -n],
                                                qs_target[:, -(n + 1)],
                                                a_dot[..., -n], - dt,
                                                M1, M2, N, A1, A2, C, tara, CTC,
                                                Vdp, Wdp, modes, delta_s, dx)

    if scheme == "implicit_midpoint":
        for n in range(1, Nt):
            as_adj[:, -(n + 1)] = implicit_midpoint_sPODG_adj(RHS_adjoint_sPODG_FRTO_impl, as_adj[:, -n], f0[:, -n],
                                                              f0[:, -(n + 1)],
                                                              a_[:, -n], a_[:, -(n + 1)], qs_target[:, -n],
                                                              qs_target[:, -(n + 1)],
                                                              a_dot[..., -n], - dt,
                                                              M1, M2, N, A1, A2, C, tara, CTC,
                                                              Vdp, Wdp, modes, delta_s, dx, scheme)
    elif scheme == "DIRK":
        for n in range(1, Nt):
            as_adj[:, -(n + 1)] = DIRK_sPODG_adj(RHS_adjoint_sPODG_FRTO_impl, as_adj[:, -n], f0[:, -n],
                                                 f0[:, -(n + 1)],
                                                 a_[:, -n], a_[:, -(n + 1)], qs_target[:, -n],
                                                 qs_target[:, -(n + 1)],
                                                 a_dot[..., -n], - dt,
                                                 M1, M2, N, A1, A2, C, tara, CTC,
                                                 Vdp, Wdp, modes, delta_s, dx, scheme)
    elif scheme == "BDF2":
        # last 2 steps (x_{n-1}, x_{n-2}) with RK4 (Effectively 2nd order)
        for n in range(1, 2):
            as_adj[:, -(n + 1)] = rk4_sPODG_adj(RHS_adjoint_sPODG_FRTO_expl, as_adj[:, -n], f0[:, -n],
                                                f0[:, -(n + 1)],
                                                a_[:, -n], a_[:, -(n + 1)], qs_target[:, -n],
                                                qs_target[:, -(n + 1)],
                                                a_dot[..., -n], - dt,
                                                M1, M2, N, A1, A2, C, tara, CTC,
                                                Vdp, Wdp, modes, delta_s, dx)
        for n in range(2, Nt):
            as_adj[:, -(n + 1)] = bdf2_sPODG_adj(RHS_adjoint_sPODG_FRTO_impl, as_adj,
                                                 f0[:, -(n + 1)],
                                                 a_[:, -(n + 1)],
                                                 qs_target[:, -(n + 1)],
                                                 a_dot[..., -(n + 1)], - dt,
                                                 M1, M2, N, A1, A2, C, tara, CTC,
                                                 Vdp, Wdp, modes, delta_s, dx, scheme, n)

    return as_adj


# -------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #


#############
## FOTR sPOD
#############

@njit
def IC_primal_sPODG_FOTR_kdv(q0, V):
    z = 0
    a = V.transpose() @ q0
    # Initialize the shifts with zero for online phase
    a = np.concatenate((a, np.asarray([z])))

    return a


def mat_primal_sPODG_FOTR_kdv(T_delta, V_delta_primal, W_delta_primal, U_delta_primal, V_deim, num_sample,
                              modes, modes_deim, params_primal, delta_s):
    # Construct LHS matrix
    LHS_matrix = LHS_offline_primal_FOTR_kdv(V_delta_primal, W_delta_primal, U_delta_primal, num_sample, modes)

    # Construct RHS matrix
    A = - params_primal['alpha'] * params_primal['c'] * params_primal['D1'] \
        - params_primal['gamma'] * params_primal['D3'] + params_primal['nu'] * params_primal['D2']
    RHS_matrix = RHS_offline_primal_FOTR_kdv(V_delta_primal, W_delta_primal, U_delta_primal, A, num_sample, modes)

    # Construct the DEIM matrices
    omega = params_primal['omega']
    DEIM_matrix = DEIM_primal_FOTR_kdv(T_delta, V_delta_primal, W_delta_primal, V_deim,
                                       omega,
                                       num_sample, modes, modes_deim, delta_s)

    # Construct the control matrix
    B = params_primal['B']
    C_matrix = Control_offline_primal_FOTR(V_delta_primal, W_delta_primal, B, num_sample, modes)

    return LHS_matrix, RHS_matrix, DEIM_matrix, C_matrix


@njit
def RHS_primal_sPODG_FOTR_kdv_expl(a, f, lhs, rhs, deim, c, ds, modes):
    # Prepare the online primal matrices
    M, A, intervalIdx, weight = Matrices_online_primal_FOTR_kdv_expl(lhs, rhs, deim, c, f, a, ds, modes)

    # Solve the linear system of equations
    X = solve_lin_system(M, A)

    return X, intervalIdx, weight


def TI_primal_sPODG_FOTR_kdv_expl(lhs, rhs, deim, c, a, f, delta_s, modes, Nt, dt):
    # Time loop
    as_ = np.zeros((a.shape[0], Nt), order="F")
    f = np.asfortranarray(f)
    IntIds = np.zeros(Nt, dtype=np.int32)
    weights = np.zeros(Nt)

    as_[:, 0] = a

    for n in range(1, Nt):
        as_[:, n], _, IntIds[n - 1], weights[n - 1] = rk4_sPODG_prim_kdvb(RHS_primal_sPODG_FOTR_kdv_expl,
                                                                          as_[:, n - 1],
                                                                          f[:, n - 1],
                                                                          f[:, n], dt, lhs, rhs, deim, c,
                                                                          delta_s,
                                                                          modes)

    IntIds[-1], weights[-1] = findIntervalAndGiveInterpolationWeight_1D(delta_s[2], -as_[-1, -1])

    print('RK4 reduced primal finished')

    return as_, IntIds, weights


@njit
def IC_adjoint_sPODG_FOTR_kdv(Nm_a, z):
    a = np.concatenate((np.zeros(Nm_a), np.asarray([z])))
    return a


def mat_adjoint_sPODG_FOTR_kdv(T_delta, V_delta_adjoint, W_delta_adjoint, U_delta_adjoint,
                               V_delta_primal, W_delta_primal, V_deim_a, num_sample,
                               modes_p, modes_a, modes_deim_a, params_adjoint):
    # First two funtion calls can take advantage of the already defined FOTR functions for primal.

    # Construct LHS matrix
    LHS_matrix = LHS_offline_primal_FOTR_kdv(V_delta_adjoint, W_delta_adjoint, U_delta_adjoint, num_sample, modes_a)

    # Construct RHS matrix
    A = - params_adjoint['alpha'] * params_adjoint['c'] * params_adjoint['D1'].T \
        - params_adjoint['gamma'] * params_adjoint['D3'].T + params_adjoint['nu'] * params_adjoint['D2'].T
    RHS_matrix = RHS_offline_primal_FOTR_kdv(V_delta_adjoint, W_delta_adjoint, U_delta_adjoint, A, num_sample, modes_a)

    # Construct the DEIM matrices
    omega = params_adjoint['omega']
    DEIM_matrix, DEIM_mix_mat = DEIM_adjoint_FOTR_kdv(T_delta, V_delta_adjoint, W_delta_adjoint, U_delta_adjoint,
                                                      V_delta_primal, W_delta_primal, V_deim_a,
                                                      omega,
                                                      num_sample, modes_p, modes_a, modes_deim_a)

    # Construct the Target matrix
    CTC = params_adjoint['CTC']
    TAR_matrix = Target_offline_adjoint_FOTR(V_delta_primal, V_delta_adjoint, W_delta_adjoint,
                                             num_sample, modes_a, modes_p, CTC)

    return LHS_matrix, RHS_matrix, DEIM_matrix, DEIM_mix_mat, TAR_matrix


@njit
def RHS_adjoint_sPODG_FOTR_kdv_expl(as_adj, as_, qs_target, lhs, rhs, deim, deim_mix, tar,
                                    CTC, Vda, Wda, modes_a, modes_p, delta_s, dx):
    # Prepare the online adjoint matrices
    M, A = Matrices_online_adjoint_FOTR_kdv_expl(lhs, rhs, deim, deim_mix, tar, CTC, Vda, Wda, qs_target, as_adj, as_,
                                                 modes_a, modes_p, delta_s, dx)

    # Solve the linear system of equations
    if np.linalg.cond(M) == np.inf:
        return solve_lin_system_Tikh_reg(M, A)
    else:
        return solve_lin_system(M, A)


def TI_adjoint_sPODG_FOTR_kdv_expl(lhs, rhs, deim, deim_mix, tar, Vda, Wda, a_a, as_, qs_target,
                                   modes_a, modes_p, delta_s, dx, Nt, dt, params_adjoint):
    # Time loop
    as_adj = np.zeros((modes_a + 1, Nt), order="F")
    as_ = np.asfortranarray(as_)
    as_adj[:, -1] = a_a

    CTC = params_adjoint['CTC']
    for n in range(1, Nt):
        as_adj[:, -(n + 1)] = rk4_sPODG_adj_kdvb(RHS_adjoint_sPODG_FOTR_kdv_expl, as_adj[:, -n],
                                                 as_[:, -n], as_[:, -(n + 1)],
                                                 qs_target[:, -n], qs_target[:, -(n + 1)],
                                                 - dt, lhs, rhs,
                                                 deim, deim_mix, tar, CTC,
                                                 Vda, Wda, modes_a, modes_p, delta_s, dx)
    print('RK4 reduced adjoint finished')
    return as_adj


#############
## FRTO sPOD
#############


@njit
def IC_primal_sPODG_FRTO_kdv(q0, V):
    z = 0
    a = V.transpose() @ q0
    # Initialize the shifts with zero for online phase
    a = np.concatenate((a, np.asarray([z])))

    return a


def mat_primal_sPODG_FRTO_kdv(T_delta, V_delta_primal, W_delta_primal, U_delta_primal, V_deim, num_sample,
                              modes, modes_deim, params_primal, params_adjoint, delta_s):
    # Construct LHS matrix
    LHS_matrix = LHS_offline_primal_FRTO_kdv(V_delta_primal, W_delta_primal, U_delta_primal, num_sample, modes)

    # Construct RHS matrix
    A = - params_primal['alpha'] * params_primal['c'] * params_primal['D1'] \
        - params_primal['gamma'] * params_primal['D3'] + params_primal['nu'] * params_primal['D2']
    RHS_matrix = RHS_offline_primal_FRTO_kdv(V_delta_primal, W_delta_primal, U_delta_primal, A, num_sample, modes)

    # Construct the DEIM matrices
    omega = params_primal['omega']
    D1 = params_primal['D1']
    DEIM_matrix, DEIM_mat = DEIM_primal_FRTO_kdv(T_delta, V_delta_primal, W_delta_primal, U_delta_primal, V_deim, D1,
                                                 omega, num_sample, modes, modes_deim, delta_s)

    # Construct the control matrix
    B = params_primal['B']
    C_matrix = Control_offline_primal_FRTO(V_delta_primal, W_delta_primal, U_delta_primal, B, num_sample, modes)

    # Construct the target matrix for adjoint
    CTC = params_adjoint['CTC']
    TAR_matrix = Target_online_adjoint_FRTO(V_delta_primal, W_delta_primal, CTC, num_sample, modes)

    return LHS_matrix, RHS_matrix, DEIM_matrix, DEIM_mat, C_matrix, TAR_matrix


@njit
def RHS_primal_sPODG_FRTO_kdv_expl(a, f, lhs, rhs, deim, c, ds, modes):
    # Prepare the online primal matrices
    M, A, intervalIdx, weight = Matrices_online_primal_FRTO_kdv_expl(lhs, rhs, deim, c, f, a, ds, modes)

    X = solve_lin_system(M, A)

    return X, intervalIdx, weight


def TI_primal_sPODG_FRTO_kdv_expl(lhs, rhs, deim, c, a, f0, delta_s, modes, Nt, dt):
    # Time loop
    types_of_dots = 5  # derivatives to approximate
    as_ = np.zeros((a.shape[0], Nt), order="F")
    as_dot = np.zeros((types_of_dots, a.shape[0], Nt), order="F")
    f0 = np.asfortranarray(f0)
    IntIds = np.zeros(Nt, dtype=np.int32)
    weights = np.zeros(Nt)

    as_[:, 0] = a
    for n in range(1, Nt):
        as_[:, n], as_dot[..., n], IntIds[n - 1], weights[n - 1] = rk4_sPODG_prim_kdvb(RHS_primal_sPODG_FRTO_kdv_expl,
                                                                                       as_[:, n - 1],
                                                                                       f0[:, n - 1],
                                                                                       f0[:, n], dt, lhs, rhs, deim, c,
                                                                                       delta_s,
                                                                                       modes)

    IntIds[-1], weights[-1] = findIntervalAndGiveInterpolationWeight_1D(delta_s[2], -as_[-1, -1])
    as_dot[..., 0] = as_dot[..., 1].copy()

    print('RK4 reduced primal finished')

    return as_, as_dot, IntIds, weights


@njit
def Jacobian_sPODG_FRTO_kdv(M1, M2, N, A1, A2, D, M1_dash, M2_dash, N_dash, A1_dash, A2_dash,
                            WT_B, UT_B, ST_V, ST_D1V, ST_D1V_dash, STdash_V, STdash_D1V, ST_U_dash, ST_U_inv,
                            epsilon1, epsilon2, epsilon3, epsilon1_dash, epsilon2_dash,
                            q_mid, u_mid, del_q, modes, dt):
    del_a = del_q[:modes]
    del_z = del_q[modes:]
    a_mid = q_mid[:modes]
    z_mid = q_mid[modes:]

    # Left hand side Jacobian parts
    J_l_1_1 = np.empty((modes + 1, modes + 1))
    J_l_1_1[:modes, :modes] = 0.0
    J_l_1_1[:modes, modes:] = N @ del_a[:, None]
    J_l_1_1[modes:, :modes] = J_l_1_1[:modes, modes:].T
    J_l_1_1[modes:, modes:] = del_a[None, :] @ (M2 @ D) + (D.T @ M2) @ del_a[:, None]

    J_l_1_2 = np.empty((modes + 1, modes + 1))
    J_l_1_2[:modes, :modes] = M1_dash * del_z
    J_l_1_2[:modes, modes:] = (N_dash @ D) * del_z
    J_l_1_2[modes:, :modes] = J_l_1_2[:modes, modes:].T
    J_l_1_2[modes:, modes:] = ((D.T @ M2_dash) @ D) * del_z

    J_l_1_3 = np.empty((modes + 1, modes + 1))
    J_l_1_3[:modes, :modes] = M1
    J_l_1_3[:modes, modes:] = N @ D
    J_l_1_3[modes:, :modes] = J_l_1_3[:modes, modes:].T
    J_l_1_3[modes:, modes:] = ((D.T @ M2) @ D)

    J_l = (0.5 * (J_l_1_1 + J_l_1_2) + J_l_1_3) / dt

    # Right hand side Jacobian parts
    J_r_1_1 = np.empty((modes + 1, modes + 1))
    J_r_1_1[:modes, :modes] = 0.0
    J_r_1_1[:modes, modes:] = 0.0
    J_r_1_1[modes:, :modes] = a_mid[None, :] @ A2
    J_r_1_1[modes:, modes:] = 0.0

    J_r_1_2 = np.empty((modes + 1, modes + 1))
    J_r_1_2[:modes, :modes] = A1_dash * z_mid
    J_r_1_2[:modes, modes:] = 0.0
    J_r_1_2[modes:, :modes] = D.T @ (A2_dash * z_mid)
    J_r_1_2[modes:, modes:] = 0.0

    J_r_1_3 = np.empty((modes + 1, modes + 1))
    J_r_1_3[:modes, :modes] = A1
    J_r_1_3[:modes, modes:] = 0.0
    J_r_1_3[modes:, :modes] = D.T @ A2
    J_r_1_3[modes:, modes:] = 0.0

    J_r_1 = 0.5 * (J_r_1_1 + J_r_1_2 + J_r_1_3)

    J_r_2 = np.empty((modes + 1, modes + 1))
    lamda1 = np.diag(ST_D1V.dot(a_mid)) @ ST_V + np.diag(ST_V.dot(a_mid)) @ ST_D1V
    lamda2 = ((ST_V.dot(a_mid)) * (ST_D1V.dot(a_mid)))[:, None]
    lamda3 = (ST_D1V.dot(a_mid) * (STdash_V.dot(a_mid) + ST_D1V.dot(a_mid)))[:, None]
    lamda4 = (ST_V.dot(a_mid) * (STdash_D1V.dot(a_mid) + ST_D1V_dash.dot(a_mid)))[:, None]
    J_r_2[:modes, :modes] = 0.5 * epsilon1 @ lamda1
    J_r_2[:modes, modes:] = 0.5 * (epsilon2 @ lamda2 + epsilon1_dash @ lamda2 - epsilon1 @ (
            ST_U_dash @ (ST_U_inv @ lamda2)) + epsilon1 @ (lamda3 + lamda4))
    J_r_2[modes:, :modes] = 0.5 * ((epsilon2 @ lamda2).T + D.T @ (epsilon2 @ lamda1))
    J_r_2[modes:, modes:] = 0.5 * D.T @ (epsilon3 @ lamda2 + epsilon2_dash @ lamda2 - epsilon2 @ (
            ST_U_dash @ (ST_U_inv @ lamda2)) + epsilon2 @ (lamda3 + lamda4))

    J_r_3 = np.empty((modes + 1, modes + 1))
    J_r_3[:modes, :modes] = 0.0
    J_r_3[:modes, modes:] = 0.5 * (WT_B @ u_mid[:, None])  # (V^T)' B = W^T B
    J_r_3[modes:, :modes] = 0.5 * (WT_B @ u_mid)[None, :]
    J_r_3[modes:, modes:] = 0.5 * (D.T @ (UT_B @ u_mid[:, None]))  # (W^T)' B = U^T B

    J = J_l - J_r_1 + J_r_2 - J_r_3

    return J


@njit
def RHS_primal_sPODG_FRTO_kdv_impl(LHS_matrix, RHS_matrix, DEIM_matrix, DEIM_mat, C_matrix, f, a, ds, modes):
    M = np.empty((modes + 1, modes + 1))
    A = np.empty(modes + 1)
    as_ = a[:-1]
    z = a[-1]

    # Compute the interpolation weight and the interval in which the shift lies
    intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], -z)

    Da = as_.reshape(-1, 1)

    M1 = np.add(weight * LHS_matrix[0, intervalIdx], (1 - weight) * LHS_matrix[0, intervalIdx + 1])
    N = np.add(weight * LHS_matrix[1, intervalIdx], (1 - weight) * LHS_matrix[1, intervalIdx + 1])
    M2 = np.add(weight * LHS_matrix[2, intervalIdx], (1 - weight) * LHS_matrix[2, intervalIdx + 1])
    A1 = np.add(weight * RHS_matrix[0, intervalIdx], (1 - weight) * RHS_matrix[0, intervalIdx + 1])
    A2 = np.add(weight * RHS_matrix[1, intervalIdx], (1 - weight) * RHS_matrix[1, intervalIdx + 1])
    VT_B = np.add(weight * C_matrix[0, intervalIdx], (1 - weight) * C_matrix[0, intervalIdx + 1])
    WT_B = np.add(weight * C_matrix[1, intervalIdx], (1 - weight) * C_matrix[1, intervalIdx + 1])
    UT_B = np.add(weight * C_matrix[2, intervalIdx], (1 - weight) * C_matrix[2, intervalIdx + 1])
    M1_dash = np.add(weight * LHS_matrix[3, intervalIdx], (1 - weight) * LHS_matrix[3, intervalIdx + 1])
    N_dash = np.add(weight * LHS_matrix[4, intervalIdx], (1 - weight) * LHS_matrix[4, intervalIdx + 1])
    M2_dash = np.add(weight * LHS_matrix[5, intervalIdx], (1 - weight) * LHS_matrix[5, intervalIdx + 1])
    A1_dash = np.add(weight * RHS_matrix[2, intervalIdx], (1 - weight) * RHS_matrix[2, intervalIdx + 1])
    A2_dash = np.add(weight * RHS_matrix[3, intervalIdx], (1 - weight) * RHS_matrix[3, intervalIdx + 1])

    epsilon1 = np.add(weight * DEIM_matrix[0, intervalIdx], (1 - weight) * DEIM_matrix[0, intervalIdx + 1])
    epsilon2 = np.add(weight * DEIM_matrix[1, intervalIdx], (1 - weight) * DEIM_matrix[1, intervalIdx + 1])
    epsilon3 = np.add(weight * DEIM_matrix[2, intervalIdx], (1 - weight) * DEIM_matrix[2, intervalIdx + 1])
    epsilon1_dash = np.add(weight * DEIM_matrix[3, intervalIdx], (1 - weight) * DEIM_matrix[3, intervalIdx + 1])
    epsilon2_dash = np.add(weight * DEIM_matrix[4, intervalIdx], (1 - weight) * DEIM_matrix[4, intervalIdx + 1])
    ST_V = (np.add(weight * DEIM_matrix[5, intervalIdx], (1 - weight) * DEIM_matrix[5, intervalIdx + 1])).T
    ST_D1V = (np.add(weight * DEIM_matrix[6, intervalIdx],
                     (1 - weight) * DEIM_matrix[6, intervalIdx + 1])).T  # S^T D1 @ V = S^T V' = S^T W
    ST_D1V_dash = (np.add(weight * DEIM_matrix[7, intervalIdx],
                          (1 - weight) * DEIM_matrix[7, intervalIdx + 1])).T  # S^T D1 @ V' = S^T D1 @ W = S^T U
    STdash_V = (
        np.add(weight * DEIM_matrix[8, intervalIdx], (1 - weight) * DEIM_matrix[8, intervalIdx + 1])).T  # (S^T)' @ V
    STdash_D1V = (np.add(weight * DEIM_matrix[9, intervalIdx],
                         (1 - weight) * DEIM_matrix[9, intervalIdx + 1])).T  # (S^T)' D1 @ V = (S^T)' @ W

    ST_U_dash = np.add(weight * DEIM_mat[0, intervalIdx], (1 - weight) * DEIM_mat[0, intervalIdx + 1])
    ST_U_inv = np.add(weight * DEIM_mat[1, intervalIdx], (1 - weight) * DEIM_mat[1, intervalIdx + 1])

    M[:modes, :modes] = M1
    M[:modes, modes:] = N @ Da
    M[modes:, :modes] = M[:modes, modes:].T
    M[modes:, modes:] = Da.T @ (M2 @ Da)

    # DEIM term
    fnonlinear = (ST_V.dot(as_)) * (ST_D1V.dot(as_))

    A[:modes] = A1 @ as_ - epsilon1 @ fnonlinear + VT_B @ f
    A[modes:] = Da.T @ (A2 @ as_ - epsilon2 @ fnonlinear + WT_B @ f)

    return np.ascontiguousarray(M), np.ascontiguousarray(A), np.ascontiguousarray(Da), np.ascontiguousarray(
        M1), np.ascontiguousarray(M2), \
        np.ascontiguousarray(N), np.ascontiguousarray(A1), np.ascontiguousarray(A2), np.ascontiguousarray(VT_B), \
        np.ascontiguousarray(WT_B), np.ascontiguousarray(UT_B), np.ascontiguousarray(M1_dash), np.ascontiguousarray(
        M2_dash), \
        np.ascontiguousarray(N_dash), np.ascontiguousarray(A1_dash), np.ascontiguousarray(A2_dash), \
        np.ascontiguousarray(epsilon1), np.ascontiguousarray(epsilon2), np.ascontiguousarray(epsilon3), \
        np.ascontiguousarray(epsilon1_dash), \
        np.ascontiguousarray(epsilon2_dash), \
        np.ascontiguousarray(ST_V), \
        np.ascontiguousarray(ST_D1V), np.ascontiguousarray(ST_D1V_dash), np.ascontiguousarray(
        STdash_V), np.ascontiguousarray(STdash_D1V), \
        np.ascontiguousarray(ST_U_dash), np.ascontiguousarray(ST_U_inv), \
        intervalIdx, weight


def TI_primal_sPODG_FRTO_kdv_impl(lhs, rhs, deim, deim_hl, c, a, f0, delta_s, modes, Nt, dt):
    # Time loop
    types_of_dots = 2  # derivatives to approximate
    as_ = np.zeros((a.shape[0], Nt), order="F")
    as_dot = np.zeros((types_of_dots, a.shape[0], Nt), order="F")
    f0 = np.asfortranarray(f0)
    IntIds = np.zeros(Nt, dtype=np.int32)
    weights = np.zeros(Nt)

    as_[:, 0] = a
    for n in range(1, Nt):
        # print(n)
        as_[:, n], as_dot[..., n], IntIds[n - 1], weights[n - 1] = implicit_midpoint_sPODG_FRTO_primal_kdvb(
            RHS_primal_sPODG_FRTO_kdv_impl,
            Jacobian_sPODG_FRTO_kdv,
            as_[:, n - 1],
            f0[:, n - 1],
            f0[:, n],
            lhs,
            rhs,
            deim,
            deim_hl,
            c,
            delta_s,
            modes,
            dt)

    IntIds[-1], weights[-1] = findIntervalAndGiveInterpolationWeight_1D(delta_s[2], -as_[-1, -1])
    as_dot[..., 0] = as_dot[..., 1].copy()

    print('Implicit midpoint reduced primal finished')
    return as_, as_dot, IntIds, weights


@njit
def IC_adjoint_sPODG_FRTO_kdv(modes):
    z = 0
    # Initialize the shifts with zero for online phase
    a = np.concatenate((np.zeros(modes), np.asarray([z])))

    return a


@njit
def RHS_adjoint_sPODG_FRTO_kdv_impl(LHS_matrix, RHS_matrix, DEIM_matrix, DEIM_mat, C_matrix, TAR_matrix, CTC, Vdp, Wdp,
                                    as_p, as_p_dot, f, qs_tar, ds, dx, modes):
    M = np.empty((modes + 1, modes + 1))
    E = np.empty((modes + 1, modes + 1))
    T = np.empty(modes + 1)

    as_ = as_p[:-1]  # Take the modes from the primal solution
    z_ = as_p[-1]  # Take the shifts from the primal solution
    as_dot = as_p_dot[:-1]  # Take the modes derivative from the primal
    z_dot = as_p_dot[-1:]  # Take the shift derivative from the primal

    # Compute the interpolation weight and the interval in which the shift lies
    # (This is very IMPORTANT. DO NOT REPLACE !!!!!!) The time integration steps need the intermediate value of
    # shift for correct interpolation index and weights calculation
    intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], -z_)

    Da_p = as_.reshape(-1, 1)

    M1 = np.add(weight * LHS_matrix[0, intervalIdx], (1 - weight) * LHS_matrix[0, intervalIdx + 1])
    N = np.add(weight * LHS_matrix[1, intervalIdx], (1 - weight) * LHS_matrix[1, intervalIdx + 1])
    M2 = np.add(weight * LHS_matrix[2, intervalIdx], (1 - weight) * LHS_matrix[2, intervalIdx + 1])
    A1 = np.add(weight * RHS_matrix[0, intervalIdx], (1 - weight) * RHS_matrix[0, intervalIdx + 1])
    A2 = np.add(weight * RHS_matrix[1, intervalIdx], (1 - weight) * RHS_matrix[1, intervalIdx + 1])
    VT_B = np.add(weight * C_matrix[0, intervalIdx], (1 - weight) * C_matrix[0, intervalIdx + 1])
    WT_B = np.add(weight * C_matrix[1, intervalIdx], (1 - weight) * C_matrix[1, intervalIdx + 1])
    UT_B = np.add(weight * C_matrix[2, intervalIdx], (1 - weight) * C_matrix[2, intervalIdx + 1])
    M1_dash = np.add(weight * LHS_matrix[3, intervalIdx], (1 - weight) * LHS_matrix[3, intervalIdx + 1])
    N_dash = np.add(weight * LHS_matrix[4, intervalIdx], (1 - weight) * LHS_matrix[4, intervalIdx + 1])
    M2_dash = np.add(weight * LHS_matrix[5, intervalIdx], (1 - weight) * LHS_matrix[5, intervalIdx + 1])
    A1_dash = np.add(weight * RHS_matrix[2, intervalIdx], (1 - weight) * RHS_matrix[2, intervalIdx + 1])
    A2_dash = np.add(weight * RHS_matrix[3, intervalIdx], (1 - weight) * RHS_matrix[3, intervalIdx + 1])

    epsilon1 = np.add(weight * DEIM_matrix[0, intervalIdx], (1 - weight) * DEIM_matrix[0, intervalIdx + 1])
    epsilon2 = np.add(weight * DEIM_matrix[1, intervalIdx], (1 - weight) * DEIM_matrix[1, intervalIdx + 1])
    epsilon3 = np.add(weight * DEIM_matrix[2, intervalIdx], (1 - weight) * DEIM_matrix[2, intervalIdx + 1])
    epsilon1_dash = np.add(weight * DEIM_matrix[3, intervalIdx], (1 - weight) * DEIM_matrix[3, intervalIdx + 1])
    epsilon2_dash = np.add(weight * DEIM_matrix[4, intervalIdx], (1 - weight) * DEIM_matrix[4, intervalIdx + 1])
    ST_V = (np.add(weight * DEIM_matrix[5, intervalIdx], (1 - weight) * DEIM_matrix[5, intervalIdx + 1])).T
    ST_D1V = (np.add(weight * DEIM_matrix[6, intervalIdx],
                     (1 - weight) * DEIM_matrix[6, intervalIdx + 1])).T  # S^T D1 @ V = S^T V' = S^T W
    ST_D1V_dash = (np.add(weight * DEIM_matrix[7, intervalIdx],
                          (1 - weight) * DEIM_matrix[7, intervalIdx + 1])).T  # S^T D1 @ V' = S^T D1 @ W = S^T U
    STdash_V = (
        np.add(weight * DEIM_matrix[8, intervalIdx], (1 - weight) * DEIM_matrix[8, intervalIdx + 1])).T  # (S^T)' @ V
    STdash_D1V = (np.add(weight * DEIM_matrix[9, intervalIdx],
                         (1 - weight) * DEIM_matrix[9, intervalIdx + 1])).T  # (S^T)' D1 @ V = (S^T)' @ W

    ST_U_dash = np.add(weight * DEIM_mat[0, intervalIdx], (1 - weight) * DEIM_mat[0, intervalIdx + 1])
    ST_U_inv = np.add(weight * DEIM_mat[1, intervalIdx], (1 - weight) * DEIM_mat[1, intervalIdx + 1])

    lamda1 = np.diag(ST_D1V.dot(as_)) @ ST_V + np.diag(ST_V.dot(as_)) @ ST_D1V
    lamda2 = ((ST_V.dot(as_)) * (ST_D1V.dot(as_)))
    lamda3 = (ST_D1V.dot(as_) * (STdash_V.dot(as_) + ST_D1V.dot(as_)))
    lamda4 = (ST_V.dot(as_) * (STdash_D1V.dot(as_) + ST_D1V_dash.dot(as_)))

    VT = np.add(weight * Vdp[intervalIdx], (1 - weight) * Vdp[intervalIdx + 1]).T
    WT = np.add(weight * Wdp[intervalIdx], (1 - weight) * Wdp[intervalIdx + 1]).T
    VTqs_tar = VT[:, CTC] @ qs_tar[CTC]
    WTqs_tar = WT[:, CTC] @ qs_tar[CTC]

    VTV = np.add(weight * TAR_matrix[0, intervalIdx], (1 - weight) * TAR_matrix[0, intervalIdx + 1])
    WTV = np.add(weight * TAR_matrix[1, intervalIdx], (1 - weight) * TAR_matrix[1, intervalIdx + 1])

    # Assemble the M matrix
    M[:modes, :modes] = M1.T
    M[:modes, modes:] = N @ Da_p
    M[modes:, :modes] = M[:modes, modes:].T
    M[modes:, modes:] = Da_p.T @ (M2.T @ Da_p)

    # Assemble the E matrix
    E[:modes, :modes] = E11_kdvb(M1_dash, N, A1, epsilon1, lamda1, z_dot, modes).T
    E[:modes, modes:] = E12_kdvb(M2, N, N_dash, A2, Da_p, WT_B, epsilon2, lamda1, lamda2, as_dot, z_dot, as_, f,
                                 modes).T
    E[modes:, :modes] = E21_kdvb(N, Da_p, M1_dash, N_dash, A1_dash, WT_B, epsilon1, epsilon2, epsilon1_dash,
                                 ST_U_dash, ST_U_inv, lamda2, lamda3, lamda4, as_dot, z_dot, as_, f).T
    E[modes:, modes:] = E22_kdvb(M2, M2_dash, N_dash, A2_dash, epsilon2, epsilon2_dash, epsilon3,
                                 ST_U_dash, ST_U_inv, lamda2, lamda3, lamda4, Da_p, UT_B, as_dot, z_dot, as_, f).T

    # Assemble the T vector
    T[:modes] = C1(VTV, as_, VTqs_tar, dx)
    T[modes:] = C2(WTV, as_, WTqs_tar, dx)

    return np.ascontiguousarray(M), np.ascontiguousarray(E), np.ascontiguousarray(T)


def TI_adjoint_sPODG_FRTO_kdv_impl(as_adj0, f, as_p, qs_target, a_dot, lhs, rhs, deim, deim_hl, c, tar, Vdp, Wdp,
                                   delta_s, modes, Nt,
                                   dt, dx, params_adjoint):
    as_adj = np.zeros((as_adj0.shape[0], Nt), order="F")
    as_adj[:, -1] = as_adj0

    for n in range(1, Nt):
        as_adj[:, -(n + 1)] = implicit_midpoint_sPODG_FRTO_adjoint_kdvb(RHS_adjoint_sPODG_FRTO_kdv_impl,
                                                                        as_adj[:, -n],
                                                                        as_p[:, -n],
                                                                        as_p[:, -(n + 1)],
                                                                        qs_target[:, -n],
                                                                        qs_target[:, -(n + 1)],
                                                                        f[:, -n],
                                                                        f[:, -(n + 1)],
                                                                        a_dot[..., -n],
                                                                        lhs,
                                                                        rhs,
                                                                        deim,
                                                                        deim_hl,
                                                                        c,
                                                                        tar,
                                                                        Vdp,
                                                                        Wdp,
                                                                        delta_s,
                                                                        modes,
                                                                        dx,
                                                                        dt,
                                                                        params_adjoint)

    print('Implicit midpoint reduced adjoint finished')

    return as_adj
