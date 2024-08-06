from Helper_sPODG import *
from rk4 import rk4, rk4_, rk4__
from numba import njit


class advection:
    def __init__(self, Nxi: int, Neta: int, timesteps: int, cfl: float, tilt_from: int) -> None:
        # Assertion statements for checking the sanctity of the input variables
        assert Nxi > 0, f"Please input sensible values for the X grid points"
        assert Neta > 0, f"Please input sensible values for the Y grid points"
        assert timesteps >= 0, f"Please input sensible values for time steps"

        # First we define the public variables of the class. All the variables with "__" in front are private variables
        self.X = None
        self.Y = None
        self.X_2D = None
        self.Y_2D = None
        self.dx = None
        self.dy = None
        self.t = None
        self.dt = None

        # Private variables
        self.Lxi = 100
        self.Leta = 1
        self.Nxi = Nxi
        self.Neta = Neta
        self.NN = self.Nxi * self.Neta
        self.Nt = timesteps
        self.cfl = cfl

        self.M = self.Nxi * self.Neta

        # Order of accuracy for the derivative matrices of the first and second order
        self.firstderivativeOrder = "6thOrder"

        self.v_x = 0.5 * np.ones(self.Nt)
        self.v_y = np.zeros(self.Nt)
        self.C = 1.0

        self.v_x_target = self.v_x
        self.v_y_target = self.v_y
        self.v_x_target[tilt_from:] = 1.0

    def Grid(self):
        self.X = np.arange(1, self.Nxi + 1) * self.Lxi / self.Nxi
        self.dx = self.X[1] - self.X[0]

        if self.Neta == 1:
            self.Y = 0
            self.dy = 0
        else:
            self.Y = np.arange(1, self.Neta + 1) * self.Leta / self.Neta
            self.dy = self.Y[1] - self.Y[0]

        dt = (np.sqrt(self.dx ** 2 + self.dy ** 2)) * self.cfl / self.C
        self.t = dt * np.arange(self.Nt)
        self.dt = self.t[1] - self.t[0]

        print('dt = ', dt)
        print('Final time : ', self.t[-1])

    def IC_primal(self):
        if self.Neta == 1:
            q = np.exp(-((self.X - self.Lxi / 12) ** 2) / 7)

        q = np.reshape(q, newshape=self.NN, order="F")

        return q

    def RHS_primal(self, q, f, A, psi):

        qdot = A.dot(q) + psi @ f

        return qdot

    def TI_primal(self, q, f0=None, A=None, psi=None):
        # Time loop
        qs = np.zeros((self.Nxi * self.Neta, self.Nt))
        qs[:, 0] = q

        for n in range(1, self.Nt):
            t = qs[:, n - 1]
            tt = f0[:, n - 1]
            qs[:, n] = rk4(self.RHS_primal, t[:, np.newaxis], tt[:, np.newaxis], self.dt, A, psi).squeeze()

        return qs

    def IC_adjoint(self):
        if self.Neta == 1:
            q_adj = np.zeros_like(self.X)

        q_adj = np.reshape(q_adj, newshape=self.NN, order="F")

        return q_adj

    def RHS_adjoint(self, q_adj, f, q, q_tar, A):

        q_adj_dot = - A.dot(q_adj) - (q - q_tar)

        return q_adj_dot

    def TI_adjoint(self, q0_adj, f0, qs, qs_target, A):
        # Time loop
        qs_adj = np.zeros((self.Nxi * self.Neta, self.Nt))
        qs_adj[:, -1] = q0_adj

        for n in range(1, self.Nt):
            t = qs_adj[:, -n]
            tt = qs[:, -n]
            ttt = qs_target[:, -n]
            qs_adj[:, -(n + 1)] = rk4(self.RHS_adjoint, t[:, np.newaxis], f0[:, -n], -self.dt, tt[:, np.newaxis],
                                      ttt[:, np.newaxis], A).squeeze()

        return qs_adj

    def RHS_primal_target(self, q, f, Mat, v_x, v_y):

        DT = v_x * Mat.Grad_Xi_kron + v_y * Mat.Grad_Eta_kron
        qdot = - DT.dot(q)

        return qdot

    def TI_primal_target(self, q, Mat, f0=None):
        # Time loop
        qs = np.zeros((self.Nxi * self.Neta, self.Nt))
        qs[:, 0] = q

        for n in range(1, self.Nt):
            qs[:, n] = rk4(self.RHS_primal_target, qs[:, n - 1], f0[:, n - 1], self.dt, Mat, self.v_x_target[n - 1],
                           self.v_y_target[n - 1])

        return qs

    ##################################################  FRTO POD  ######################################################
    def IC_primal_PODG_FRTO(self, V, q0):

        return V.transpose() @ q0

    def RHS_primal_PODG_FRTO(self, a, f, Ar_p, psir_p):

        return Ar_p @ a + psir_p @ f

    def TI_primal_PODG_FRTO(self, a, f0, Ar_p, psir_p):
        # Time loop
        as_ = np.zeros((a.shape[0], self.Nt))
        as_[:, 0] = a

        for n in range(1, self.Nt):
            as_[:, n] = rk4(self.RHS_primal_PODG_FRTO, as_[:, n - 1], f0[:, n - 1], self.dt, Ar_p, psir_p)

        return as_

    def mat_primal_PODG_FRTO(self, A_p, V, psi):

        V_T = V.transpose()

        return (V_T @ A_p) @ V, V_T @ psi

    def IC_adjoint_PODG_FRTO(self, V, q0_adj):

        return V.transpose() @ q0_adj

    def RHS_adjoint_PODG_FRTO(self, a_adj, f, a, Tarr_a, Ar_a):

        return - (Ar_a.dot(a_adj) + (a - Tarr_a))

    def TI_adjoint_PODG_FRTO(self, at_adj, f0, as_, Ar_a, Tarr_a):
        # Time loop
        as_adj = np.zeros((at_adj.shape[0], self.Nt))
        as_adj[:, -1] = at_adj

        for n in range(1, self.Nt):
            as_adj[:, -(n + 1)] = rk4(self.RHS_adjoint_PODG_FRTO, as_adj[:, -n], f0[:, -n], -self.dt,
                                      as_[:, -n],
                                      Tarr_a[:, -n],
                                      Ar_a)

        return as_adj

    def mat_adjoint_PODG_FRTO(self, V, A_a, qs_target):
        V_T = V.transpose()
        return (V_T @ A_a) @ V, V_T @ qs_target

    ############################################### FRTO sPOD ############################################
    def IC_primal_sPODG_FRTO(self, q0, ds, Vd):
        z = 0
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], z)
        V = weight * Vd[intervalIdx] + (1 - weight) * Vd[intervalIdx + 1]
        a = V.transpose() @ q0
        # Initialize the shifts with zero for online phase
        a = np.concatenate((a, np.asarray([z])))

        return a

    def mat_primal_sPODG_FRTO(self, T_delta, V_p, A_p, psi, D, samples):

        # Construct V_delta and W_delta matrix
        V_delta_primal, W_delta_primal = make_V_W_delta(V_p, T_delta, D, samples)

        # Construct LHS matrix
        LHS_matrix = LHS_offline_primal_FRTO(V_delta_primal, W_delta_primal)

        # Construct RHS matrix
        RHS_matrix = RHS_offline_primal_FRTO(V_delta_primal, W_delta_primal, A_p)

        # Construct the control matrix
        C_matrix = Control_offline_primal_FRTO(V_delta_primal, W_delta_primal, psi, D)

        # Construct target matrix components for the adjoint equation
        Tar_matrix = Target_offline_adjoint_FRTO(D, V_delta_primal)

        return V_delta_primal, W_delta_primal, LHS_matrix, RHS_matrix, C_matrix, Tar_matrix

    def RHS_primal_sPODG_FRTO(self, a, f, lhs, rhs, c, ds):

        # Compute the interpolation weight and the interval in which the shift lies
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], -a[-1])

        # Assemble the dynamic matrix D(a)
        Da = make_Da(a)

        # Prepare the LHS side of the matrix using D(a)
        M = LHS_online_primal_FRTO(lhs, Da)

        # Prepare the RHS side of the matrix using D(a)
        A = RHS_online_primal_FRTO(rhs, Da)

        # Prepare the online control matrix
        C = Control_online_primal_FRTO(f, c, Da, intervalIdx, weight)

        return np.linalg.solve(M, A @ a + C)

    def TI_primal_sPODG_FRTO(self, lhs, rhs, c, a, f0, delta_s):
        # Time loop
        as_ = np.zeros((a.shape[0], self.Nt))
        as_dot = np.zeros((4, a.shape[0], self.Nt))

        as_[:, 0] = a
        for n in range(1, self.Nt):
            as_[:, n], a_dot = rk4_(self.RHS_primal_sPODG_FRTO, as_[:, n - 1], f0[:, n - 1], self.dt, lhs, rhs, c,
                                    delta_s)
            as_dot[0, :, n - 1] = a_dot[0]
            as_dot[1, :, n - 1] = a_dot[1]
            as_dot[2, :, n - 1] = a_dot[2]
            as_dot[3, :, n - 1] = a_dot[3]

        _, a_dot = rk4_(self.RHS_primal_sPODG_FRTO, as_[:, -1], f0[:, -1], self.dt, lhs, rhs, c, delta_s)
        as_dot[0, :, -1] = a_dot[0]
        as_dot[1, :, -1] = a_dot[1]
        as_dot[2, :, -1] = a_dot[2]
        as_dot[3, :, -1] = a_dot[3]

        return as_, as_dot

    def IC_adjoint_sPODG_FRTO(self, q0_adj, ds, Vd):
        z = 0
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], z)
        V = weight * Vd[intervalIdx] + (1 - weight) * Vd[intervalIdx + 1]
        a = V.transpose() @ q0_adj
        # Initialize the shifts with zero for online phase
        a = np.concatenate((a, np.asarray([z])))

        return a

    def RHS_adjoint_sPODG_FRTO(self, a, f, a_dot, z_dot, a_, Vdp, Wdp, lhsp, rhsp, cp, Tp, qs_target, ds, Dfd, psi):

        # Compute the interpolation weight and the interval in which the shift lies
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], -a_[-1])

        # Assemble the dynamic matrix D(a_)
        Da = make_Da(a_)

        # Prepare the LHS side of the matrix using D(a_)
        M = LHS_online_adjoint_FRTO(lhsp, Da)

        # Prepare the RHS side of the matrix using D(a_)
        A = RHS_online_adjoint_FRTO(rhsp, lhsp, cp, psi, a_dot, z_dot, Da, a_, Dfd, Vdp, Wdp, f, intervalIdx, weight)

        # Prepare the online control matrix
        C = Target_online_adjoint_FRTO(a_, Dfd, Vdp, qs_target, Tp, intervalIdx, weight)

        return np.linalg.solve(M, -A @ a - C)

    def TI_adjoint_sPODG_FRTO(self, at_adj, f0, as_, lhsp, rhsp, cp, Tp, Vdp, Wdp, qs_target, Dfd, A, psi, as_dot, delta_s):
        as_adj = np.zeros((at_adj.shape[0], self.Nt))
        as_adj[:, -1] = at_adj

        for n in range(1, self.Nt):
            a_dot = [np.squeeze(as_dot[0, :-1, -n]),
                     np.squeeze(as_dot[1, :-1, -n]),
                     np.squeeze(as_dot[2, :-1, -n]),
                     np.squeeze(as_dot[3, :-1, -n])]
            z_dot = [np.squeeze(as_dot[0, -1:, -n]),
                     np.squeeze(as_dot[1, -1:, -n]),
                     np.squeeze(as_dot[2, -1:, -n]),
                     np.squeeze(as_dot[3, -1:, -n])]
            as_adj[:, -(n + 1)] = rk4__(self.RHS_adjoint_sPODG_FRTO, as_adj[:, -n], f0[:, -n],
                                        -self.dt, a_dot, z_dot, as_[:, -n], Vdp, Wdp, lhsp, rhsp, cp, Tp, qs_target[:, -n],
                                        delta_s, Dfd, psi)

        return as_adj

    ################################## FRTO sPOD (new cost functional) #########################################
    def RHS_adjoint_sPODG_FRTO_newcost(self, a, f, a_dot, z_dot, a_, Vdp, Wdp, lhsp, rhsp, cp, var_target, ds, Dfd, psi):

        # Compute the interpolation weight and the interval in which the shift lies
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], -a_[-1])

        # Assemble the dynamic matrix D(a_)
        Da = make_Da(a_)

        # Prepare the LHS side of the matrix using D(a_)
        M = LHS_online_adjoint_FRTO(lhsp, Da)

        # Prepare the RHS side of the matrix using D(a_)
        A = RHS_online_adjoint_FRTO(rhsp, lhsp, cp, psi, a_dot, z_dot, Da, a_, Dfd, Vdp, Wdp, f, intervalIdx, weight)

        # Prepare the online control matrix
        C = Target_online_adjoint_FRTO_newcost(a_, var_target)

        return np.linalg.solve(M, -A @ a - C)

    def TI_adjoint_sPODG_FRTO_newcost(self, at_adj, f0, as_, lhsp, rhsp, cp, Vdp, Wdp, var_target, Dfd, A, psi, as_dot, delta_s):
        # Time loop
        as_adj = np.zeros((at_adj.shape[0], self.Nt))
        as_adj[:, -1] = at_adj

        for n in range(1, self.Nt):
            a_dot = [np.squeeze(as_dot[0, :-1, -n]),
                     np.squeeze(as_dot[1, :-1, -n]),
                     np.squeeze(as_dot[2, :-1, -n]),
                     np.squeeze(as_dot[3, :-1, -n])]
            z_dot = [np.squeeze(as_dot[0, -1:, -n]),
                     np.squeeze(as_dot[1, -1:, -n]),
                     np.squeeze(as_dot[2, -1:, -n]),
                     np.squeeze(as_dot[3, -1:, -n])]
            as_adj[:, -(n + 1)] = rk4__(self.RHS_adjoint_sPODG_FRTO_newcost, as_adj[:, -n], f0[:, -n],
                                        -self.dt, a_dot, z_dot, as_[:, -n], Vdp, Wdp, lhsp, rhsp, cp, var_target[:, -n],
                                        delta_s, Dfd, psi)

        return as_adj

    ############################################### FOTR POD ############################################
    def IC_primal_PODG_FOTR(self, V_p, q0):

        return V_p.transpose() @ q0

    def RHS_primal_PODG_FOTR(self, a, f, Ar_p, psir_p):

        return Ar_p.dot(a) + psir_p.dot(f)

    def TI_primal_PODG_FOTR(self, a, f0, Ar_p, psir_p):
        # Time loop
        as_ = np.zeros((a.shape[0], self.Nt))
        as_[:, 0] = a

        for n in range(1, self.Nt):
            as_[:, n] = rk4(self.RHS_primal_PODG_FOTR, as_[:, n - 1], f0[:, n - 1], self.dt, Ar_p, psir_p)

        return as_

    def mat_primal_PODG_FOTR(self, A_p, V_p, psi):

        V_pT = V_p.transpose()

        return (V_pT @ A_p) @ V_p, V_pT @ psi

    def IC_adjoint_PODG_FOTR(self, V_a, q0_adj):

        return V_a.transpose() @ q0_adj

    def RHS_adjoint_PODG_FOTR(self, a_adj, f, a, Tarr_a, Ar_a, Tr_a):

        return - (Ar_a.dot(a_adj) + (Tr_a.dot(a) - Tarr_a))

    def TI_adjoint_PODG_FOTR(self, at_adj, f0, as_, Ar_a, Tr_a, Tarr_a):
        # Time loop
        as_adj = np.zeros((at_adj.shape[0], self.Nt))
        as_adj[:, -1] = at_adj

        for n in range(1, self.Nt):
            as_adj[:, -(n + 1)] = rk4(self.RHS_adjoint_PODG_FOTR, as_adj[:, -n], f0[:, -n], -self.dt,
                                      as_[:, -n],
                                      Tarr_a[:, -n],
                                      Ar_a, Tr_a)

        return as_adj

    def mat_adjoint_PODG_FOTR(self, V_a, A_p, V_p, q_target, psi):
        V_aT = V_a.transpose()
        return (V_aT @ A_p.transpose()) @ V_a, V_aT @ V_p, V_aT @ q_target, V_aT @ psi



    ######################################### FOTR sPOD  #############################################
    def IC_primal_sPODG_FOTR(self, q0, ds, Vd):
        z = 0
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], z)
        V = weight * Vd[intervalIdx] + (1 - weight) * Vd[intervalIdx + 1]
        a = V.transpose() @ q0
        # Initialize the shifts with zero for online phase
        a = np.concatenate((a, np.asarray([z])))

        return a

    def mat_primal_sPODG_FOTR(self, T_delta, V_p, A_p, psi, D, samples):

        # Construct V_delta and W_delta matrix
        V_delta_primal, W_delta_primal = make_V_W_delta(V_p, T_delta, D, samples)

        # Construct LHS matrix
        LHS_matrix = LHS_offline_primal_FOTR(V_delta_primal, W_delta_primal)

        # Construct RHS matrix
        RHS_matrix = RHS_offline_primal_FOTR(V_delta_primal, W_delta_primal, A_p)

        # Construct the control matrix
        C_matrix = Control_offline_primal_FOTR(V_delta_primal, W_delta_primal, psi)

        return V_delta_primal, W_delta_primal, LHS_matrix, RHS_matrix, C_matrix

    def RHS_primal_sPODG_FOTR(self, a, f, lhs, rhs, c, ds):

        # Compute the interpolation weight and the interval in which the shift lies
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], -a[-1])

        # Assemble the dynamic matrix D(a)
        Da = make_Da(a)

        # Prepare the LHS side of the matrix using D(a)
        M = LHS_online_primal_FOTR(lhs, Da)

        # Prepare the RHS side of the matrix using D(a)
        A = RHS_online_primal_FOTR(rhs, Da)

        # Prepare the online control matrix
        C = Control_online_primal_FOTR(f, c, Da, intervalIdx, weight)

        return np.linalg.solve(M, A @ a + C)

    def TI_primal_sPODG_FOTR(self, lhs, rhs, c, a, f0, delta_s):
        # Time loop
        as_ = np.zeros((a.shape[0], self.Nt))
        as_[:, 0] = a

        for n in range(1, self.Nt):
            as_[:, n] = rk4(self.RHS_primal_sPODG_FOTR, as_[:, n - 1], f0[:, n - 1], self.dt, lhs, rhs, c, delta_s)

        return as_

    def IC_adjoint_sPODG_FOTR(self, Nm_adjoint, a_):
        # Initialize the shift adjoint with zero for online phase
        a = np.concatenate((np.zeros(Nm_adjoint), np.asarray([a_[-1, -1]])))

        return a

    def mat_adjoint_sPODG_FOTR(self, T_delta, V_delta_primal, V_a, A_a, D, psi, samples):

        # Construct V_delta and W_delta matrix
        V_delta_adjoint, W_delta_adjoint = make_V_W_delta(V_a, T_delta, D, samples)

        # Construct LHS matrix
        LHS_matrix = LHS_offline_primal_FOTR(V_delta_adjoint, W_delta_adjoint)

        # Construct RHS matrix
        RHS_matrix = RHS_offline_primal_FOTR(V_delta_adjoint, W_delta_adjoint, A_a)

        # Construct the target precomputed terms
        Tar_matrix = Target_offline_adjoint_FOTR(V_delta_primal, V_delta_adjoint, W_delta_adjoint)

        # Construct the control update precomputation matrix
        C_matrix = Control_update_online_adjoint_FOTR(V_delta_adjoint, psi)

        return V_delta_adjoint, W_delta_adjoint, LHS_matrix, RHS_matrix, Tar_matrix, C_matrix

    def RHS_adjoint_sPODG_FOTR(self, a, f, a_, lhs, rhs, T_a, Vda, Wda, qs_target, ds):

        # Compute the interpolation weight and the interval in which the shift lies
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], -a_[-1])

        # Assemble the dynamic matrix D(a)
        Da = make_Da(a)

        # Prepare the LHS side of the matrix using D(a)
        M = LHS_online_primal_FOTR(lhs, Da)

        # Prepare the RHS side of the matrix using D(a)
        A = RHS_online_primal_FOTR(rhs, Da)

        # Prepare the online control matrix
        C = Target_online_adjoint_FOTR(T_a, Vda, Wda, qs_target, a_[:-1], Da, intervalIdx, weight)

        if np.linalg.cond(M, p='fro') == np.inf:
            res = np.linalg.solve(M.T.dot(M) + 1e-14 * np.identity(M.shape[1]), M.T.dot(-A @ a - C))
        else:
            res = np.linalg.solve(M, -A @ a - C)

        return res

    def TI_adjoint_sPODG_FOTR(self, lhs, rhs, T_a, Vda, Wda, qs_target, at_adj, f0, as_, delta_s):
        # Time loop
        as_adj = np.zeros((at_adj.shape[0], self.Nt))
        as_adj[:, -1] = at_adj

        for n in range(1, self.Nt):
            as_adj[:, -(n + 1)] = rk4(self.RHS_adjoint_sPODG_FOTR, as_adj[:, -n], f0[:, -n],
                                      -self.dt, as_[:, -n], lhs, rhs, T_a, Vda, Wda,
                                      qs_target[:, -n], delta_s)

        return as_adj

