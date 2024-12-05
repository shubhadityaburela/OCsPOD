from Helper import RHS_PODG_FOTR_solve
from Helper_sPODG import *
from rk4 import rk4, rk4_adj, rk4_rpr, rk4_radj, rk4_rpr_dot, rk4_rpr_, rk4_radj_


class advection:
    def __init__(self, Nxi: int, Neta: int, timesteps: int, cfl: float, tilt_from: int, v_x: float, v_x_t: float,
                 variance: float, offset: float) -> None:
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

        self.v_x = v_x * np.ones(self.Nt)
        self.v_y = np.zeros(self.Nt)
        self.C = 1.0

        self.v_x_target = self.v_x
        self.v_y_target = self.v_y
        self.v_x_target[tilt_from:] = v_x_t

        self.variance = variance  # Variance of the gaussian for the initial condition
        self.offset = offset  # Offset from where the wave starts

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
            q = np.exp(-((self.X - self.Lxi / self.offset) ** 2) / self.variance)

        q = np.reshape(q, newshape=self.NN, order="F")

        return q

    def RHS_primal(self, q, f, A, psi):

        return A @ q + psi @ f

    def TI_primal(self, q, f0=None, A=None, psi=None):
        # Time loop
        qs = np.zeros((self.Nxi * self.Neta, self.Nt))
        qs[:, 0] = q
        for n in range(1, self.Nt):
            qs[:, n] = rk4(self.RHS_primal, qs[:, n - 1], f0[:, n - 1], f0[:, n], self.dt, A, psi)
        return qs

    def IC_adjoint(self):
        q_adj = np.zeros_like(self.X)
        q_adj = np.reshape(q_adj, newshape=self.NN, order="F")

        return q_adj

    def RHS_adjoint(self, q_adj, q, q_tar, A, CTC):

        return - A @ q_adj - CTC @ (q - q_tar)

    def TI_adjoint(self, q0_adj, qs, qs_target, A, CTC):
        # Time loop
        qs_adj = np.zeros((self.Nxi * self.Neta, self.Nt))
        qs_adj[:, -1] = q0_adj

        for n in range(1, self.Nt):
            qs_adj[:, -(n + 1)] = rk4_adj(self.RHS_adjoint, qs_adj[:, -n], qs[:, -n], qs[:, -(n + 1)],
                                          qs_target[:, -n], qs_target[:, -(n + 1)], -self.dt, A, CTC)
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
            qs[:, n] = rk4(self.RHS_primal_target, qs[:, n - 1], f0[:, n - 1], f0[:, n], self.dt, Mat,
                           self.v_x_target[n - 1],
                           self.v_y_target[n - 1])

        return qs

    ############################################### FOTR POD ############################################
    def IC_primal_PODG_FOTR(self, V_p, q0):

        return V_p.T @ q0

    def RHS_primal_PODG_FOTR(self, a, f, Ar_p, psir_p):

        return RHS_PODG_FOTR_solve(Ar_p, a, psir_p, f)

    def TI_primal_PODG_FOTR(self, a, f0, Ar_p, psir_p):
        # Time loop
        as_ = np.zeros((a.shape[0], self.Nt))
        as_[:, 0] = a

        for n in range(1, self.Nt):
            as_[:, n] = rk4(self.RHS_primal_PODG_FOTR, as_[:, n - 1], f0[:, n - 1], f0[:, n], self.dt, Ar_p, psir_p)

        return as_

    def mat_primal_PODG_FOTR(self, A_p, V_p, psi):

        V_pT = V_p.T

        return (V_pT @ A_p) @ V_p, V_pT @ psi

    ######################################### FOTR sPOD  #############################################
    def IC_primal_sPODG_FOTR(self, q0, V):
        z = 0
        a = V.transpose() @ q0
        # Initialize the shifts with zero for online phase
        a = np.concatenate((a, np.asarray([z])))

        return a

    def mat_primal_sPODG_FOTR(self, T_delta, V_p, A_p, psi, D, samples, modes):

        # Construct V_delta and W_delta matrix
        V_delta_primal, W_delta_primal = make_V_W_delta(V_p, T_delta, D, samples, self.Nxi, modes)

        # Construct LHS matrix
        LHS_matrix = LHS_offline_primal_FOTR(V_delta_primal, W_delta_primal, modes)

        # Construct RHS matrix
        RHS_matrix = RHS_offline_primal_FOTR(V_delta_primal, W_delta_primal, A_p, modes)

        # Construct the control matrix
        C_matrix = Control_offline_primal_FOTR(V_delta_primal, W_delta_primal, psi, samples, modes)

        return V_delta_primal, W_delta_primal, LHS_matrix, RHS_matrix, C_matrix

    def RHS_primal_sPODG_FOTR(self, a, f, lhs, rhs, c, ds, modes):

        # Prepare the online primal matrices
        M, A, intervalIdx, weight = Matrices_online_primal_FOTR(lhs, rhs, c, f, a, ds, modes)

        # Solve the linear system of equations
        X = solve_lin_system(M, A)

        return X, intervalIdx, weight

    def TI_primal_sPODG_FOTR(self, lhs, rhs, c, a, f0, delta_s, modes):
        # Time loop
        as_ = np.zeros((a.shape[0], self.Nt), order="F")
        f0 = np.asfortranarray(f0)
        IntIds = np.zeros(self.Nt, dtype=np.int32)
        weights = np.zeros(self.Nt)

        as_[:, 0] = a

        for n in range(1, self.Nt):
            as_[:, n], IntIds[n - 1], weights[n - 1] = rk4_rpr(self.RHS_primal_sPODG_FOTR, as_[:, n - 1], f0[:, n - 1],
                                                               f0[:, n], self.dt, lhs, rhs, c, delta_s, modes)

        IntIds[-1], weights[-1] = findIntervalAndGiveInterpolationWeight_1D(delta_s[2], -as_[-1, -1])

        return as_, IntIds, weights

    def TI_primal_sPODG_FOTR_dot(self, lhs, rhs, c, a, f0, delta_s, modes):
        # Time loop
        as_ = np.zeros((a.shape[0], self.Nt), order="F")
        as_dot = np.zeros((a.shape[0], self.Nt - 1), order="F")
        f0 = np.asfortranarray(f0)
        IntIds = np.zeros(self.Nt, dtype=np.int32)
        weights = np.zeros(self.Nt)

        as_[:, 0] = a

        for n in range(1, self.Nt):
            as_[:, n], IntIds[n - 1], weights[n - 1], as_dot[:, n - 1] = rk4_rpr_dot(self.RHS_primal_sPODG_FOTR,
                                                                                     as_[:, n - 1],
                                                                                     f0[:, n - 1],
                                                                                     f0[:, n], self.dt, lhs, rhs, c,
                                                                                     delta_s, modes)

        IntIds[-1], weights[-1] = findIntervalAndGiveInterpolationWeight_1D(delta_s[2], -as_[-1, -1])

        return as_, IntIds, weights, as_dot

    def IC_adjoint_sPODG_FOTR(self, Nm_a, z):
        a = np.concatenate((np.zeros(Nm_a), np.asarray([z])))
        return a

    def mat_adjoint_sPODG_FOTR(self, T_delta, V_a, A_a, D, V_delta_primal, CTC, samples, modes_a, modes_p):

        # Construct V_delta and W_delta matrix
        V_delta_adjoint, W_delta_adjoint = make_V_W_delta(V_a, T_delta, D, samples, self.Nxi, modes_a)

        # Construct LHS matrix
        LHS_matrix = LHS_offline_primal_FOTR(V_delta_adjoint, W_delta_adjoint, modes_a)

        # Construct RHS matrix
        RHS_matrix = RHS_offline_primal_FOTR(V_delta_adjoint, W_delta_adjoint, A_a, modes_a)

        # Construct the control matrix
        Tar_matrix_1, Tar_matrix_2 = Target_offline_adjoint_FOTR(V_delta_primal, V_delta_adjoint, W_delta_adjoint,
                                                                 CTC, samples, modes_a, modes_p, self.Nxi)

        return V_delta_adjoint, W_delta_adjoint, LHS_matrix, RHS_matrix, Tar_matrix_1, Tar_matrix_2

    def RHS_adjoint_sPODG_FOTR(self, as_adj, as_, qs_target, lhs, rhs, tar1, tar2, modes_a, modes_p, intId, weight):

        # Prepare the online adjoint matrices
        M, A = Matrices_online_adjoint_FOTR(lhs, rhs, tar1, tar2, qs_target, as_adj, as_,
                                            modes_a, modes_p, intId, weight)

        # Solve the linear system of equations
        if np.linalg.cond(M, p='fro') == np.inf:
            return solve_lin_system(M.T.dot(M) + 1e-14 * np.identity(M.shape[1]), M.T.dot(-A))
        else:
            return solve_lin_system(M, -A)

    def TI_adjoint_sPODG_FOTR(self, lhs, rhs, tar1, tar2, a_a, as_, qs_target,
                              modes_a, modes_p, intIds, weights):
        # Time loop
        as_adj = np.zeros((modes_a + 1, self.Nt), order="F")
        as_ = np.asfortranarray(as_)
        as_adj[:, -1] = a_a

        for n in range(1, self.Nt):
            as_adj[:, -(n + 1)] = rk4_radj(self.RHS_adjoint_sPODG_FOTR, as_adj[:, -n], as_[:, -n], as_[:, -(n + 1)],
                                           qs_target[:, -n], qs_target[:, -(n + 1)], -self.dt, lhs, rhs, tar1, tar2,
                                           modes_a, modes_p, intIds[-n], weights[-n])
        return as_adj

    ############################################ FRTO sPOD (New Cost)  #############################################
    def IC_primal_sPODG_FRTO(self, q0, V):
        z = 0
        a = V.transpose() @ q0
        # Initialize the shifts with zero for online phase
        a = np.concatenate((a, np.asarray([z])))

        return a

    def mat_primal_sPODG_FRTO(self, T_delta, V_p, A_p, psi, D, samples, modes):

        # Construct V_delta and W_delta matrix
        V_delta_primal, W_delta_primal, U_delta_primal = make_V_W_U_delta(V_p, T_delta, D, samples, self.Nxi, modes)

        # Construct LHS matrix
        LHS_matrix = LHS_offline_primal_FRTO(V_delta_primal, W_delta_primal, modes)

        # Construct RHS matrix
        RHS_matrix = RHS_offline_primal_FRTO(V_delta_primal, W_delta_primal, A_p, modes)

        # Construct the control matrix
        C_matrix = Control_offline_primal_FRTO(V_delta_primal, W_delta_primal, U_delta_primal, psi, samples, modes)

        return V_delta_primal, W_delta_primal, U_delta_primal, LHS_matrix, RHS_matrix, C_matrix

    def RHS_primal_sPODG_FRTO(self, a, f, lhs, rhs, c, ds, modes):

        # Prepare the online primal matrices
        M, A, intervalIdx, weight = Matrices_online_primal_FRTO(lhs, rhs, c, f, a, ds, modes)

        # Solve the linear system of equations
        X = solve_lin_system(M, A)

        return X, intervalIdx, weight

    def TI_primal_sPODG_FRTO(self, lhs, rhs, c, a, f0, delta_s, modes):
        # Time loop
        as_ = np.zeros((a.shape[0], self.Nt), order="F")
        as_dot = np.zeros((4, a.shape[0], self.Nt), order="F")
        f0 = np.asfortranarray(f0)
        IntIds = np.zeros(self.Nt, dtype=np.int32)
        weights = np.zeros(self.Nt)

        as_[:, 0] = a

        for n in range(1, self.Nt):
            as_[:, n], a_dot, IntIds[n - 1], weights[n - 1] = rk4_rpr_(self.RHS_primal_sPODG_FRTO, as_[:, n - 1],
                                                                       f0[:, n - 1],
                                                                       f0[:, n], self.dt, lhs, rhs, c, delta_s, modes)
            as_dot[..., n - 1] = a_dot

        _, a_dot, IntIds[-1], weights[-1] = rk4_rpr_(self.RHS_primal_sPODG_FRTO, as_[:, -1],
                                                     f0[:, -1],
                                                     f0[:, -1], self.dt, lhs, rhs, c, delta_s, modes)
        as_dot[..., -1] = a_dot

        return as_, as_dot, IntIds, weights

    def IC_adjoint_sPODG_FRTO(self, modes):
        z = 0
        # Initialize the shifts with zero for online phase
        a = np.concatenate((np.zeros(modes), np.asarray([z])))

        return a

    def RHS_adjoint_sPODG_FRTO(self, a, f, a_, a_target, a_dot, M1, M2, N, A1, A2, C, modes, intId, weight):

        # Prepare the online primal matrices
        M, A = Matrices_online_adjoint_FRTO_NC(M1, M2, N, A1, A2, C, f, a, a_, a_target, a_dot, modes, intId, weight)

        # Solve the linear system of equations
        X = solve_lin_system(M, -A)

        return X

    def TI_adjoint_sPODG_FRTO(self, at_adj, f0, a_, a_target, a_dot, lhsp, rhsp, C, modes, intIds, weights):
        # Time loop
        as_adj = np.zeros((at_adj.shape[0], self.Nt), order="F")
        as_adj[:, -1] = at_adj

        M1 = lhsp[0]
        N = lhsp[1]
        M2 = lhsp[2]
        A1 = rhsp[0]
        A2 = rhsp[1]

        for n in range(1, self.Nt):
            as_adj[:, -(n + 1)] = rk4_radj_(self.RHS_adjoint_sPODG_FRTO, as_adj[:, -n], f0[:, -n], f0[:, -(n + 1)],
                                            a_[:, -n], a_[:, -(n + 1)], a_target[:, -n], a_target[:, -(n + 1)],
                                            a_dot[..., -n], - self.dt, M1, M2, N, A1, A2, C, modes,
                                            intIds[-n], weights[-n])

        return as_adj
