from Costs import *
from FOM_solver import TI_primal
from Grads import *
from Helper import *
from PODG_solver import TI_primal_PODG_FOTR, TI_primal_PODG_FRTO
from sPODG_solver import TI_primal_sPODG_FOTR, TI_primal_sPODG_FRTO


def Update_Control_TWBT(f, q0, qs_target, mask, A_p, J_prev, omega_prev, dL_du, dL_du_norm_square, **kwargs):  # Two-way back tracking

    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = f - omega * dL_du
    qs = TI_primal(q0, f_new, A_p, mask, kwargs['Nx'], kwargs['Nt'], kwargs['dt'])
    J = Calc_Cost(qs, qs_target, f_new, kwargs['dx'], kwargs['dt'], kwargs['lamda'])
    dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
    if J < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            J_final = J
            omega = omega / beta
            f_new = f - omega * dL_du
            qs = TI_primal(q0, f_new, A_p, mask, kwargs['Nx'], kwargs['Nt'], kwargs['dt'])
            J = Calc_Cost(qs, qs_target, f_new, kwargs['dx'], kwargs['dt'], kwargs['lamda'])
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J >= dJ:
                print(
                    f"Armijo satisfied and the inner loop converged at the {n}th step with final omega = {omega_final}")
                return f_new_final, J_final, dL_du, omega_final, False
            if kwargs['verbose']: print(f"Armijo satisfied but omega too low! thus omega increased to = {omega}",
                                        f"at inner step={n}")
            n = n + 1
    else:  # If Armijo not satisfied
        k = 0
        while True:
            omega = beta * omega
            f_new = f - omega * dL_du
            qs = TI_primal(q0, f_new, A_p, mask, kwargs['Nx'], kwargs['Nt'], kwargs['dt'])
            J = Calc_Cost(qs, qs_target, f_new, kwargs['dx'], kwargs['dt'], kwargs['lamda'])
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J < dJ:  # If Armijo satisfied
                if omega < kwargs['omega_cutoff']:
                    print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                    return f_new, J, dL_du, omega, True
                else:
                    print(f"Armijo converged on the {k}th step with final omega = {omega}")
                    return f_new, J, dL_du, omega, False
            if omega < kwargs['omega_cutoff']:
                print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                return f_new, J, dL_du, omega, True

            if kwargs['verbose']: print(f"Armijo not satisfied thus omega decreased to = {omega}", f"at step={k}")
            k = k + 1


def Update_Control_BB(fNew, dL_du_New, omega):  # Barzilai Borwein
    f_bb_new = fNew - omega * dL_du_New

    return f_bb_new



def Update_Control_PODG_FOTR_FA_TWBT(f, a0_primal, qs_adj, qs_target, V_p, Ar_p, psir_p, mask, J_prev, omega_prev,
                                     **kwargs):
    dL_du = Calc_Grad(mask, f, qs_adj, kwargs['lamda'])
    dL_du_norm_square = L2norm_ROM(dL_du, kwargs['dt'])
    dL_du_norm = np.sqrt(dL_du_norm_square)

    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = f - omega * dL_du
    as_ = TI_primal_PODG_FOTR(a0_primal, f_new, Ar_p, psir_p, kwargs['Nt'], kwargs['dt'])
    J = Calc_Cost_PODG(V_p, as_, qs_target, f_new, kwargs['dx'], kwargs['dt'], kwargs['lamda'])
    dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
    if J < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            J_final = J
            omega = omega / beta
            f_new = f - omega * dL_du
            as_ = TI_primal_PODG_FOTR(a0_primal, f_new, Ar_p, psir_p, kwargs['Nt'], kwargs['dt'])
            J = Calc_Cost_PODG(V_p, as_, qs_target, f_new, kwargs['dx'], kwargs['dt'], kwargs['lamda'])
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J >= dJ:
                print(
                    f"Armijo satisfied and the inner loop converged at the {n}th step with final omega = {omega_final}")
                return f_new_final, J_final, dL_du, dL_du_norm, omega_final, False
            if kwargs['verbose']: print(f"Armijo satisfied but omega too low! thus omega increased to = {omega}",
                                        f"at inner step={n}")
            n = n + 1
    else:  # If Armijo not satisfied
        k = 0
        while True:
            omega = beta * omega
            f_new = f - omega * dL_du
            as_ = TI_primal_PODG_FOTR(a0_primal, f_new, Ar_p, psir_p, kwargs['Nt'], kwargs['dt'])
            J = Calc_Cost_PODG(V_p, as_, qs_target, f_new, kwargs['dx'], kwargs['dt'], kwargs['lamda'])
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J < dJ:  # If Armijo satisfied
                if omega < kwargs['omega_cutoff']:
                    print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                    return f_new, J, dL_du, dL_du_norm, omega, True
                else:
                    print(f"Armijo converged on the {k}th step with final omega = {omega}")
                    return f_new, J, dL_du, dL_du_norm, omega, False
            if omega < kwargs['omega_cutoff']:
                print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                return f_new, J, dL_du, dL_du_norm, omega, True

            if kwargs['verbose']: print(f"Armijo not satisfied thus omega decreased to = {omega}", f"at step={k}")
            k = k + 1


def Update_Control_PODG_FOTR_RA_TWBT(f, a0_primal, qs_target, V_p, Ar_p, psir_p, J_prev,
                                     omega_prev, dL_du, dL_du_norm_square, **kwargs):
    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = f - omega * dL_du
    as_ = TI_primal_PODG_FOTR(a0_primal, f_new, Ar_p, psir_p, kwargs['Nt'], kwargs['dt'])
    J = Calc_Cost_PODG(V_p, as_, qs_target, f_new, kwargs['dx'], kwargs['dt'], kwargs['lamda'])
    dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
    if J < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            J_final = J
            omega = omega / beta
            f_new = f - omega * dL_du
            as_ = TI_primal_PODG_FOTR(a0_primal, f_new, Ar_p, psir_p, kwargs['Nt'], kwargs['dt'])
            J = Calc_Cost_PODG(V_p, as_, qs_target, f_new, kwargs['dx'], kwargs['dt'], kwargs['lamda'])
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J >= dJ:
                print(
                    f"Armijo satisfied and the inner loop converged at the {n}th step with final omega = {omega_final}")
                return f_new_final, J_final, omega_final, False
            if kwargs['verbose']: print(f"Armijo satisfied but omega too low! thus omega increased to = {omega}",
                                        f"at inner step={n}")
            n = n + 1
    else:  # If Armijo not satisfied
        k = 0
        while True:
            omega = beta * omega
            f_new = f - omega * dL_du
            as_ = TI_primal_PODG_FOTR(a0_primal, f_new, Ar_p, psir_p, kwargs['Nt'], kwargs['dt'])
            J = Calc_Cost_PODG(V_p, as_, qs_target, f_new, kwargs['dx'], kwargs['dt'], kwargs['lamda'])
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J < dJ:  # If Armijo satisfied
                if omega < kwargs['omega_cutoff']:
                    print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                    return f_new, J, omega, True
                else:
                    print(f"Armijo converged on the {k}th step with final omega = {omega}")
                    return f_new, J, omega, False
            if omega < kwargs['omega_cutoff']:
                print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                return f_new, J, omega, True

            if kwargs['verbose']: print(f"Armijo not satisfied thus omega decreased to = {omega}", f"at step={k}")
            k = k + 1


def Update_Control_sPODG_FOTR_FA_TWBT(f, lhs, rhs, c, a0_primal, qs_adj, qs_target, delta_s, Vdp, mask, J_prev,
                                      omega_prev, modes, **kwargs):
    dL_du = Calc_Grad(mask, f, qs_adj, kwargs['lamda'])
    dL_du_norm_square = L2norm_ROM(dL_du, kwargs['dt'])
    dL_du_norm = np.sqrt(dL_du_norm_square)

    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = f - omega * dL_du
    as_, intIds, weights = TI_primal_sPODG_FOTR(lhs, rhs, c, a0_primal, f_new, delta_s, modes, kwargs['Nt'],
                                                kwargs['dt'])
    J, qq = Calc_Cost_sPODG(Vdp, as_[:-1], qs_target, f_new, intIds, weights,
                            kwargs['dx'], kwargs['dt'], kwargs['lamda'])
    dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
    if J < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            J_final = J
            qq_final = np.copy(qq)
            omega = omega / beta
            f_new = f - omega * dL_du
            as_, intIds_n, weights_n = TI_primal_sPODG_FOTR(lhs, rhs, c, a0_primal, f_new, delta_s, modes,
                                                            kwargs['Nt'], kwargs['dt'])
            J, qq = Calc_Cost_sPODG(Vdp, as_[:-1], qs_target, f_new, intIds_n, weights_n,
                                    kwargs['dx'], kwargs['dt'], kwargs['lamda'])
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J >= dJ:
                print(
                    f"Armijo satisfied and the inner loop converged at the {n}th step with final omega = {omega_final}")
                return f_new_final, J_final, dL_du, dL_du_norm, omega_final, qq_final, False
            if kwargs['verbose']: print(f"Armijo satisfied but omega too low! thus omega increased to = {omega}",
                                        f"at inner step={n}")
            n = n + 1
    else:  # If Armijo not satisfied
        k = 0
        while True:
            omega = beta * omega
            f_new = f - omega * dL_du
            as_, intIds_k, weights_k = TI_primal_sPODG_FOTR(lhs, rhs, c, a0_primal, f_new, delta_s, modes,
                                                            kwargs['Nt'], kwargs['dt'])
            J, qq = Calc_Cost_sPODG(Vdp, as_[:-1], qs_target, f_new, intIds_k, weights_k,
                                    kwargs['dx'], kwargs['dt'], kwargs['lamda'])
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J < dJ:  # If Armijo satisfied
                if omega < kwargs['omega_cutoff']:
                    print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                    return f_new, J, dL_du, dL_du_norm, omega, qq, True
                else:
                    print(f"Armijo converged on the {k}th step with final omega = {omega}")
                    return f_new, J, dL_du, dL_du_norm, omega, qq, False
            if omega < kwargs['omega_cutoff']:
                print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                return f_new, J, dL_du, dL_du_norm, omega, qq, True

            if kwargs['verbose']: print(f"Armijo not satisfied thus omega decreased to = {omega}", f"at step={k}")
            k = k + 1


def Update_Control_sPODG_FOTR_RA_TWBT(f, lhs, rhs, c, a0_primal, qs_target, delta_s, Vdp, J_prev,
                                      omega_prev, modes, dL_du, dL_du_norm_square, **kwargs):

    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = f - omega * dL_du
    as_, intIds, weights = TI_primal_sPODG_FOTR(lhs, rhs, c, a0_primal, f_new, delta_s, modes, kwargs['Nt'],
                                                kwargs['dt'])
    J, qq = Calc_Cost_sPODG(Vdp, as_[:-1], qs_target, f_new, intIds, weights,
                            kwargs['dx'], kwargs['dt'], kwargs['lamda'])
    dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
    if J < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            J_final = J
            qq_final = np.copy(qq)
            omega = omega / beta
            f_new = f - omega * dL_du
            as_, intIds_n, weights_n = TI_primal_sPODG_FOTR(lhs, rhs, c, a0_primal, f_new, delta_s, modes,
                                                            kwargs['Nt'], kwargs['dt'])
            J, qq = Calc_Cost_sPODG(Vdp, as_[:-1], qs_target, f_new, intIds_n, weights_n,
                                    kwargs['dx'], kwargs['dt'], kwargs['lamda'])
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J >= dJ:
                print(
                    f"Armijo satisfied and the inner loop converged at the {n}th step with final omega = {omega_final}")
                return f_new_final, J_final, omega_final, qq_final, False
            if kwargs['verbose']: print(f"Armijo satisfied but omega too low! thus omega increased to = {omega}",
                                        f"at inner step={n}")
            n = n + 1
    else:  # If Armijo not satisfied
        k = 0
        while True:
            omega = beta * omega
            f_new = f - omega * dL_du
            as_, intIds_k, weights_k = TI_primal_sPODG_FOTR(lhs, rhs, c, a0_primal, f_new, delta_s, modes,
                                                            kwargs['Nt'], kwargs['dt'])
            J, qq = Calc_Cost_sPODG(Vdp, as_[:-1], qs_target, f_new, intIds_k, weights_k,
                                    kwargs['dx'], kwargs['dt'], kwargs['lamda'])
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J < dJ:  # If Armijo satisfied
                if omega < kwargs['omega_cutoff']:
                    print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                    return f_new, J, omega, qq, True
                else:
                    print(f"Armijo converged on the {k}th step with final omega = {omega}")
                    return f_new, J, omega, qq, False
            if omega < kwargs['omega_cutoff']:
                print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                return f_new, J, omega, qq, True

            if kwargs['verbose']: print(f"Armijo not satisfied thus omega decreased to = {omega}", f"at step={k}")
            k = k + 1


def Update_Control_PODG_FOTR_RA_BB(fNew, dL_du_New, omega):  # Barzilai Borwein

    f_bb_new = fNew - omega * dL_du_New

    return f_bb_new


def Update_Control_sPODG_FOTR_RA_BB(fNew, dL_du_New, omega):  # Barzilai Borwein

    f_bb_new = fNew - omega * dL_du_New

    return f_bb_new


def Update_Control_sPODG_FRTO_TWBT(f, lhs_p, rhs_p, c_p, Vd_p, a_p, qs_target, delta_s, J_prev,
                                   omega_prev, modes, dL_du, dL_du_norm_square, **kwargs):
    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = f - omega * dL_du
    as_, _, intIds, weights = TI_primal_sPODG_FRTO(lhs_p, rhs_p, c_p, a_p, f_new, delta_s, modes=modes, Nt=kwargs['Nt'],
                                                   dt=kwargs['dt'])
    J, _ = Calc_Cost_sPODG(Vd_p, as_[:-1], qs_target, f_new, intIds, weights,
                           kwargs['dx'], kwargs['dt'], kwargs['lamda'])
    dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
    if J < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            J_final = J
            omega = omega / beta
            f_new = f - omega * dL_du
            as_, _, intIds_n, weights_n = TI_primal_sPODG_FRTO(lhs_p, rhs_p, c_p, a_p, f_new, delta_s, modes=modes,
                                                               Nt=kwargs['Nt'],
                                                               dt=kwargs['dt']
                                                               )
            J, _ = Calc_Cost_sPODG(Vd_p, as_[:-1], qs_target, f_new, intIds_n, weights_n,
                                   kwargs['dx'], kwargs['dt'], kwargs['lamda'])
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J >= dJ:
                print(
                    f"Armijo satisfied and the inner loop converged at the {n}th step with final omega = {omega_final}")
                return f_new_final, J_final, omega_final, False
            if kwargs['verbose']: print(f"Armijo satisfied but omega too low! thus omega increased to = {omega}",
                                        f"at inner step={n}")
            n = n + 1
    else:  # If Armijo not satisfied
        k = 0
        while True:
            omega = beta * omega
            f_new = f - omega * dL_du
            as_, _, intIds_k, weights_k = TI_primal_sPODG_FRTO(lhs_p, rhs_p, c_p, a_p, f_new, delta_s, modes=modes,
                                                               Nt=kwargs['Nt'],
                                                               dt=kwargs['dt']
                                                               )
            J, _ = Calc_Cost_sPODG(Vd_p, as_[:-1], qs_target, f_new, intIds_k, weights_k,
                                   kwargs['dx'], kwargs['dt'], kwargs['lamda'])
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J < dJ:  # If Armijo satisfied
                if omega < kwargs['omega_cutoff']:
                    print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                    return f_new, J, omega, True
                else:
                    print(f"Armijo converged on the {k}th step with final omega = {omega}")
                    return f_new, J, omega, False
            if omega < kwargs['omega_cutoff']:
                print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                return f_new, J, omega, True

            if kwargs['verbose']: print(f"Armijo not satisfied thus omega decreased to = {omega}", f"at step={k}")
            k = k + 1


def Update_Control_sPODG_FRTO_BB(fNew, dL_du_New, omega):  # Barzilai Borwein
    f_bb_new = fNew - omega * dL_du_New

    return f_bb_new


def Update_Control_PODG_FRTO_TWBT(f, Ar_p, psir_p, V, a_p, qs_target, J_prev,
                                  omega_prev, dL_du, dL_du_norm_square, **kwargs):

    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = f - omega * dL_du
    as_ = TI_primal_PODG_FRTO(a_p, f_new, Ar_p, psir_p, kwargs['Nt'], kwargs['dt'])
    J = Calc_Cost_PODG(V, as_, qs_target, f_new,
                       kwargs['dx'], kwargs['dt'], kwargs['lamda'])
    dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
    if J < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            J_final = J
            omega = omega / beta
            f_new = f - omega * dL_du
            as_ = TI_primal_PODG_FRTO(a_p, f_new, Ar_p, psir_p, kwargs['Nt'], kwargs['dt'])
            J = Calc_Cost_PODG(V, as_, qs_target, f_new,
                               kwargs['dx'], kwargs['dt'], kwargs['lamda'])
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J >= dJ:
                print(
                    f"Armijo satisfied and the inner loop converged at the {n}th step with final omega = {omega_final}")
                return f_new_final, J_final, omega_final, False
            if kwargs['verbose']: print(f"Armijo satisfied but omega too low! thus omega increased to = {omega}",
                                        f"at inner step={n}")
            n = n + 1
    else:  # If Armijo not satisfied
        k = 0
        while True:
            omega = beta * omega
            f_new = f - omega * dL_du
            as_ = TI_primal_PODG_FRTO(a_p, f_new, Ar_p, psir_p, kwargs['Nt'], kwargs['dt'])
            J = Calc_Cost_PODG(V, as_, qs_target, f_new,
                               kwargs['dx'], kwargs['dt'], kwargs['lamda'])
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J < dJ:  # If Armijo satisfied
                if omega < kwargs['omega_cutoff']:
                    print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                    return f_new, J, omega, True
                else:
                    print(f"Armijo converged on the {k}th step with final omega = {omega}")
                    return f_new, J, omega, False
            if omega < kwargs['omega_cutoff']:
                print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                return f_new, J, omega, True

            if kwargs['verbose']: print(f"Armijo not satisfied thus omega decreased to = {omega}", f"at step={k}")
            k = k + 1



def Update_Control_PODG_FRTO_BB(fNew, dL_du_New, omega):  # Barzilai Borwein

    f_bb_new = fNew - omega * dL_du_New

    return f_bb_new


def get_BB_step(fOld, fNew, dL_du_Old, dL_du_New, itr, **kwargs):
    alpha = BarzilaiBorwein(itr, kwargs['dt'], fNew, fOld, dL_du_New, dL_du_Old)
    omega = 1 / alpha

    print(f"Step Size: {omega}")

    return omega




def Update_Control_TWBT_elastic_net(f, q0, qs_target, psi, qs_adj, A_p, J_prev, omega_prev, **kwargs):  # Two-way back tracking

    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    dL_du_smooth = Calc_Grad_smooth(psi, qs_adj, f, kwargs['lamda_l2'])
    f_new = prox_l1(f - omega * dL_du_smooth, omega * kwargs['lamda_l1'])
    qs = TI_primal(q0, f_new, A_p, psi, kwargs['Nx'], kwargs['Nt'], kwargs['dt'])
    J_smooth = Calc_Cost_smooth(qs, qs_target, f_new, kwargs['dx'], kwargs['dt'], kwargs['lamda_l2'])
    dJ = J_prev + L2inner_prod(dL_du_smooth, f_new - f, kwargs['dt']) + \
         (1 / (2 * omega)) * L2norm_ROM(f_new - f, kwargs['dt'])
    if J_smooth < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            omega = omega / beta
            f_new = prox_l1(f - omega * dL_du_smooth, omega * kwargs['lamda_l1'])
            qs = TI_primal(q0, f_new, A_p, psi, kwargs['Nx'], kwargs['Nt'], kwargs['dt'])
            J_smooth = Calc_Cost_smooth(qs, qs_target, f_new, kwargs['dx'], kwargs['dt'], kwargs['lamda_l2'])
            dJ = J_prev + L2inner_prod(dL_du_smooth, f_new - f, kwargs['dt']) + \
                 (1 / (2 * omega)) * L2norm_ROM(f_new - f, kwargs['dt'])
            if J_smooth >= dJ:
                print(
                    f"Armijo satisfied and the inner loop converged at the {n}th step with final omega = {omega_final}")
                return f_new_final, omega_final, False
            if kwargs['verbose']: print(f"Armijo satisfied but omega too low! thus omega increased to = {omega}",
                                        f"at inner step={n}")
            n = n + 1
    else:  # If Armijo not satisfied
        k = 0
        while True:
            omega = beta * omega
            f_new = prox_l1(f - omega * dL_du_smooth, omega * kwargs['lamda_l1'])
            qs = TI_primal(q0, f_new, A_p, psi, kwargs['Nx'], kwargs['Nt'], kwargs['dt'])
            J_smooth = Calc_Cost_smooth(qs, qs_target, f_new, kwargs['dx'], kwargs['dt'], kwargs['lamda_l2'])
            dJ = J_prev + L2inner_prod(dL_du_smooth, f_new - f, kwargs['dt']) + \
                 (1 / (2 * omega)) * L2norm_ROM(f_new - f, kwargs['dt'])
            if J_smooth < dJ:  # If Armijo satisfied
                if omega < kwargs['omega_cutoff']:
                    print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                    return f_new, omega, True
                else:
                    print(f"Armijo converged on the {k}th step with final omega = {omega}")
                    return f_new, omega, False
            if omega < kwargs['omega_cutoff']:
                print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                return f_new, omega, True

            if kwargs['verbose']: print(f"Armijo not satisfied thus omega decreased to = {omega}", f"at step={k}")
            k = k + 1


def Update_Control_BB_L1(fNew, dL_du_New, omega, lamda):  # Barzilai Borwein
    f_bb_new = prox_l1(fNew - omega * dL_du_New, omega * lamda)

    return f_bb_new