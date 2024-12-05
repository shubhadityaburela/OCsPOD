from Costs import *
from Grads import *
from Helper import *


def Update_Control_TWBT(f, q0, qs_adj, qs_target, mask, A_p, J_prev, omega_prev, wf, **kwargs):  # Two-way back tracking

    dL_du = Calc_Grad(mask, f, qs_adj, kwargs['lamda'])
    dL_du_norm_square = L2norm_ROM(dL_du, kwargs['dt'])
    dL_du_norm = np.sqrt(dL_du_norm_square)

    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = f - omega * dL_du
    qs = wf.TI_primal(q0, f_new, A_p, mask)
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
            qs = wf.TI_primal(q0, f_new, A_p, mask)
            J = Calc_Cost(qs, qs_target, f_new, kwargs['dx'], kwargs['dt'], kwargs['lamda'])
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
            qs = wf.TI_primal(q0, f_new, A_p, mask)
            J = Calc_Cost(qs, qs_target, f_new, kwargs['dx'], kwargs['dt'], kwargs['lamda'])
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



def Update_Control_Arm(f, q0, qs_adj, qs_target, mask, A_p, J_prev, wf, **kwargs):
    itr = 5
    count = 0
    omega = kwargs['omega_init']
    dL_du = Calc_Grad(mask, f, qs_adj, kwargs['lamda'])
    dL_du_norm_square = L2norm_ROM(dL_du, kwargs['dt'])
    dL_du_norm = np.sqrt(dL_du_norm_square)

    for k in range(kwargs['Armijo_iter']):
        f_new = f - omega * dL_du

        # Solve the primal equation
        qs = wf.TI_primal(q0, f_new, A_p, mask)

        if np.isnan(qs).any() and k < kwargs['Armijo_iter'] - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / kwargs['omega_decr']
        elif np.isnan(qs).any() and k == kwargs['Armijo_iter'] - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost(qs, qs_target, f_new, kwargs['dx'], kwargs['dt'], kwargs['lamda'])
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J < dJ:
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_new, J, dL_du, dL_du_norm, False
            elif J >= dJ or np.isnan(J):
                if k == kwargs['Armijo_iter'] - 1:
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_new, J, dL_du, dL_du_norm, True
                else:
                    if J == dJ:
                        if kwargs['verbose']: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                                                    f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / kwargs['omega_decr']
                        count = count + 1
                        if count > itr:
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_new, J, dL_du, dL_du_norm, True
                    else:
                        if kwargs['verbose']: print(f"Step size omega = {omega} too large!",
                                                    f"Reducing omega at iter={k + 1}")
                        omega = omega / kwargs['omega_decr']



def Update_Control_PODG_FOTR_FA_TWBT(f, a0_primal, qs_adj, qs_target, V_p, Ar_p, psir_p, mask, J_prev, omega_prev,
                                     wf, **kwargs):
    dL_du = Calc_Grad(mask, f, qs_adj, kwargs['lamda'])
    dL_du_norm_square = L2norm_ROM(dL_du, kwargs['dt'])
    dL_du_norm = np.sqrt(dL_du_norm_square)

    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = f - omega * dL_du
    as_ = wf.TI_primal_PODG_FOTR(a0_primal, f_new, Ar_p, psir_p)
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
            as_ = wf.TI_primal_PODG_FOTR(a0_primal, f_new, Ar_p, psir_p)
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
            as_ = wf.TI_primal_PODG_FOTR(a0_primal, f_new, Ar_p, psir_p)
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



def Update_Control_PODG_FOTR_FA_Arm(f, a0_primal, qs_adj, qs_target, V_p, Ar_p, psir_p, mask, J_prev,
                                     wf, **kwargs):
    itr = 5
    count = 0
    omega = kwargs['omega_init']
    dL_du = Calc_Grad(mask, f, qs_adj, kwargs['lamda'])
    dL_du_norm_square = L2norm_ROM(dL_du, kwargs['dt'])
    dL_du_norm = np.sqrt(dL_du_norm_square)

    for k in range(kwargs['Armijo_iter']):
        f_new = f - omega * dL_du

        # Solve the reduced primal equation
        as_ = wf.TI_primal_PODG_FOTR(a0_primal, f_new, Ar_p, psir_p)

        if np.isnan(as_).any() and k < kwargs['Armijo_iter'] - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / kwargs['omega_decr']
        elif np.isnan(as_).any() and k == kwargs['Armijo_iter'] - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost_PODG(V_p, as_, qs_target, f_new, kwargs['dx'], kwargs['dt'], kwargs['lamda'])
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J < dJ:
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_new, J, dL_du, dL_du_norm, False
            elif J >= dJ or np.isnan(J):
                if k == kwargs['Armijo_iter'] - 1:
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_new, J, dL_du, dL_du_norm, True
                else:
                    if J == dJ:
                        if kwargs['verbose']: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                                                    f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / kwargs['omega_decr']
                        count = count + 1
                        if count > itr:
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_new, J, dL_du, dL_du_norm, True
                    else:
                        if kwargs['verbose']: print(f"Step size omega = {omega} too large!",
                                                    f"Reducing omega at iter={k + 1}")
                        omega = omega / kwargs['omega_decr']


def Update_Control_sPODG_FOTR_FA_TWBT(f, lhs, rhs, c, a0_primal, qs_adj, qs_target, delta_s, Vdp, mask, J_prev,
                                      omega_prev, modes, wf, **kwargs):
    dL_du = Calc_Grad(mask, f, qs_adj, kwargs['lamda'])
    dL_du_norm_square = L2norm_ROM(dL_du, kwargs['dt'])
    dL_du_norm = np.sqrt(dL_du_norm_square)

    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = f - omega * dL_du
    as_, intIds, weights = wf.TI_primal_sPODG_FOTR(lhs, rhs, c, a0_primal, f_new, delta_s, modes)
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
            as_, intIds_n, weights_n = wf.TI_primal_sPODG_FOTR(lhs, rhs, c, a0_primal, f_new, delta_s, modes)
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
            as_, intIds_k, weights_k = wf.TI_primal_sPODG_FOTR(lhs, rhs, c, a0_primal, f_new, delta_s, modes)
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


def Update_Control_sPODG_FOTR_FA_Arm(f, lhs, rhs, c, a0_primal, qs_adj, qs_target, delta_s, Vdp, mask, J_prev,
                                      modes, wf, **kwargs):
    itr = 5
    count = 0
    omega = kwargs['omega_init']
    dL_du = Calc_Grad(mask, f, qs_adj, kwargs['lamda'])
    dL_du_norm_square = L2norm_ROM(dL_du, kwargs['dt'])
    dL_du_norm = np.sqrt(dL_du_norm_square)

    for k in range(kwargs['Armijo_iter']):
        f_new = f - omega * dL_du

        # Solve the reduced primal equation
        as_, intIds, weights = wf.TI_primal_sPODG_FOTR(lhs, rhs, c, a0_primal, f_new, delta_s, modes)

        if np.isnan(as_).any() and k < kwargs['Armijo_iter'] - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / kwargs['omega_decr']
        elif np.isnan(as_).any() and k == kwargs['Armijo_iter'] - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J, qq = Calc_Cost_sPODG(Vdp, as_[:-1], qs_target, f_new, intIds, weights,
                                    kwargs['dx'], kwargs['dt'], kwargs['lamda'])
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J < dJ:
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_new, J, dL_du, dL_du_norm, qq, False
            elif J >= dJ or np.isnan(J):
                if k == kwargs['Armijo_iter'] - 1:
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_new, J, dL_du, dL_du_norm, qq, True
                else:
                    if J == dJ:
                        if kwargs['verbose']: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                                                    f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / kwargs['omega_decr']
                        count = count + 1
                        if count > itr:
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_new, J, dL_du, dL_du_norm, qq, True
                    else:
                        if kwargs['verbose']: print(f"Step size omega = {omega} too large!",
                                                    f"Reducing omega at iter={k + 1}")
                        omega = omega / kwargs['omega_decr']



def Update_Control_sPODG_FOTR_FA_TWBT_dot(f, lhs, rhs, c, a0_primal, qs_adj, qs_target, delta_s, Vdp, mask, J_prev,
                                          omega_prev, modes, wf, **kwargs):
    dL_du = Calc_Grad(mask, f, qs_adj, kwargs['lamda'])
    dL_du_norm_square = L2norm_ROM(dL_du, kwargs['dt'])
    dL_du_norm = np.sqrt(dL_du_norm_square)

    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = f - omega * dL_du
    as_, intIds, weights, as_dot = wf.TI_primal_sPODG_FOTR_dot(lhs, rhs, c, a0_primal, f_new, delta_s, modes)
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
            as_final = np.copy(as_)
            intIds_final = np.copy(intIds)
            weights_final = np.copy(weights)
            as_dot_final = np.copy(as_dot)
            omega = omega / beta
            f_new = f - omega * dL_du
            as_, intIds, weights, as_dot = wf.TI_primal_sPODG_FOTR_dot(lhs, rhs, c, a0_primal, f_new, delta_s, modes)
            J, qq = Calc_Cost_sPODG(Vdp, as_[:-1], qs_target, f_new, intIds, weights,
                                    kwargs['dx'], kwargs['dt'], kwargs['lamda'])
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J >= dJ:
                print(
                    f"Armijo satisfied and the inner loop converged at the {n}th step with final omega = {omega_final}")
                return f_new_final, J_final, dL_du, dL_du_norm, omega_final, qq_final, as_final, \
                    intIds_final, weights_final, as_dot_final, False
            if kwargs['verbose']: print(f"Armijo satisfied but omega too low! thus omega increased to = {omega}",
                                        f"at inner step={n}")
            n = n + 1
    else:  # If Armijo not satisfied
        k = 0
        while True:
            omega = beta * omega
            f_new = f - omega * dL_du
            as_, intIds_k, weights_k, as_dot = wf.TI_primal_sPODG_FOTR_dot(lhs, rhs, c, a0_primal, f_new, delta_s, modes)
            J, qq = Calc_Cost_sPODG(Vdp, as_[:-1], qs_target, f_new, intIds_k, weights_k,
                                    kwargs['dx'], kwargs['dt'], kwargs['lamda'])
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J < dJ:  # If Armijo satisfied
                if omega < kwargs['omega_cutoff']:
                    print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                    return f_new, J, dL_du, dL_du_norm, omega, qq, as_, intIds_k, weights_k, as_dot, True
                else:
                    print(f"Armijo converged on the {k}th step with final omega = {omega}")
                    return f_new, J, dL_du, dL_du_norm, omega, qq, as_, intIds_k, weights_k, as_dot, False
            if omega < kwargs['omega_cutoff']:
                print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                return f_new, J, dL_du, dL_du_norm, omega, qq, as_, intIds_k, weights_k, as_dot, True

            if kwargs['verbose']: print(f"Armijo not satisfied thus omega decreased to = {omega}", f"at step={k}")
            k = k + 1


def Update_Control_sPODG_FOTR_RA_TWBT(f, lhs, rhs, c, a0_primal, Vd_a, as_adj, qs_target, delta_s, Vdp, mask, J_prev,
                                      omega_prev, modes, intIds, weights, wf, **kwargs):
    dL_du = Calc_Grad_sPODG(mask, f, Vd_a, as_adj[:-1], intIds, weights, kwargs['lamda'])
    dL_du_norm_square = L2norm_ROM(dL_du, kwargs['dt'])
    dL_du_norm = np.sqrt(dL_du_norm_square)

    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = f - omega * dL_du
    as_, intIds, weights = wf.TI_primal_sPODG_FOTR(lhs, rhs, c, a0_primal, f_new, delta_s, modes)
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
            as_, intIds_n, weights_n = wf.TI_primal_sPODG_FOTR(lhs, rhs, c, a0_primal, f_new, delta_s, modes)
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
            as_, intIds_k, weights_k = wf.TI_primal_sPODG_FOTR(lhs, rhs, c, a0_primal, f_new, delta_s, modes)
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


def Update_Control_BB(fOld, fNew, dL_du_Old, qs_adj, mask, itr, **kwargs):  # Barzilai Borwein

    dL_du_New = Calc_Grad(mask, fNew, qs_adj, kwargs['lamda'])
    dL_du_norm_square = L2norm_ROM(dL_du_New, kwargs['dt'])
    dL_du_norm = np.sqrt(dL_du_norm_square)
    alpha = BarzilaiBorwein(itr, kwargs['dt'], fNew, fOld, dL_du_New, dL_du_Old)
    omega = 1 / alpha

    print(f"Step Size: {omega}")

    f_bb_new = fNew - omega * dL_du_New

    return f_bb_new, dL_du_New, dL_du_norm


def Update_Control_sPODG_FOTR_RA_BB(fOld, fNew, dL_du_Old, Vd_a, as_adj, mask, itr, intIds, weights,
                                    **kwargs):  # Barzilai Borwein

    dL_du_New = Calc_Grad_sPODG(mask, fNew, Vd_a, as_adj[:-1], intIds, weights, kwargs['lamda'])
    dL_du_norm_square = L2norm_ROM(dL_du_New, kwargs['dt'])
    dL_du_norm = np.sqrt(dL_du_norm_square)
    alpha = BarzilaiBorwein(itr, kwargs['dt'], fNew, fOld, dL_du_New, dL_du_Old)
    omega = 1 / alpha

    print(f"Step Size: {omega}")

    f_bb_new = fNew - omega * dL_du_New

    return f_bb_new, dL_du_New, dL_du_norm



def Update_Control_sPODG_FRTO_TWBT(f, lhs_p, rhs_p, c_p, a_p, as_adj, as_, as_target, delta_s, J_prev,
                                      omega_prev, modes, intIds, weights, wf, **kwargs):
    dL_du = Calc_Grad_sPODG_FRTO(f, c_p, as_adj, as_, intIds, weights, kwargs['lamda'])
    dL_du_norm_square = L2norm_ROM(dL_du, kwargs['dt'])
    dL_du_norm = np.sqrt(dL_du_norm_square)

    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = f - omega * dL_du
    as_, _, _, _ = wf.TI_primal_sPODG_FRTO(lhs_p, rhs_p, c_p, a_p, f_new, delta_s, modes=modes)
    J = Calc_Cost_sPODG_FRTO_NC(as_, as_target, f_new, kwargs['dt'], kwargs['lamda'])
    dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
    if J < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            J_final = J
            omega = omega / beta
            f_new = f - omega * dL_du
            as_, _, _, _ = wf.TI_primal_sPODG_FRTO(lhs_p, rhs_p, c_p, a_p, f_new, delta_s, modes=modes)
            J = Calc_Cost_sPODG_FRTO_NC(as_, as_target, f_new, kwargs['dt'], kwargs['lamda'])
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
            as_, _, _, _ = wf.TI_primal_sPODG_FRTO(lhs_p, rhs_p, c_p, a_p, f_new, delta_s, modes=modes)
            J = Calc_Cost_sPODG_FRTO_NC(as_, as_target, f_new, kwargs['dt'], kwargs['lamda'])
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


def Update_Control_sPODG_FRTO_BB(fOld, fNew, dL_du_Old, as_adj, as_, C, itr, intIds, weights, **kwargs):  # Barzilai Borwein

    dL_du_New = Calc_Grad_sPODG_FRTO(fNew, C, as_adj, as_, intIds, weights, kwargs['lamda'])
    dL_du_norm_square = L2norm_ROM(dL_du_New, kwargs['dt'])
    dL_du_norm = np.sqrt(dL_du_norm_square)
    alpha = BarzilaiBorwein(itr, kwargs['dt'], fNew, fOld, dL_du_New, dL_du_Old)
    omega = 1 / alpha

    print(f"Step Size: {omega}")

    f_bb_new = fNew - omega * dL_du_New

    return f_bb_new, dL_du_New, dL_du_norm