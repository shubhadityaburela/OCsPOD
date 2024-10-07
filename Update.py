from Costs import *
from Grads import *
from Helper import *
from time import perf_counter

from Helper_sPODG import findIntervals


def Update_Control(f, q0, qs_adj, qs_target, mask, A_p, J_prev, wf, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5
    omega = kwargs['omega']

    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad(mask, f, qs_adj, **kwargs)
    time_odeint = perf_counter() - time_odeint
    dL_du_norm_square = L2norm_ROM(dL_du, **kwargs)
    dL_du_norm = np.sqrt(dL_du_norm_square)
    if kwargs['verbose']: print("Calc_Grad t_cpu = %1.6f" % time_odeint)
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
            J = Calc_Cost(qs, qs_target, f_new, **kwargs)
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, dL_du_norm, False
            elif J >= dJ or np.isnan(J):
                if k == kwargs['Armijo_iter'] - 1:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_opt, J_opt, dL_du_norm, True
                else:
                    if J == dJ:
                        if kwargs['verbose']: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                                                    f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / kwargs['omega_decr']
                        count = count + 1
                        if count > itr:
                            J_opt = J
                            f_opt = f_new
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_opt, J_opt, dL_du_norm, True
                    else:
                        if kwargs['verbose']: print(f"No NANs found but step size omega = {omega} too large!",
                                                    f"Reducing omega at iter={k + 1}")
                        omega = omega / kwargs['omega_decr']


def Update_Control_TWBT(f, q0, qs_adj, qs_target, mask, A_p, J_prev, omega_prev, wf, **kwargs):  # Two-way back tracking

    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad(mask, f, qs_adj, **kwargs)
    time_odeint = perf_counter() - time_odeint
    dL_du_norm_square = L2norm_ROM(dL_du, **kwargs)
    dL_du_norm = np.sqrt(dL_du_norm_square)
    if kwargs['verbose']: print("Calc_Grad t_cpu = %1.6f" % time_odeint)

    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = f - omega * dL_du
    qs = wf.TI_primal(q0, f_new, A_p, mask)
    J = Calc_Cost(qs, qs_target, f_new, **kwargs)
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
            J = Calc_Cost(qs, qs_target, f_new, **kwargs)
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J >= dJ:
                print(
                    f"Armijo satisfied and the inner loop converged at the {n}th step with final omega = {omega_final}")
                return f_new_final, J_final, dL_du_norm, omega_final, False
            if kwargs['verbose']: print(f"Armijo satisfied but omega too low! thus omega increased to = {omega}",
                                        f"at inner step={n}")
            n = n + 1
    else:  # If Armijo not satisfied
        k = 0
        while True:
            omega = beta * omega
            f_new = f - omega * dL_du
            qs = wf.TI_primal(q0, f_new, A_p, mask)
            J = Calc_Cost(qs, qs_target, f_new, **kwargs)
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J < dJ:  # If Armijo satisfied
                if omega < kwargs['omega_cutoff']:
                    print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                    return f_new, J, dL_du_norm, omega, True
                else:
                    print(f"Armijo converged on the {k}th step with final omega = {omega}")
                    return f_new, J, dL_du_norm, omega, False
            if omega < kwargs['omega_cutoff']:
                print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                return f_new, J, dL_du_norm, omega, True

            if kwargs['verbose']: print(f"Armijo not satisfied thus omega decreased to = {omega}", f"at step={k}")
            k = k + 1


def Update_Control_PODG_FRTO(f, a0_primal, as_adj, qs_target, V, Ar, psir, J_prev, wf, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5
    omega = kwargs['omega']

    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad_PODG(psir, f, as_adj, **kwargs)
    time_odeint = perf_counter() - time_odeint
    dL_du_norm_square = L2norm_ROM(dL_du, **kwargs)
    dL_du_norm = np.sqrt(dL_du_norm_square)
    if kwargs['verbose']: print("Calc_Grad t_cpu = %1.6f" % time_odeint)
    for k in range(kwargs['Armijo_iter']):
        f_new = f - omega * dL_du

        # Solve the primal equation
        as_ = wf.TI_primal_PODG_FRTO(a0_primal, f_new, Ar, psir)

        if np.isnan(as_).any() and k < kwargs['Armijo_iter'] - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / kwargs['omega_decr']
        elif np.isnan(as_).any() and k == kwargs['Armijo_iter'] - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost_PODG(V, as_, qs_target, f_new, **kwargs)
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, dL_du_norm, False
            elif J >= dJ or np.isnan(J):
                if k == kwargs['Armijo_iter'] - 1:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_opt, J_opt, dL_du_norm, True
                else:
                    if J == dJ:
                        if kwargs['verbose']: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                                                    f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / kwargs['omega_decr']
                        count = count + 1
                        if count > itr:
                            J_opt = J
                            f_opt = f_new
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_opt, J_opt, dL_du_norm, True
                    else:
                        if kwargs['verbose']: print(f"No NANs found but step size omega = {omega} too large!",
                                                    f"Reducing omega at iter={k + 1}")
                        omega = omega / kwargs['omega_decr']


def Update_Control_sPODG_FRTO(f, lhs, rhs, c, Vd_p, a0_primal, as_, as_adj, qs_target, delta_s, J_prev,
                              intIds, weights, wf, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5
    omega = kwargs['omega']

    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad_sPODG_FRTO(f, c, intIds, weights, as_, as_adj, **kwargs)
    time_odeint = perf_counter() - time_odeint
    dL_du_norm_square = L2norm_ROM(dL_du, **kwargs)
    dL_du_norm = np.sqrt(dL_du_norm_square)
    if kwargs['verbose']: print("Calc_Grad t_cpu = %1.6f" % time_odeint)
    for k in range(kwargs['Armijo_iter']):
        f_new = f - omega * dL_du

        # Solve the primal equation
        as_, _ = wf.TI_primal_sPODG_FRTO(lhs, rhs, c, a0_primal, f_new, delta_s)

        if np.isnan(as_).any() and k < kwargs['Armijo_iter'] - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / kwargs['omega_decr']
        elif np.isnan(as_).any() and k == kwargs['Armijo_iter'] - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost_sPODG(Vd_p, as_, qs_target, f_new, intIds, weights, **kwargs)
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, dL_du_norm, False
            elif J >= dJ or np.isnan(J):
                if k == kwargs['Armijo_iter'] - 1:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_opt, J_opt, dL_du_norm, True
                else:
                    if J == dJ:
                        if kwargs['verbose']: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                                                    f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / kwargs['omega_decr']
                        count = count + 1
                        if count > itr:
                            J_opt = J
                            f_opt = f_new
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_opt, J_opt, dL_du_norm, True
                    else:
                        if kwargs['verbose']: print(f"No NANs found but step size omega = {omega} too large!",
                                                    f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / kwargs['omega_decr']


def Update_Control_PODG_FOTR_adaptive(f, a0_primal, qs_adj, qs_target, V_p, Ar_p, psir_p, mask, J_prev, wf, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5
    omega = kwargs['omega']

    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad(mask, f, qs_adj, **kwargs)
    time_odeint = perf_counter() - time_odeint
    dL_du_norm_square = L2norm_ROM(dL_du, **kwargs)
    dL_du_norm = np.sqrt(dL_du_norm_square)
    if kwargs['verbose']: print("Calc_Grad t_cpu = %1.6f" % time_odeint)
    for k in range(kwargs['Armijo_iter']):
        f_new = f - omega * dL_du

        # Solve the primal equation
        as_ = wf.TI_primal_PODG_FOTR(a0_primal, f_new, Ar_p, psir_p)

        if np.isnan(as_).any() and k < kwargs['Armijo_iter'] - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / kwargs['omega_decr']
        elif np.isnan(as_).any() and k == kwargs['Armijo_iter'] - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost_PODG(V_p, as_, qs_target, f_new, **kwargs)
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, dL_du_norm, False
            elif J >= dJ or np.isnan(J):
                if k == kwargs['Armijo_iter'] - 1:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_opt, J_opt, dL_du_norm, True
                else:
                    if J == dJ:
                        if kwargs['verbose']: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                                                    f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / kwargs['omega_decr']
                        count = count + 1
                        if count > itr:
                            J_opt = J
                            f_opt = f_new
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_opt, J_opt, dL_du_norm, True
                    else:
                        if kwargs['verbose']: print(f"No NANs found but step size omega = {omega} too large!",
                                                    f"Reducing omega at iter={k + 1}")
                        omega = omega / kwargs['omega_decr']


def Update_Control_PODG_FOTR_adaptive_TWBT(f, a0_primal, qs_adj, qs_target, V_p, Ar_p, psir_p, mask, J_prev, omega_prev,
                                           wf, **kwargs):
    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad(mask, f, qs_adj, **kwargs)
    time_odeint = perf_counter() - time_odeint
    dL_du_norm_square = L2norm_ROM(dL_du, **kwargs)
    dL_du_norm = np.sqrt(dL_du_norm_square)
    if kwargs['verbose']: print("Calc_Grad t_cpu = %1.6f" % time_odeint)

    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = f - omega * dL_du
    as_ = wf.TI_primal_PODG_FOTR(a0_primal, f_new, Ar_p, psir_p)
    J = Calc_Cost_PODG(V_p, as_, qs_target, f_new, **kwargs)
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
            J = Calc_Cost_PODG(V_p, as_, qs_target, f_new, **kwargs)
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J >= dJ:
                print(
                    f"Armijo satisfied and the inner loop converged at the {n}th step with final omega = {omega_final}")
                return f_new_final, J_final, dL_du_norm, omega_final, False
            if kwargs['verbose']: print(f"Armijo satisfied but omega too low! thus omega increased to = {omega}",
                                        f"at inner step={n}")
            n = n + 1
    else:  # If Armijo not satisfied
        k = 0
        while True:
            omega = beta * omega
            f_new = f - omega * dL_du
            as_ = wf.TI_primal_PODG_FOTR(a0_primal, f_new, Ar_p, psir_p)
            J = Calc_Cost_PODG(V_p, as_, qs_target, f_new, **kwargs)
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J < dJ:  # If Armijo satisfied
                if omega < kwargs['omega_cutoff']:
                    print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                    return f_new, J, dL_du_norm, omega, True
                else:
                    print(f"Armijo converged on the {k}th step with final omega = {omega}")
                    return f_new, J, dL_du_norm, omega, False
            if omega < kwargs['omega_cutoff']:
                print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                return f_new, J, dL_du_norm, omega, True

            if kwargs['verbose']: print(f"Armijo not satisfied thus omega decreased to = {omega}", f"at step={k}")
            k = k + 1


def Update_Control_sPODG_FOTR_adaptive(f, lhs, rhs, c, a0_primal, qs_adj, qs_target, delta_s, Vdp, mask, J_prev, wf, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5
    omega = kwargs['omega']

    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad(mask, f, qs_adj, **kwargs)
    time_odeint = perf_counter() - time_odeint
    dL_du_norm_square = L2norm_ROM(dL_du, **kwargs)
    dL_du_norm = np.sqrt(dL_du_norm_square)
    if kwargs['verbose']: print("Calc_Grad t_cpu = %1.6f" % time_odeint)
    for k in range(kwargs['Armijo_iter']):
        f_new = f - omega * dL_du

        # Solve the primal equation
        as_ = wf.TI_primal_sPODG_FOTR(lhs, rhs, c, a0_primal, f_new, delta_s)
        intIds_k, weights_k = findIntervals(delta_s, as_[-1, :])

        if np.isnan(as_).any() and k < kwargs['Armijo_iter'] - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / kwargs['omega_decr']
        elif np.isnan(as_).any() and k == kwargs['Armijo_iter'] - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost_sPODG(Vdp, as_, qs_target, f_new, intIds_k, weights_k, **kwargs)
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, dL_du_norm, False
            elif J >= dJ or np.isnan(J):
                if k == kwargs['Armijo_iter'] - 1:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_opt, J_opt, dL_du_norm, True
                else:
                    if J == dJ:
                        if kwargs['verbose']: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                                                    f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / kwargs['omega_decr']
                        count = count + 1
                        if count > itr:
                            J_opt = J
                            f_opt = f_new
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_opt, J_opt, dL_du_norm, True
                    else:
                        if kwargs['verbose']: print(f"No NANs found but step size omega = {omega} too large!",
                                                    f"Reducing omega at iter={k + 1}")
                        omega = omega / kwargs['omega_decr']


def Update_Control_sPODG_FOTR_adaptive_TWBT(f, lhs, rhs, c, a0_primal, qs_adj, qs_target, delta_s, Vdp, mask, J_prev,
                                            omega_prev, wf, **kwargs):
    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad(mask, f, qs_adj, **kwargs)
    time_odeint = perf_counter() - time_odeint
    dL_du_norm_square = L2norm_ROM(dL_du, **kwargs)
    dL_du_norm = np.sqrt(dL_du_norm_square)
    if kwargs['verbose']: print("Calc_Grad t_cpu = %1.6f" % time_odeint)

    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = f - omega * dL_du
    as_ = wf.TI_primal_sPODG_FOTR(lhs, rhs, c, a0_primal, f_new, delta_s)
    intIds, weights = findIntervals(delta_s, as_[-1, :])
    J = Calc_Cost_sPODG(Vdp, as_, qs_target, f_new, intIds, weights, **kwargs)
    dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
    if J < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            J_final = J
            omega = omega / beta
            f_new = f - omega * dL_du
            as_ = wf.TI_primal_sPODG_FOTR(lhs, rhs, c, a0_primal, f_new, delta_s)
            intIds_n, weights_n = findIntervals(delta_s, as_[-1, :])
            J = Calc_Cost_sPODG(Vdp, as_, qs_target, f_new, intIds_n, weights_n, **kwargs)
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J >= dJ:
                print(
                    f"Armijo satisfied and the inner loop converged at the {n}th step with final omega = {omega_final}")
                return f_new_final, J_final, dL_du_norm, omega_final, False
            if kwargs['verbose']: print(f"Armijo satisfied but omega too low! thus omega increased to = {omega}",
                                        f"at inner step={n}")
            n = n + 1
    else:  # If Armijo not satisfied
        k = 0
        while True:
            omega = beta * omega
            f_new = f - omega * dL_du
            as_ = wf.TI_primal_sPODG_FOTR(lhs, rhs, c, a0_primal, f_new, delta_s)
            intIds_k, weights_k = findIntervals(delta_s, as_[-1, :])
            J = Calc_Cost_sPODG(Vdp, as_, qs_target, f_new, intIds_k, weights_k, **kwargs)
            dJ = J_prev - kwargs['delta'] * omega * dL_du_norm_square
            if J < dJ:  # If Armijo satisfied
                if omega < kwargs['omega_cutoff']:
                    print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                    return f_new, J, dL_du_norm, omega, True
                else:
                    print(f"Armijo converged on the {k}th step with final omega = {omega}")
                    return f_new, J, dL_du_norm, omega, False
            if omega < kwargs['omega_cutoff']:
                print(f"Omega went below the omega cutoff on the {k}th step, thus exiting the loop !!!!!!")
                return f_new, J, dL_du_norm, omega, True

            if kwargs['verbose']: print(f"Armijo not satisfied thus omega decreased to = {omega}", f"at step={k}")
            k = k + 1
