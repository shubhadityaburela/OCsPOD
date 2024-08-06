from Costs import *
from Grads import *
from Helper import *
from time import perf_counter


def Update_Control(f, q0, qs_adj, qs_target, mask, A_p, J_prev, wf, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5
    omega = kwargs['omega']

    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad(mask, f, qs_adj, **kwargs)
    time_odeint = perf_counter() - time_odeint
    if kwargs['verbose']: print("Calc_Grad t_cpu = %1.6f" % time_odeint)
    for k in range(kwargs['Armijo_iter']):
        f_new = f - omega * dL_du

        # Solve the primal equation
        qs = wf.TI_primal(q0, f_new, A_p, mask)

        if np.isnan(qs).any() and k < kwargs['Armijo_iter'] - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / 4
        elif np.isnan(qs).any() and k == kwargs['Armijo_iter'] - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost(qs, qs_target, f_new, **kwargs)
            dJ = J_prev - kwargs['delta'] * omega * np.linalg.norm(dL_du) ** 2
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, np.linalg.norm(dL_du)
            elif J >= dJ or np.isnan(J):
                if k == kwargs['Armijo_iter'] - 1:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_opt, J_opt, np.linalg.norm(dL_du)
                else:
                    if J == dJ:
                        if kwargs['verbose']: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                              f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 4
                        count = count + 1
                        if count > itr:
                            J_opt = J
                            f_opt = f_new
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_opt, J_opt, np.linalg.norm(dL_du)
                    else:
                        if kwargs['verbose']: print(f"No NANs found but step size omega = {omega} too large!",
                              f"Reducing omega at iter={k + 1}")
                        omega = omega / 4



def Update_Control_PODG_FRTO(f, a0_primal, as_adj, qs_target, V, Ar, psir, J_prev, wf, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5
    omega = kwargs['omega']

    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad_PODG(psir, f, as_adj, **kwargs)
    time_odeint = perf_counter() - time_odeint
    if kwargs['verbose']: print("Calc_Grad t_cpu = %1.6f" % time_odeint)
    for k in range(kwargs['Armijo_iter']):
        f_new = f - omega * dL_du

        # Solve the primal equation
        as_ = wf.TI_primal_PODG_FRTO(a0_primal, f_new, Ar, psir)

        if np.isnan(as_).any() and k < kwargs['Armijo_iter'] - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / 4
        elif np.isnan(as_).any() and k == kwargs['Armijo_iter'] - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost_PODG(V, as_, qs_target, f_new, **kwargs)
            dJ = J_prev - kwargs['delta'] * omega * np.linalg.norm(dL_du) ** 2
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, np.linalg.norm(dL_du), False
            elif J >= dJ or np.isnan(J):
                if k == kwargs['Armijo_iter'] - 1:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_opt, J_opt, np.linalg.norm(dL_du), True
                else:
                    if J == dJ:
                        if kwargs['verbose']: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                              f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 4
                        count = count + 1
                        if count > itr:
                            J_opt = J
                            f_opt = f_new
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_opt, J_opt, np.linalg.norm(dL_du), True
                    else:
                        if kwargs['verbose']: print(f"No NANs found but step size omega = {omega} too large!",
                              f"Reducing omega at iter={k + 1}")
                        omega = omega / 4


def Update_Control_sPODG_FRTO(f, lhs, rhs, c, Vd_p, a0_primal, as_, as_adj, qs_target, delta_s, J_prev,
                              intIds, weights, wf, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5
    omega = kwargs['omega']

    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad_sPODG_FRTO(f, c, intIds, weights, as_, as_adj, **kwargs)
    time_odeint = perf_counter() - time_odeint
    if kwargs['verbose']: print("Calc_Grad t_cpu = %1.6f" % time_odeint)
    for k in range(kwargs['Armijo_iter']):
        f_new = f - omega * dL_du

        # Solve the primal equation
        as_, _ = wf.TI_primal_sPODG_FRTO(lhs, rhs, c, a0_primal, f_new, delta_s)

        if np.isnan(as_).any() and k < kwargs['Armijo_iter'] - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / 4
        elif np.isnan(as_).any() and k == kwargs['Armijo_iter'] - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost_sPODG(Vd_p, as_, qs_target, f_new, intIds, weights, **kwargs)
            dJ = J_prev - kwargs['delta'] * omega * np.linalg.norm(dL_du) ** 2
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, np.linalg.norm(dL_du), False
            elif J >= dJ or np.isnan(J):
                if k == kwargs['Armijo_iter'] - 1:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_opt, J_opt, np.linalg.norm(dL_du), True
                else:
                    if J == dJ:
                        if kwargs['verbose']: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                              f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 4
                        count = count + 1
                        if count > itr:
                            J_opt = J
                            f_opt = f_new
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_opt, J_opt, np.linalg.norm(dL_du), True
                    else:
                        if kwargs['verbose']: print(f"No NANs found but step size omega = {omega} too large!",
                              f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 4


def Update_Control_sPODG_FRTO_newcost(f, lhs, rhs, c, Vd_p, a0_primal, as_, as_adj, as_target, z_target, delta_s, J_prev,
                                      intIds, weights, wf, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5
    omega = kwargs['omega']

    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad_sPODG_FRTO(f, c, intIds, weights, as_, as_adj, **kwargs)
    time_odeint = perf_counter() - time_odeint
    if kwargs['verbose']: print("Calc_Grad t_cpu = %1.6f" % time_odeint)
    for k in range(kwargs['Armijo_iter']):
        f_new = f - omega * dL_du

        # Solve the primal equation
        as_, _ = wf.TI_primal_sPODG_FRTO(lhs, rhs, c, a0_primal, f_new, delta_s)

        if np.isnan(as_).any() and k < kwargs['Armijo_iter'] - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / 4
        elif np.isnan(as_).any() and k == kwargs['Armijo_iter'] - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost_sPODG_newcost(as_, as_target, z_target, f_new, **kwargs)
            dJ = J_prev - kwargs['delta'] * omega * np.linalg.norm(dL_du) ** 2
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, np.linalg.norm(dL_du), False
            elif J >= dJ or np.isnan(J):
                if k == kwargs['Armijo_iter'] - 1:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_opt, J_opt, np.linalg.norm(dL_du), True
                else:
                    if J == dJ:
                        if kwargs['verbose']: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                                                    f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 4
                        count = count + 1
                        if count > itr:
                            J_opt = J
                            f_opt = f_new
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_opt, J_opt, np.linalg.norm(dL_du), True
                    else:
                        if kwargs['verbose']: print(f"No NANs found but step size omega = {omega} too large!",
                                                    f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 4


def Update_Control_PODG_FOTR(f, a0_primal, as_adj, qs_target, V_p, Ar_p, psir_p, psir_a, J_prev, wf, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5
    omega = kwargs['omega']

    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad_PODG(psir_a, f, as_adj, **kwargs)
    time_odeint = perf_counter() - time_odeint
    if kwargs['verbose']: print("Calc_Grad t_cpu = %1.6f" % time_odeint)
    for k in range(kwargs['Armijo_iter']):
        f_new = f - omega * dL_du

        # Solve the primal equation
        as_ = wf.TI_primal_PODG_FOTR(a0_primal, f_new, Ar_p, psir_p)

        if np.isnan(as_).any() and k < kwargs['Armijo_iter'] - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / 4
        elif np.isnan(as_).any() and k == kwargs['Armijo_iter'] - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost_PODG(V_p, as_, qs_target, f_new, **kwargs)
            dJ = J_prev - kwargs['delta'] * omega * np.linalg.norm(dL_du) ** 2
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, np.linalg.norm(dL_du), False
            elif J >= dJ or np.isnan(J):
                if k == kwargs['Armijo_iter'] - 1:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_opt, J_opt, np.linalg.norm(dL_du), True
                else:
                    if J == dJ:
                        if kwargs['verbose']: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                              f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 4
                        count = count + 1
                        if count > itr:
                            J_opt = J
                            f_opt = f_new
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_opt, J_opt, np.linalg.norm(dL_du), True
                    else:
                        if kwargs['verbose']: print(f"No NANs found but step size omega = {omega} too large!",
                              f"Reducing omega at iter={k + 1}")
                        omega = omega / 4


def Update_Control_sPODG_FOTR(f, lhs, rhs, c, a0_primal, as_adj, qs_target, delta_s, Vdp, C_a, J_prev, intIds,
                              weights, wf, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5
    omega = kwargs['omega']

    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad_sPODG_FOTR(f, C_a, intIds, weights, as_adj, **kwargs)
    time_odeint = perf_counter() - time_odeint
    if kwargs['verbose']: print("Calc_Grad t_cpu = %1.6f" % time_odeint)
    for k in range(kwargs['Armijo_iter']):
        f_new = f - omega * dL_du

        # Solve the primal equation
        as_ = wf.TI_primal_sPODG_FOTR(lhs, rhs, c, a0_primal, f_new, delta_s)

        if np.isnan(as_).any() and k < kwargs['Armijo_iter'] - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / 4
        elif np.isnan(as_).any() and k == kwargs['Armijo_iter'] - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost_sPODG(Vdp, as_, qs_target, f_new, intIds, weights, **kwargs)
            dJ = J_prev - kwargs['delta'] * omega * np.linalg.norm(dL_du) ** 2
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, np.linalg.norm(dL_du), False
            elif J >= dJ or np.isnan(J):
                if k == kwargs['Armijo_iter'] - 1:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_opt, J_opt, np.linalg.norm(dL_du), True
                else:
                    if J == dJ:
                        if kwargs['verbose']: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                                          f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 4
                        count = count + 1
                        if count > itr:
                            J_opt = J
                            f_opt = f_new
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_opt, J_opt, np.linalg.norm(dL_du), True
                    else:
                        if kwargs['verbose']: print(f"No NANs found but step size omega = {omega} too large!",
                              f"Reducing omega at iter={k + 1}")
                        omega = omega / 4
