from Costs import *
from FOM_solver import TI_primal, TI_primal_kdv_impl, TI_primal_kdv_expl
from Grads import *
from Helper import *
from PODG_solver import TI_primal_PODG_FOTR, TI_primal_PODG_FRTO, \
    TI_primal_PODG_FOTR_kdv_expl, TI_primal_PODG_FRTO_kdv_expl
from sPODG_solver import TI_primal_sPODG_FOTR, TI_primal_sPODG_FRTO, \
    TI_primal_sPODG_FOTR_kdv_expl, \
    TI_primal_sPODG_FRTO_kdv_expl, TI_primal_sPODG_FRTO_kdv_impl


def Update_Control_TWBT(f, q0, qs_target, mask, A_p, J_s_prev, omega_prev, dL_du_s, C, adjust,
                        **kwargs):  # Two-way back tracking

    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
    qs = TI_primal(q0, f_new, A_p, mask, kwargs['Nx'], kwargs['Nt'], kwargs['dt'])
    J_s, _ = Calc_Cost(qs, qs_target, f_new, C, kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'], kwargs['lamda_l2'],
                       adjust)
    dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
         (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
    if J_s < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            omega = omega / beta
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            qs = TI_primal(q0, f_new, A_p, mask, kwargs['Nx'], kwargs['Nt'], kwargs['dt'])
            J_s, _ = Calc_Cost(qs, qs_target, f_new, C, kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'],
                               kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
                 (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
            if J_s >= dJ:
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
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            qs = TI_primal(q0, f_new, A_p, mask, kwargs['Nx'], kwargs['Nt'], kwargs['dt'])
            J_s, _ = Calc_Cost(qs, qs_target, f_new, C, kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'],
                               kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
                 (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
            if J_s < dJ:  # If Armijo satisfied
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


def Update_Control_BB(fNew, dL_du_New, omega, lamda1):  # Barzilai Borwein
    f_bb_new = prox_l1(fNew - omega * dL_du_New, omega * lamda1)

    return f_bb_new


def Update_Control_PODG_FOTR_RA_TWBT(f, a0_primal, qs_target, V_p, Ar_p, psir_p, J_s_prev,
                                     omega_prev, dL_du_s, C, adjust, **kwargs):
    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
    as_ = TI_primal_PODG_FOTR(a0_primal, f_new, Ar_p, psir_p, kwargs['Nt'], kwargs['dt'])
    J_s, _ = Calc_Cost_PODG(V_p, as_, qs_target, f_new, C, kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'],
                            kwargs['lamda_l2'], adjust)
    dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
         (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
    if J_s < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            omega = omega / beta
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            as_ = TI_primal_PODG_FOTR(a0_primal, f_new, Ar_p, psir_p, kwargs['Nt'], kwargs['dt'])
            J_s, _ = Calc_Cost_PODG(V_p, as_, qs_target, f_new, C, kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'],
                                    kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + (kwargs['delta'] / omega) * L2norm_ROM(
                f_new - f, kwargs['dt'])
            if J_s >= dJ:
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
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            as_ = TI_primal_PODG_FOTR(a0_primal, f_new, Ar_p, psir_p, kwargs['Nt'], kwargs['dt'])
            J_s, _ = Calc_Cost_PODG(V_p, as_, qs_target, f_new, C, kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'],
                                    kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + (kwargs['delta'] / omega) * L2norm_ROM(
                f_new - f, kwargs['dt'])
            if J_s < dJ:  # If Armijo satisfied
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


def Update_Control_sPODG_FOTR_RA_TWBT(f, lhs, c, a0_primal, qs_target, delta_s, Vdp, J_s_prev,
                                      omega_prev, modes, dL_du_s, adjust, v, **kwargs):
    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
    as_p, intIds, weights = TI_primal_sPODG_FOTR(lhs, c, a0_primal, f_new, delta_s, modes, kwargs['Nt'],
                                                 kwargs['dt'], v)
    J_s, J_ns, _ = Calc_Cost_sPODG(Vdp, as_p[:-1], qs_target, f_new, intIds, weights,
                                   kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
    dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
         (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
    if J_s < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            omega = omega / beta
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            as_p, intIds_n, weights_n = TI_primal_sPODG_FOTR(lhs, c, a0_primal, f_new, delta_s, modes, kwargs['Nt'],
                                                             kwargs['dt'], v)
            J_s, J_ns, _ = Calc_Cost_sPODG(Vdp, as_p[:-1], qs_target, f_new, intIds_n, weights_n,
                                           kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
                 (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
            if J_s >= dJ:
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
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            as_p, intIds_k, weights_k = TI_primal_sPODG_FOTR(lhs, c, a0_primal, f_new, delta_s, modes, kwargs['Nt'],
                                                             kwargs['dt'], v)
            J_s, J_ns, _ = Calc_Cost_sPODG(Vdp, as_p[:-1], qs_target, f_new, intIds_k, weights_k,
                                           kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
                 (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
            if J_s < dJ:  # If Armijo satisfied
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


def Update_Control_sPODG_FRTO_TWBT(f, lhs_p, c_p, Vd_p, a_p, qs_target, delta_s, J_s_prev,
                                   omega_prev, modes, dL_du_s, adjust, v, **kwargs):
    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
    as_p, _, intIds, weights = TI_primal_sPODG_FRTO(lhs_p, c_p, a_p, f_new, delta_s, modes=modes,
                                                    Nt=kwargs['Nt'],
                                                    dt=kwargs['dt'], v=v)
    J_s, J_ns, _ = Calc_Cost_sPODG(Vd_p, as_p[:-1], qs_target, f_new, intIds, weights,
                                   kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
    dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
         (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
    if J_s < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            omega = omega / beta
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            as_p, _, intIds_n, weights_n = TI_primal_sPODG_FRTO(lhs_p, c_p, a_p, f_new, delta_s, modes=modes,
                                                                Nt=kwargs['Nt'],
                                                                dt=kwargs['dt'], v=v
                                                                )
            J_s, J_ns, _ = Calc_Cost_sPODG(Vd_p, as_p[:-1], qs_target, f_new, intIds_n, weights_n,
                                           kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
                 (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
            if J_s >= dJ:
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
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            as_p, _, intIds_k, weights_k = TI_primal_sPODG_FRTO(lhs_p, c_p, a_p, f_new, delta_s, modes=modes,
                                                                Nt=kwargs['Nt'],
                                                                dt=kwargs['dt'], v=v
                                                                )
            J_s, J_ns, _ = Calc_Cost_sPODG(Vd_p, as_p[:-1], qs_target, f_new, intIds_k, weights_k,
                                           kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
                 (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
            if J_s < dJ:  # If Armijo satisfied
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


def Update_Control_PODG_FRTO_TWBT(f, Ar_p, psir_p, V, a_p, qs_target, J_s_prev,
                                  omega_prev, dL_du_s, C, adjust, **kwargs):
    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
    as_ = TI_primal_PODG_FRTO(a_p, f_new, Ar_p, psir_p, kwargs['Nt'], kwargs['dt'])
    J_s, _ = Calc_Cost_PODG(V, as_, qs_target, f_new, C, kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'],
                            kwargs['lamda_l2'], adjust)
    dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
         (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
    if J_s < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            omega = omega / beta
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            as_ = TI_primal_PODG_FRTO(a_p, f_new, Ar_p, psir_p, kwargs['Nt'], kwargs['dt'])
            J_s, _ = Calc_Cost_PODG(V, as_, qs_target, f_new, C, kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'],
                                    kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + (kwargs['delta'] / omega) * L2norm_ROM(
                f_new - f, kwargs['dt'])
            if J_s >= dJ:
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
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            as_ = TI_primal_PODG_FRTO(a_p, f_new, Ar_p, psir_p, kwargs['Nt'], kwargs['dt'])
            J_s, _ = Calc_Cost_PODG(V, as_, qs_target, f_new, C, kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'],
                                    kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + (kwargs['delta'] / omega) * L2norm_ROM(
                f_new - f, kwargs['dt'])
            if J_s < dJ:  # If Armijo satisfied
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


def get_BB_step(fOld, fNew, dL_du_Old, dL_du_New, itr, **kwargs):
    alpha = BarzilaiBorwein(itr, kwargs['dt'], fNew, fOld, dL_du_New, dL_du_Old)
    omega = 1 / alpha

    print(f"Step Size: {omega}")

    return omega


# -------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #


def Update_Control_TWBT_kdv(f, q0, qs_target, J_s_prev, omega_prev, dL_du_s, C, adjust, J_l, params: dict,
                            **kwargs):  # Two-way back tracking

    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
    qs = TI_primal_kdv_expl(q0, f_new, params['D1'], params['D2'], params['D3'],
                            params['B'], params['L'], kwargs['Nx'], kwargs['Nt'], kwargs['dt'],
                            params['c'], params['alpha'], params['omega'], params['gamma'],
                            params['nu'])
    J_s, _ = Calc_Cost(qs, qs_target, f_new, C, kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'], kwargs['lamda_l2'],
                       adjust)
    dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
         (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
    if J_s < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            omega = omega / beta
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            qs = TI_primal_kdv_expl(q0, f_new, params['D1'], params['D2'], params['D3'],
                                    params['B'], params['L'], kwargs['Nx'], kwargs['Nt'], kwargs['dt'],
                                    params['c'], params['alpha'], params['omega'], params['gamma'],
                                    params['nu'])
            J_s, _ = Calc_Cost(qs, qs_target, f_new, C, kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'],
                               kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
                 (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
            if J_s >= dJ:
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
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            qs = TI_primal_kdv_expl(q0, f_new, params['D1'], params['D2'], params['D3'],
                                    params['B'], params['L'], kwargs['Nx'], kwargs['Nt'], kwargs['dt'],
                                    params['c'], params['alpha'], params['omega'], params['gamma'],
                                    params['nu'])
            J_s, _ = Calc_Cost(qs, qs_target, f_new, C, kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'],
                               kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
                 (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
            if J_s < dJ:  # If Armijo satisfied
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


def Update_Control_BB_kdv(fNew, dL_du_New, omega, lamda1):  # Barzilai Borwein
    f_bb_new = prox_l1(fNew - omega * dL_du_New, omega * lamda1)

    return f_bb_new


def Update_Control_PODG_FOTR_RA_TWBT_kdv(f, a0_primal, qs_target, V_p, primal_mat, J_s_prev,
                                         omega_prev, dL_du_s, C, adjust, params_primal, **kwargs):
    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
    as_ = TI_primal_PODG_FOTR_kdv_expl(a0_primal, f_new, primal_mat.D_1r, primal_mat.D_2r, primal_mat.D_3r,
                                       primal_mat.kron_1, primal_mat.kron_2, primal_mat.kron_3,
                                       primal_mat.B_r, primal_mat.L_r, params_primal['c'],
                                       params_primal['alpha'],
                                       params_primal['omega'], params_primal['gamma'],
                                       params_primal['nu'], kwargs['Nt'], kwargs['dt'])
    J_s, _ = Calc_Cost_PODG(V_p, as_, qs_target, f_new, C, kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'],
                            kwargs['lamda_l2'], adjust)
    dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
         (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
    if J_s < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            omega = omega / beta
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            as_ = TI_primal_PODG_FOTR_kdv_expl(a0_primal, f_new, primal_mat.D_1r, primal_mat.D_2r, primal_mat.D_3r,
                                               primal_mat.kron_1, primal_mat.kron_2, primal_mat.kron_3,
                                               primal_mat.B_r, primal_mat.L_r, params_primal['c'],
                                               params_primal['alpha'],
                                               params_primal['omega'], params_primal['gamma'],
                                               params_primal['nu'], kwargs['Nt'], kwargs['dt'])
            J_s, _ = Calc_Cost_PODG(V_p, as_, qs_target, f_new, C, kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'],
                                    kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + (kwargs['delta'] / omega) * L2norm_ROM(
                f_new - f, kwargs['dt'])
            if J_s >= dJ:
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
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            as_ = TI_primal_PODG_FOTR_kdv_expl(a0_primal, f_new, primal_mat.D_1r, primal_mat.D_2r, primal_mat.D_3r,
                                               primal_mat.kron_1, primal_mat.kron_2, primal_mat.kron_3,
                                               primal_mat.B_r, primal_mat.L_r, params_primal['c'],
                                               params_primal['alpha'],
                                               params_primal['omega'], params_primal['gamma'],
                                               params_primal['nu'], kwargs['Nt'], kwargs['dt'])
            J_s, _ = Calc_Cost_PODG(V_p, as_, qs_target, f_new, C, kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'],
                                    kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + (kwargs['delta'] / omega) * L2norm_ROM(
                f_new - f, kwargs['dt'])
            if J_s < dJ:  # If Armijo satisfied
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


def Update_Control_PODG_FRTO_TWBT_kdv(f, primal_mat, V, a_p, qs_target, J_s_prev,
                                      omega_prev, dL_du_s, C, adjust, params_primal, **kwargs):
    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
    as_ = TI_primal_PODG_FRTO_kdv_expl(a_p, f_new, primal_mat.D_1r, primal_mat.D_2r, primal_mat.D_3r,
                                       primal_mat.kron_1, primal_mat.kron_2, primal_mat.kron_3,
                                       primal_mat.B_r, primal_mat.L_r, params_primal['c'],
                                       params_primal['alpha'],
                                       params_primal['omega'], params_primal['gamma'],
                                       params_primal['nu'], kwargs['Nt'], kwargs['dt'])
    J_s, _ = Calc_Cost_PODG(V, as_, qs_target, f_new, C, kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'],
                            kwargs['lamda_l2'], adjust)
    dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
         (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
    if J_s < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            omega = omega / beta
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            as_ = TI_primal_PODG_FRTO_kdv_expl(a_p, f_new, primal_mat.D_1r, primal_mat.D_2r, primal_mat.D_3r,
                                               primal_mat.kron_1, primal_mat.kron_2, primal_mat.kron_3,
                                               primal_mat.B_r, primal_mat.L_r, params_primal['c'],
                                               params_primal['alpha'],
                                               params_primal['omega'], params_primal['gamma'],
                                               params_primal['nu'], kwargs['Nt'], kwargs['dt'])
            J_s, _ = Calc_Cost_PODG(V, as_, qs_target, f_new, C, kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'],
                                    kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + (kwargs['delta'] / omega) * L2norm_ROM(
                f_new - f, kwargs['dt'])
            if J_s >= dJ:
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
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            as_ = TI_primal_PODG_FRTO_kdv_expl(a_p, f_new, primal_mat.D_1r, primal_mat.D_2r, primal_mat.D_3r,
                                               primal_mat.kron_1, primal_mat.kron_2, primal_mat.kron_3,
                                               primal_mat.B_r, primal_mat.L_r, params_primal['c'],
                                               params_primal['alpha'],
                                               params_primal['omega'], params_primal['gamma'],
                                               params_primal['nu'], kwargs['Nt'], kwargs['dt'])
            J_s, _ = Calc_Cost_PODG(V, as_, qs_target, f_new, C, kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'],
                                    kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + (kwargs['delta'] / omega) * L2norm_ROM(
                f_new - f, kwargs['dt'])
            if J_s < dJ:  # If Armijo satisfied
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


def Update_Control_sPODG_FOTR_RA_TWBT_kdv(f, lhs, rhs, rhs_nl, c, a0_primal, qs_target, delta_s, Vdp, J_s_prev,
                                          omega_prev, modes, dL_du_s, C, adjust, **kwargs):
    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
    as_p, intIds, weights = TI_primal_sPODG_FOTR_kdv_expl(lhs, rhs, rhs_nl, c, a0_primal, f_new, delta_s, modes,
                                                          kwargs['Nt'], kwargs['dt'])
    J_s, J_ns, _ = Calc_Cost_sPODG(Vdp, as_p[:-1], qs_target, f_new, C, intIds, weights,
                                   kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
    dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
         (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
    if J_s < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            omega = omega / beta
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            as_p, intIds_n, weights_n = TI_primal_sPODG_FOTR_kdv_expl(lhs, rhs, rhs_nl, c, a0_primal, f_new, delta_s,
                                                                      modes,
                                                                      kwargs['Nt'], kwargs['dt'])
            J_s, J_ns, _ = Calc_Cost_sPODG(Vdp, as_p[:-1], qs_target, f_new, C, intIds_n, weights_n,
                                           kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
                 (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
            if J_s >= dJ:
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
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            as_p, intIds_k, weights_k = TI_primal_sPODG_FOTR_kdv_expl(lhs, rhs, rhs_nl, c, a0_primal, f_new, delta_s,
                                                                      modes,
                                                                      kwargs['Nt'], kwargs['dt'])
            J_s, J_ns, _ = Calc_Cost_sPODG(Vdp, as_p[:-1], qs_target, f_new, C, intIds_k, weights_k,
                                           kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
                 (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
            if J_s < dJ:  # If Armijo satisfied
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


def Update_Control_sPODG_FRTO_TWBT_kdv(f, lhs_p, rhs_p, rhs_nl_p, c_p, Vd_p, a_p, qs_target, delta_s,
                                       J_s_prev,
                                       omega_prev, modes, dL_du_s, C, adjust, **kwargs):
    # Choosing the step size for two-way backtracking
    beta = kwargs['beta']
    omega = omega_prev

    # Checking the Armijo condition
    f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
    as_p, _, intIds, weights = TI_primal_sPODG_FRTO_kdv_expl(lhs_p, rhs_p, rhs_nl_p, c_p, a_p, f_new,
                                                             delta_s, modes, kwargs['Nt'], kwargs['dt'])
    J_s, J_ns, _ = Calc_Cost_sPODG(Vd_p, as_p[:-1], qs_target, f_new, C, intIds, weights,
                                   kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
    dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
         (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
    if J_s < dJ:  # If Armijo satisfied
        n = 0
        while True:
            omega_final = omega
            f_new_final = np.copy(f_new)
            omega = omega / beta
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            as_p, _, intIds_n, weights_n = TI_primal_sPODG_FRTO_kdv_expl(lhs_p, rhs_p, rhs_nl_p, c_p, a_p, f_new,
                                                                         delta_s, modes, kwargs['Nt'], kwargs['dt'])
            J_s, J_ns, _ = Calc_Cost_sPODG(Vd_p, as_p[:-1], qs_target, f_new, C, intIds_n, weights_n,
                                           kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
                 (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
            if J_s >= dJ:
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
            f_new = prox_l1(f - omega * dL_du_s, omega * kwargs['lamda_l1'])
            as_p, _, intIds_k, weights_k = TI_primal_sPODG_FRTO_kdv_expl(lhs_p, rhs_p, rhs_nl_p, c_p, a_p, f_new,
                                                                         delta_s, modes, kwargs['Nt'], kwargs['dt'])
            J_s, J_ns, _ = Calc_Cost_sPODG(Vd_p, as_p[:-1], qs_target, f_new, C, intIds_k, weights_k,
                                           kwargs['dx'], kwargs['dt'], kwargs['lamda_l1'], kwargs['lamda_l2'], adjust)
            dJ = J_s_prev + L2inner_prod(dL_du_s, f_new - f, kwargs['dt']) + \
                 (kwargs['delta'] / omega) * L2norm_ROM(f_new - f, kwargs['dt'])
            if J_s < dJ:  # If Armijo satisfied
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
