import numpy as np
from Helper_sPODG import make_Da


def Calc_Grad(mask, f, qs_adj, **kwargs):
    dL_du = kwargs['lamda'] * f + mask.transpose() @ qs_adj

    return dL_du


def Calc_Grad_PODG(psir, f, as_adj, **kwargs):
    dL_du = kwargs['lamda'] * f + psir.transpose() @ as_adj
    return dL_du


def Calc_Grad_sPODG_FRTO(f, ct, intIds, weights, as_, as_adj, **kwargs):
    as_adj_1 = np.zeros_like(f)
    as_adj_2 = np.zeros_like(f)
    for i in range(kwargs['Nt']):
        m1 = (weights[i] * ct[intIds[i]][0] + (1 - weights[i]) * ct[intIds[i] + 1][0]).transpose()
        m2 = (weights[i] * ct[intIds[i]][1] + (1 - weights[i]) * ct[intIds[i] + 1][1]).transpose()
        Da = make_Da(as_[:, i])
        as_adj_1[:, i] = m1 @ as_adj[:-1, i]
        as_adj_2[:, i] = m2 @ (Da @ as_adj[-1:, i])

    dL_du = kwargs['lamda'] * f + as_adj_1 + as_adj_2

    return dL_du


def Calc_Grad_sPODG_FOTR(f, C_a, intIds, weights, as_adj, **kwargs):
    as_adj_1 = np.zeros_like(f)
    for i in range(kwargs['Nt']):
        c_a = (weights[i] * C_a[intIds[i]] + (1 - weights[i]) * C_a[intIds[i] + 1])
        as_adj_1[:, i] = c_a @ as_adj[:-1, i]

    dL_du = kwargs['lamda'] * f + as_adj_1

    return dL_du
