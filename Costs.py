from Helper import *


def Calc_Cost(q, q_target, f, **kwargs):
    q_res = np.copy(q - q_target)
    cost = 1 / 2 * (L2norm_FOM(q_res, **kwargs)) + (kwargs['lamda'] / 2) * (L2norm_ROM(f, **kwargs))

    return cost


def Calc_Cost_PODG(V, as_, qs_target, f, **kwargs):
    q_res = np.copy(V @ as_ - qs_target)

    cost = 1 / 2 * (L2norm_FOM(q_res, **kwargs)) + (kwargs['lamda'] / 2) * (L2norm_ROM(f, **kwargs))

    return cost


def Calc_Cost_sPODG(V, as_, qs_target, f, intIds, weights, **kwargs):
    q = np.zeros_like(qs_target)
    for i in range(f.shape[1]):
        V_delta = weights[i] * V[intIds[i]] + (1 - weights[i]) * V[intIds[i] + 1]
        q[:, i] = V_delta @ as_[:-1, i]

    q_res = np.copy(q - qs_target)

    cost = 1 / 2 * (L2norm_FOM(q_res, **kwargs)) + (kwargs['lamda'] / 2) * (L2norm_ROM(f, **kwargs))

    return cost


def Calc_Cost_sPODG_newcost(as_, as_target, z_target, f, **kwargs):
    a_res = np.copy(as_[:-1, :] - as_target)
    z_res = np.copy(as_[-1:, :] - z_target)

    cost = 1 / 2 * (L2norm_ROM(a_res, **kwargs)) + 1 / 2 * (L2norm_ROM(z_res, **kwargs)) \
           + (kwargs['lamda'] / 2) * (L2norm_ROM(f, **kwargs))

    return cost




########################################################################################################################
import sys
sys.path.append('./sPOD/lib/')
from transforms import Transform
def calc_shift_matrix(V, as_, **kwargs):
    V_delta = []
    z = as_[-1, :]

    Nx = kwargs['Nx']
    Nt = kwargs['Nt']

    data_shape = [Nx, 1, 1, Nt]
    dx = kwargs['dx']
    L = [100.0]
    # Create the transformations
    trafo_1 = Transform(data_shape, L, shifts=z,
                        dx=[dx],
                        use_scipy_transform=False,
                        interp_order=5)
    for i in range(Nt):
        V_delta.append(trafo_1.shifts_pos[i] @ V)

    return V_delta


def Calc_Cost_sPODG_tmp(V_delta, as_, qs_target, f, **kwargs):
    q = np.zeros_like(qs_target)
    for i in range(f.shape[1]):
        q[:, i] = V_delta[i] @ as_[:-1, i]

    q_res = np.copy(q - qs_target)

    cost = 1 / 2 * (L2norm_FOM(q_res, **kwargs)) + (kwargs['lamda'] / 2) * (L2norm_ROM(f, **kwargs))

    return cost

