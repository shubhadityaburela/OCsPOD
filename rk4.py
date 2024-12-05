from Helper import *


def rk4(RHS: callable,
        q0: np.ndarray,
        u1: np.ndarray,
        u2: np.ndarray,
        dt,
        *args) -> np.ndarray:
    u_mid = (u1 + u2) / 2
    k1 = RHS(q0, u1, *args)
    k2 = RHS(q0 + dt / 2 * k1, u_mid, *args)
    k3 = RHS(q0 + dt / 2 * k2, u_mid, *args)
    k4 = RHS(q0 + dt * k3, u2, *args)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1


def rk2(RHS: callable,
        q0: np.ndarray,
        u1: np.ndarray,
        u2: np.ndarray,
        dt,
        *args) -> np.ndarray:
    k1 = RHS(q0, u1, *args)
    k2 = RHS(q0 + dt / 2 * k1, (u1 + u2) / 2, *args)

    q1 = q0 + dt * k2

    return q1


def exp_eul(RHS: callable,
            q0: np.ndarray,
            u: np.ndarray,
            dt,
            *args) -> np.ndarray:
    return q0 + dt * RHS(q0, u, *args)


def rk4_rpr(RHS: callable,
            q0: np.ndarray,
            u1: np.ndarray,
            u2: np.ndarray,
            dt,
            *args):
    u_mid = (u1 + u2) / 2
    k1, i, w = RHS(q0, u1, *args)
    k2, _, _ = RHS(q0 + dt / 2 * k1, u_mid, *args)
    k3, _, _ = RHS(q0 + dt / 2 * k2, u_mid, *args)
    k4, _, _ = RHS(q0 + dt * k3, u2, *args)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1, i, w


def rk4_rpr_dot(RHS: callable,
                q0: np.ndarray,
                u1: np.ndarray,
                u2: np.ndarray,
                dt,
                *args):
    u_mid = (u1 + u2) / 2
    k1, i, w = RHS(q0, u1, *args)
    k2, _, _ = RHS(q0 + dt / 2 * k1, u_mid, *args)
    k3, _, _ = RHS(q0 + dt / 2 * k2, u_mid, *args)
    k4, _, _ = RHS(q0 + dt * k3, u2, *args)

    q1_dot = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    q1 = q0 + dt * q1_dot

    return q1, i, w, q1_dot


def exp_eul_rpr(RHS: callable,
                q0: np.ndarray,
                u: np.ndarray,
                dt,
                *args):
    k1, i, w = RHS(q0, u, *args)
    return q0 + dt * k1, i, w


def rk4_adj(RHS: callable,
            q0: np.ndarray,
            a1: np.ndarray,
            a2: np.ndarray,
            b1: np.ndarray,
            b2: np.ndarray,
            dt,
            *args) -> np.ndarray:
    a_mid = (a1 + a2) / 2
    b_mid = (b1 + b2) / 2
    k1 = RHS(q0, a1, b1, *args)
    k2 = RHS(q0 + dt / 2 * k1, a_mid, b_mid, *args)
    k3 = RHS(q0 + dt / 2 * k2, a_mid, b_mid, *args)
    k4 = RHS(q0 + dt * k3, a2, b2, *args)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1


def rk2_adj(RHS: callable,
            q0: np.ndarray,
            a1: np.ndarray,
            a2: np.ndarray,
            b1: np.ndarray,
            b2: np.ndarray,
            dt,
            *args) -> np.ndarray:
    k1 = RHS(q0, a1, b1, *args)
    k2 = RHS(q0 + dt / 2 * k1, (a1 + a2) / 2, (b1 + b2) / 2, *args)

    q1 = q0 + dt * k2

    return q1


def exp_eul_adj(RHS: callable,
                q0: np.ndarray,
                a: np.ndarray,
                b: np.ndarray,
                dt,
                *args) -> np.ndarray:
    return q0 + dt * RHS(q0, a, b, *args)


def rk4_rpr_(RHS: callable,
             q0: np.ndarray,
             u1: np.ndarray,
             u2: np.ndarray,
             dt,
             *args):
    u_mid = (u1 + u2) / 2
    k1, i, w = RHS(q0, u1, *args)
    k2, _, _ = RHS(q0 + dt / 2 * k1, u_mid, *args)
    k3, _, _ = RHS(q0 + dt / 2 * k2, u_mid, *args)
    k4, _, _ = RHS(q0 + dt * k3, u2, *args)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1, np.stack([k1, k2, k3, k4]), i, w


def rk4_radj(RHS: callable,
             q0: np.ndarray,
             a1: np.ndarray,
             a2: np.ndarray,
             b1: np.ndarray,
             b2: np.ndarray,
             dt,
             *args) -> np.ndarray:
    a_mid = (a1 + a2) / 2
    b_mid = (b1 + b2) / 2
    k1 = RHS(q0, a1, b1, *args)
    k2 = RHS(q0 + dt / 2 * k1, a_mid, b_mid, *args)
    k3 = RHS(q0 + dt / 2 * k2, a_mid, b_mid, *args)
    k4 = RHS(q0 + dt * k3, a2, b2, *args)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1



def rk4_radj_(RHS: callable,
          q0: np.ndarray,
          u1: np.ndarray,
          u2: np.ndarray,
          a1: np.ndarray,
          a2: np.ndarray,
          b1: np.ndarray,
          b2: np.ndarray,
          q_dot: np.ndarray,
          dt,
          *args):
    u_mid = (u1 + u2) / 2
    a_mid = (a1 + a2) / 2
    b_mid = (b1 + b2) / 2
    k1 = RHS(q0, u1, a1, b1, q_dot[3], *args)
    k2 = RHS(q0 + dt / 2 * k1, u_mid, a_mid, b_mid, q_dot[2], *args)
    k3 = RHS(q0 + dt / 2 * k2, u_mid, a_mid, b_mid, q_dot[1], *args)
    k4 = RHS(q0 + dt * k3, u2, a2, b2, q_dot[0], *args)

    q1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return q1
