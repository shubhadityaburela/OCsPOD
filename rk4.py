from Helper import *


def rk4(RHS: callable,
        q0: np.ndarray,
        u1: np.ndarray,
        u2: np.ndarray,
        dt,
        *args) -> np.ndarray:
    k1 = RHS(q0, u1, *args)
    k2 = RHS(q0 + dt / 2 * k1, (u1 + u2) / 2, *args)
    k3 = RHS(q0 + dt / 2 * k2, (u1 + u2) / 2, *args)
    k4 = RHS(q0 + dt * k3, u2, *args)

    u1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return u1


def rk4_adj(RHS: callable,
            q0: np.ndarray,
            a1: np.ndarray,
            a2: np.ndarray,
            b1: np.ndarray,
            b2: np.ndarray,
            dt,
            *args) -> np.ndarray:
    k1 = RHS(q0, a1, b1, *args)
    k2 = RHS(q0 + dt / 2 * k1, (a1 + a2) / 2, (b1 + b2) / 2, *args)
    k3 = RHS(q0 + dt / 2 * k2, (a1 + a2) / 2, (b1 + b2) / 2, *args)
    k4 = RHS(q0 + dt * k3, a2, b2, *args)

    u1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return u1


def rk4_(RHS: callable,
         q0: np.ndarray,
         u1: np.ndarray,
         u2: np.ndarray,
         dt,
         *args):
    k1 = RHS(q0, u1, *args)
    k2 = RHS(q0 + dt / 2 * k1, (u1 + u2) / 2, *args)
    k3 = RHS(q0 + dt / 2 * k2, (u1 + u2) / 2, *args)
    k4 = RHS(q0 + dt * k3, u2, *args)

    u1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return u1, [k1, k2, k3, k4]


def rk4__(RHS: callable,
          q0: np.ndarray,
          u1: np.ndarray,
          u2: np.ndarray,
          a1: np.ndarray,
          a2: np.ndarray,
          b1: np.ndarray,
          b2: np.ndarray,
          dt,
          q0_dot,
          z_dot,
          *args):
    k1 = RHS(q0, u1, a1, b1, q0_dot[3], z_dot[3], *args)
    k2 = RHS(q0 + dt / 2 * k1, (u1 + u2) / 2, (a1 + a2) / 2, (b1 + b2) / 2, q0_dot[2], z_dot[2], *args)
    k3 = RHS(q0 + dt / 2 * k2, (u1 + u2) / 2, (a1 + a2) / 2, (b1 + b2) / 2, q0_dot[1], z_dot[1], *args)
    k4 = RHS(q0 + dt * k3, u2, a2, b2, q0_dot[0], z_dot[0], *args)

    u1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return u1
