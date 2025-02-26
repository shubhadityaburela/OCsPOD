import numpy as np


class advection:
    def __init__(self, Nxi: int, timesteps: int, cfl: float, tilt_from: int, v_x: float, v_x_t: float,
                 variance: float, offset: float) -> None:
        # Assertion statements for checking the sanctity of the input variables
        assert Nxi > 0, f"Please input sensible values for the X grid points"
        assert timesteps >= 0, f"Please input sensible values for time steps"

        # First we define the public variables of the class. All the variables with "__" in front are private variables
        self.X = None
        self.dx = None
        self.t = None
        self.dt = None

        # Private variables
        self.Lxi = 100
        self.Nxi = Nxi
        self.Nt = timesteps
        self.cfl = cfl

        # Order of accuracy for the derivative matrices of the first and second order
        self.firstderivativeOrder = "6thOrder"

        self.v_x = v_x * np.ones(self.Nt)
        self.C = 1.0

        self.v_x_target = self.v_x
        self.v_x_target[tilt_from:] = v_x_t

        self.variance = variance  # Variance of the gaussian for the initial condition
        self.offset = offset  # Offset from where the wave starts

    def Grid(self):
        self.X = np.arange(1, self.Nxi + 1) * self.Lxi / self.Nxi
        self.dx = self.X[1] - self.X[0]

        dt = self.dx * self.cfl / self.C
        self.t = dt * np.arange(self.Nt)
        self.dt = self.t[1] - self.t[0]

        print('dt = ', dt)
        print('Final time : ', self.t[-1])


