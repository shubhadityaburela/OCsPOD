import numpy as np


class advection:
    def __init__(self, Lx: float, Nx: int, timesteps: int, cfl: float, tilt_from: int, v_x: float, v_x_t: float,
                 variance: float, offset: float) -> None:
        # Assertion statements for checking the sanctity of the input variables
        assert Nx > 0, f"Please input sensible values for the X grid points"
        assert timesteps >= 0, f"Please input sensible values for time steps"

        # First we define the public variables of the class. All the variables with "__" in front are private variables
        self.X = None
        self.dx = None
        self.t = None
        self.dt = None

        # Private variables
        self.Lx = Lx
        self.Nx = Nx
        self.Nt = timesteps
        self.cfl = cfl

        # Order of accuracy for the derivative matrices of the first and second order
        self.firstderivativeOrder = "Upwind"

        self.v_x = v_x * np.ones(self.Nt)
        self.C = 1.0

        self.v_x_target = self.v_x.copy()
        self.v_x_target[tilt_from:] = v_x_t
        self.tilt_from = tilt_from
        self.CTC_end_index = None


        self.variance = variance  # Variance of the gaussian for the initial condition
        self.offset = offset  # Offset from where the wave starts

    def Grid(self):
        self.X = np.arange(1, self.Nx + 1) * self.Lx / self.Nx
        self.dx = self.X[1] - self.X[0]

        self.dt = self.dx / self.v_x[0]
        self.t = self.dt * np.arange(self.Nt)

        self.CTC_end_index = np.abs(self.X - self.v_x[0] * self.t[self.tilt_from]).argmin()

        print('dt = ', self.dt)
        print('Final time : ', self.t[-1])



class advection_3:
    def __init__(self, Lx: float, Nx: int, timesteps: int, cfl: float, tilt_from: int, v_x: float, v_x_t: float,
                 variance: float, offset: float) -> None:
        # Assertion statements for checking the sanctity of the input variables
        assert Nx > 0, f"Please input sensible values for the X grid points"
        assert timesteps >= 0, f"Please input sensible values for time steps"

        # First we define the public variables of the class. All the variables with "__" in front are private variables
        self.X = None
        self.dx = None
        self.t = None
        self.dt = None

        # Private variables
        self.Lx = Lx
        self.Nx = Nx
        self.Nt = timesteps
        self.cfl = cfl

        # Order of accuracy for the derivative matrices of the first and second order
        self.firstderivativeOrder = "Upwind"

        self.v_x = v_x * np.ones(self.Nt)
        self.C = 1.0

        self.v_x_target = self.v_x.copy()
        self.v_x_target[tilt_from:3 * tilt_from] = v_x_t
        self.v_x_target[3 * tilt_from:] = - v_x_t / 3
        self.tilt_from = tilt_from
        self.CTC_end_index = None


        self.variance = variance  # Variance of the gaussian for the initial condition
        self.offset = offset  # Offset from where the wave starts

    def Grid(self):
        self.X = np.arange(1, self.Nx + 1) * self.Lx / self.Nx
        self.dx = self.X[1] - self.X[0]

        self.dt = self.dx / self.v_x[0]
        self.t = self.dt * np.arange(self.Nt)

        self.CTC_end_index = np.abs(self.X - self.v_x[0] * self.t[self.tilt_from]).argmin()

        print('dt = ', self.dt)
        print('Final time : ', self.t[-1])


class Korteweg_de_Vries_Burgers:
    def __init__(self, Nx: int, timesteps: int, cfl: float, v_x: float, variance: float, offset: float) -> None:
        # Assertion statements for checking the sanctity of the input variables
        assert Nx > 0, f"Please input sensible values for the X grid points"
        assert timesteps >= 0, f"Please input sensible values for time steps"

        # First we define the public variables of the class. All the variables with "__" in front are private variables
        self.X = None
        self.dx = None
        self.t = None
        self.dt = None

        # Private variables
        self.Lx = 200000  # (200 km)
        self.Nx = Nx
        self.Nt = timesteps
        self.cfl = cfl

        # Order of accuracy for the derivative matrices of the first and second order
        self.firstderivativeOrder = "6thOrder"

        self.v_x = v_x * np.ones(self.Nt)
        self.C = 200.0   # 200 m/s

        self.variance = variance  # Variance of the gaussian for the initial condition
        self.offset = offset  # Offset from where the wave starts

    def Grid(self):
        self.X = np.arange(1, self.Nx + 1) * self.Lx / self.Nx
        self.dx = self.X[1] - self.X[0]

        dt = self.dx * self.cfl / self.C
        self.t = dt * np.arange(self.Nt)
        self.dt = dt

        print(f"dx = {self.dx} meters")
        print(f"dt = {dt} seconds")
        print('Final time : ', self.t[-1])



class Korteweg_de_Vries:
    def __init__(self, Nx: int, timesteps: int, cfl: float, v_x: float, offset: float) -> None:
        # Assertion statements for checking the sanctity of the input variables
        assert Nx > 0, f"Please input sensible values for the X grid points"
        assert timesteps >= 0, f"Please input sensible values for time steps"

        # First we define the public variables of the class. All the variables with "__" in front are private variables
        self.X = None
        self.dx = None
        self.t = None
        self.dt = None

        # Private variables
        self.Lx = 80
        self.Nx = Nx
        self.Nt = timesteps
        self.cfl = cfl

        # Order of accuracy for the derivative matrices of the first and second order
        self.firstderivativeOrder = "6thOrder"

        self.v_x = v_x * np.ones(self.Nt)
        self.v_x_target = v_x * np.ones(self.Nt)
        self.c = 2 / 3
        self.C = 1.0

        self.offset = offset  # Offset from where the wave starts

    def Grid(self):
        self.X = np.arange(1, self.Nx + 1) * self.Lx / self.Nx
        self.dx = self.X[1] - self.X[0]

        dt = self.dx * self.cfl / self.C
        self.t = dt * np.arange(self.Nt)
        self.dt = dt

        print(f"dx = {self.dx} meters")
        print(f"dt = {dt} seconds")
        print('Final time : ', self.t[-1])