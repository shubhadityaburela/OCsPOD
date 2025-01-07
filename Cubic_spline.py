import numpy as np
from scipy.sparse import spdiags, diags, linalg, csc_matrix
from scipy.linalg import cholesky


def give_spline_coefficient_matrices(Nxi):
    """
    This function returns the coefficient matrices needed for determining the
    spline coefficients for a cubic spline with periodic boundary conditions.

    On each sub-interval [x_{i-1},x_i] the spline has the form
          f_i(x) = a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3.
    Given a vector y which contains sample data at the grid points with
    uniform grid size h, the spline coefficients may be computed as follows:

    - Solve the linear system M*c = (3/h^2)*D2*y for c
    - Set a=y
    - Set d=(1/(3h))*D1*c
    - Set b=(1/h)*D1*y+(h/3)*A1*c,

    cf. Appendix D.1.1 of the PhD thesis "Energy-based Model Reduction of
    Transport-dominated phenomena".

    Inputs:
    - Nxi: number of grid points

    Outputs:
    - A1: coefficient matrix of size Nxi x Nxi
    - D1: coefficient matrix of size Nxi x Nxi
    - D2: coefficient matrix of size Nxi x Nxi
    - R: Cholesky factor of the matrix M (size Nxi x Nxi)
    """

    # Initialize matrices
    # Create the A1 matrix
    diagonals = np.ones((Nxi, 1)) * np.array([1, 2, 1])
    offsets = [-1, 0, Nxi - 1]
    A1 = spdiags(diagonals.T, offsets, Nxi, Nxi).tocsc()

    # Create the D1 matrix
    diagonals = np.ones((Nxi, 1)) * np.array([-1, 1, -1])
    D1 = spdiags(diagonals.T, offsets, Nxi, Nxi).tocsc()

    # Create the D2 matrix
    diagonals = np.ones((Nxi, 1)) * np.array([1, 1, -2, 1, 1])
    offsets = [-(Nxi - 1), -1, 0, 1, Nxi - 1]
    D2 = spdiags(diagonals.T, offsets, Nxi, Nxi).tocsc()

    # Create the M matrix and compute its Cholesky decomposition
    diagonals = np.ones((Nxi, 1)) * np.array([1, 1, 4, 1, 1])
    M = diags(diagonals.T, offsets, shape=(Nxi, Nxi)).toarray()
    R = csc_matrix(cholesky(M))

    return A1, D1, D2, R


def construct_spline_coeffs_single(qs, A1, D1, D2, R, dxi):
    """
    This function determines the spline coefficients for a given vector or
    matrix of sample data qs, where a cubic spline with periodic boundary
    conditions is assumed.

    On each sub-interval [x_{i-1},x_i] the spline has the form
          f_i(x) = a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3.
    Given a vector y which contains sample data at the grid points with
    uniform grid size h, the spline coefficients may be computed as follows:

    - Solve the linear system M*c = (3/h^2)*D2*y for c
    - Set a=y
    - Set d=(1/(3h))*D1*c
    - Set b=(1/h)*D1*y+(h/3)*A1*c,

    cf. Appendix D.1.1 of the PhD thesis "Energy-based Model Reduction of
    Transport-dominated phenomena".

    Inputs:
    - qs: array of size mxn where m is the number of grid points and n the number of data sets
    - A1: coefficient matrix of size Nxi x Nxi
    - D1: coefficient matrix of size Nxi x Nxi
    - D2: coefficient matrix of size Nxi x Nxi
    - R: Cholesky factor of the matrix M (size Nxi x Nxi)
    - dxi: grid size

    Outputs:
    - splineCoeffs: struct array containing the spline coefficients for each
    data set: splineCoeffs_b, splineCoeffs_c, splineCoeffs_d
    (each of them has the same dimension as qs)
    """
    # compute the spline coefficients:
    spline_coeffs_c = (3 / dxi ** 2) * linalg.spsolve(R, linalg.spsolve(R.T, D2 @ qs))
    spline_coeffs_b = (1 / dxi) * (D1 @ qs) + (dxi / 3) * (A1 @ spline_coeffs_c)
    spline_coeffs_d = (1 / (3 * dxi)) * (D1 @ spline_coeffs_c)

    return spline_coeffs_b, spline_coeffs_c, spline_coeffs_d


def construct_spline_coeffs_multiple(qs, A1, D1, D2, R, dxi):
    """
    This function determines the spline coefficients for a given vector or
    matrix of sample data qs, where a cubic spline with periodic boundary
    conditions is assumed.

    On each sub-interval [x_{i-1},x_i] the spline has the form
          f_i(x) = a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3.
    Given a vector y which contains sample data at the grid points with
    uniform grid size h, the spline coefficients may be computed as follows:

    - Solve the linear system M*c = (3/h^2)*D2*y for c
    - Set a=y
    - Set d=(1/(3h))*D1*c
    - Set b=(1/h)*D1*y+(h/3)*A1*c,

    cf. Appendix D.1.1 of the PhD thesis "Energy-based Model Reduction of
    Transport-dominated phenomena".

    Inputs:
    - qs: array of size mxn where m is the number of grid points and n the number of data sets
    - A1: coefficient matrix of size Nxi x Nxi
    - D1: coefficient matrix of size Nxi x Nxi
    - D2: coefficient matrix of size Nxi x Nxi
    - R: Cholesky factor of the matrix M (size Nxi x Nxi)
    - dxi: grid size

    Outputs:
    - splineCoeffs: struct array containing the spline coefficients for each
    data set: splineCoeffs_b, splineCoeffs_c, splineCoeffs_d
    (each of them has the same dimension as qs)
    """

    n = qs.shape[1]
    spline_coeffs_b = np.zeros_like(qs)
    spline_coeffs_c = np.zeros_like(qs)
    spline_coeffs_d = np.zeros_like(qs)

    c_fac = (3 / dxi ** 2)
    b1_fac = (1 / dxi)
    b2_fac = (dxi / 3)
    d_fac = (1 / (3 * dxi))

    for i in range(n):
        # Assign the solved result to spline_coeffs
        aa = D2 @ qs[:, i]
        aa = linalg.spsolve(R.T, aa)
        spline_coeffs_c[:, i] = c_fac * linalg.spsolve(R, aa)
        spline_coeffs_b[:, i] = b1_fac * (D1 @ qs[:, i]) + b2_fac * (A1 @ spline_coeffs_c[:, i])
        spline_coeffs_d[:, i] = d_fac * (D1 @ spline_coeffs_c[:, i])

    return spline_coeffs_b, spline_coeffs_c, spline_coeffs_d


def shift_matrix_precomputed_coeffs_single(qs, cs, spline_coeffs_b, spline_coeffs_c, spline_coeffs_d, Nxi, dxi):
    """
    This function determines a shifted version of a data set using cubic
    spline interpolation with periodic boundary conditions.

    On each sub-interval [x_{i-1},x_i] the spline has the form
          f_i(x) = a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3.

    Inputs:
    - qs: array of size mxn where m is the number of grid points and n the number of data sets
    - cs: scalar or array of size p or array of size n containing the shift amount(s)
    - splineCoeffs_b, splineCoeffs_c, splineCoeffs_d: containing the spline coefficients for
    each data set: splineCoeffs.b, splineCoeffs.c, splineCoeffs.d
    (each of them has size mxn where m is the number of grid
    points and n the number of data sets)
    - Nxi: number of grid points
    - dxi: grid size

    Outputs:
    - qsShifted: shifted version of original data set(s) (if there are
    multiple data sets, i.e. n>1, then qsShifted is an array of size mxn;
    otherwise qsShifted is an array of size mxp where p is the number of
    shift amounts)
    """

    # Length of computational domain
    L = Nxi * dxi

    # Compute csTilde = cs modulo L
    q1 = np.floor(cs / L)
    csTilde = cs - q1 * L
    # Compute zeta = csTilde modulo dxi
    q2 = np.floor(csTilde / dxi)
    zeta = csTilde - q2 * dxi

    # Compute shifted version of the data set(s)
    qsInterpolated = qs - zeta * spline_coeffs_b + zeta ** 2 * spline_coeffs_c - zeta ** 3 * spline_coeffs_d
    qsShifted = np.concatenate((qsInterpolated[Nxi - int(q2):], qsInterpolated[:Nxi - int(q2)]))

    return qsShifted


def shift_matrix_precomputed_coeffs_multiple(qs, cs, spline_coeffs_b, spline_coeffs_c, spline_coeffs_d, Nxi, dxi):
    """
    This function determines a shifted version of a data set using cubic
    spline interpolation with periodic boundary conditions.

    On each sub-interval [x_{i-1},x_i] the spline has the form
          f_i(x) = a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3.

    Inputs:
    - qs: array of size mxn where m is the number of grid points and n the number of data sets
    - cs: scalar or array of size p or array of size n containing the shift amount(s)
    - splineCoeffs_b, splineCoeffs_c, splineCoeffs_d: containing the spline coefficients for
    each data set: splineCoeffs.b, splineCoeffs.c, splineCoeffs.d
    (each of them has size mxn where m is the number of grid
    points and n the number of data sets)
    - Nxi: number of grid points
    - dxi: grid size

    Outputs:
    - qsShifted: shifted version of original data set(s) (if there are
    multiple data sets, i.e. n>1, then qsShifted is an array of size mxn;
    otherwise qsShifted is an array of size mxp where p is the number of
    shift amounts)
    """

    n = qs.shape[1]

    # Length of computational domain
    L = Nxi * dxi

    # Compute csTilde = cs modulo L
    q1 = np.floor(cs / L)
    csTilde = cs - q1 * L
    # Compute zeta = csTilde modulo dxi
    q2 = np.floor(csTilde / dxi)

    zeta = csTilde - q2 * dxi
    zeta2 = zeta ** 2
    zeta3 = zeta ** 3

    qsShifted = np.zeros((Nxi, n))

    for i in range(n):
        # Compute interpolated data set
        qsInterpolated = qs[:, i] - zeta[i] * spline_coeffs_b[:, i] + zeta2[i] * spline_coeffs_c[:, i] \
                         - zeta3[i] * spline_coeffs_d[:, i]
        # Compute shifted data set
        qsShifted[:, i] = np.concatenate((qsInterpolated[Nxi - int(q2[i]):], qsInterpolated[:Nxi - int(q2[i])]), axis=0)

    return qsShifted



def shifted_U(U, cs, spline_coeffs_b, spline_coeffs_c, spline_coeffs_d, Nxi, dxi):
    """
    This function determines a shifted version of a data set using cubic
    spline interpolation with periodic boundary conditions.

    On each sub-interval [x_{i-1},x_i] the spline has the form
          f_i(x) = a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3.

    Inputs:
    - U: array of size mxr where m is the number of grid points and r the number of modes
    - cs: scalar the shift amount
    - splineCoeffs_b, splineCoeffs_c, splineCoeffs_d: containing the spline coefficients for
    each data set: splineCoeffs.b, splineCoeffs.c, splineCoeffs.d
    (each of them has size mxr where m is the number of grid
    points and r the number of modes)
    - Nxi: number of grid points
    - dxi: grid size

    Outputs:
    - T^z(U): shifted version of stationary modes U (if there are
    multiple modes, i.e. r>1, then T^z(U) is an array of size mxr;)
    """

    r = U.shape[1]

    # Length of computational domain
    L = Nxi * dxi

    # Compute csTilde = cs modulo L
    q1 = np.floor(cs / L)
    csTilde = cs - q1 * L
    # Compute zeta = csTilde modulo dxi
    q2 = np.floor(csTilde / dxi)
    zeta = csTilde - q2 * dxi
    zeta2 = zeta ** 2
    zeta3 = zeta ** 3

    UShifted = np.zeros((Nxi, r))

    for i in range(r):
        # Compute interpolated data set
        UInterpolated = U[:, i] - zeta * spline_coeffs_b[:, i] + zeta2 * spline_coeffs_c[:, i] \
                         - zeta3 * spline_coeffs_d[:, i]
        # Compute shifted data set
        UShifted[:, i] = np.concatenate((UInterpolated[Nxi - int(q2):], UInterpolated[:Nxi - int(q2)]), axis=0)

    return UShifted



def shiftMatrix_derivative_precomputedCoeffs_single(cs, spline_coeffs_b, spline_coeffs_c, spline_coeffs_d, Nxi, dxi):
    """
    This function determines the derivative of the shifted version of a given
    data set with respect to the shift amount, using cubic spline
    interpolation with periodic boundary conditions.

    On each sub-interval [x_{i-1},x_i] the spline has the form
          f_i(x) = a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3
    and its derivative is
          f_i'(x) = b_i+2*c_i(x-x_i)+3*d_i(x-x_i)^2.

    Inputs:
    - cs: scalar or array of size p or array of size n containing the shift amount(s)
    - splineCoeffs_b, splineCoeffs_c, splineCoeffs_d: containing the spline coefficients for
    each data set: splineCoeffs.b, splineCoeffs.c, splineCoeffs.d
    (each of them has size mxn where m is the number of grid
    points and n the number of data sets)
    - Nxi: number of grid points
    - dxi: grid size

    Outputs:
    - qsShifted: derivative of shifted data set(s) with respect to the shift
    amount (if there are multiple data sets, i.e. n>1, then qsShifted is an
    array of size mxn; otherwise qsShifted is an array of size mxp where p is
    the number of shift amounts)
    """

    # Length of computational domain
    L = Nxi * dxi

    # Compute csTilde = cs modulo L
    q1 = np.floor(cs / L)
    csTilde = cs - q1 * L

    # Compute zeta = csTilde modulo dxi
    q2 = np.floor(csTilde / dxi)
    zeta = csTilde - q2 * dxi

    # Compute derivative of shifted data set(s)
    qsInterpolated = - spline_coeffs_b + 2 * zeta * spline_coeffs_c - 3 * zeta ** 2 * spline_coeffs_d
    qsShifted = np.concatenate((qsInterpolated[Nxi - int(q2):], qsInterpolated[:Nxi - int(q2)]))

    return qsShifted


def shiftMatrix_derivative_precomputedCoeffs_multiple(cs, spline_coeffs_b, spline_coeffs_c, spline_coeffs_d, Nxi, dxi):
    """
    This function determines the derivative of the shifted version of a given
    data set with respect to the shift amount, using cubic spline
    interpolation with periodic boundary conditions.

    On each sub-interval [x_{i-1},x_i] the spline has the form
          f_i(x) = a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3
    and its derivative is
          f_i'(x) = b_i+2*c_i(x-x_i)+3*d_i(x-x_i)^2.

    Inputs:
    - cs: scalar or array of size p or array of size n containing the shift amount(s)
    - splineCoeffs_b, splineCoeffs_c, splineCoeffs_d: containing the spline coefficients for
    each data set: splineCoeffs.b, splineCoeffs.c, splineCoeffs.d
    (each of them has size mxn where m is the number of grid
    points and n the number of data sets)
    - Nxi: number of grid points
    - dxi: grid size

    Outputs:
    - qsShifted: derivative of shifted data set(s) with respect to the shift
    amount (if there are multiple data sets, i.e. n>1, then qsShifted is an
    array of size mxn; otherwise qsShifted is an array of size mxp where p is
    the number of shift amounts)
    """

    n = spline_coeffs_b.shape[1]

    # Length of computational domain
    L = Nxi * dxi

    # Compute csTilde = cs modulo L
    q1 = np.floor(cs / L)
    csTilde = cs - q1 * L

    # Compute zeta = csTilde modulo dxi
    q2 = np.floor(csTilde / dxi)
    zeta = csTilde - q2 * dxi
    zeta2 = zeta ** 2

    qsShifted = np.zeros((Nxi, n))

    for i in range(n):
        qsInterpolated = - spline_coeffs_b[:, i] + 2 * zeta[i] * spline_coeffs_c[:, i] - \
                         3 * zeta2[i] * spline_coeffs_d[:, i]
        qsShifted[:, i] = np.concatenate((qsInterpolated[Nxi - int(q2[i]):], qsInterpolated[:Nxi - int(q2[i])]), axis=0)

    return qsShifted



def first_derivative_shifted_U(cs, spline_coeffs_b, spline_coeffs_c, spline_coeffs_d, Nxi, dxi):
    """
    This function determines the derivative of the shifted version of a given
    data set with respect to the shift amount, using cubic spline
    interpolation with periodic boundary conditions.

    On each sub-interval [x_{i-1},x_i] the spline has the form
          f_i(x) = a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3
    and its derivative is
          f_i'(x) = b_i+2*c_i(x-x_i)+3*d_i(x-x_i)^2.

    Inputs:
    - cs: scalar containing the shift amount
    - splineCoeffs_b, splineCoeffs_c, splineCoeffs_d: containing the spline coefficients for
    each data set: splineCoeffs.b, splineCoeffs.c, splineCoeffs.d
    (each of them has size mxr where m is the number of grid
    points and r the number of modes)
    - Nxi: number of grid points
    - dxi: grid size

    Outputs:
    - (T^z)' U: derivative of shifted data set(s) with respect to the shift
    amount
    """

    r = spline_coeffs_b.shape[1]

    # Length of computational domain
    L = Nxi * dxi

    # Compute csTilde = cs modulo L
    q1 = np.floor(cs / L)
    csTilde = cs - q1 * L

    # Compute zeta = csTilde modulo dxi
    q2 = np.floor(csTilde / dxi)
    zeta = csTilde - q2 * dxi
    zeta2 = zeta ** 2

    dUShifted = np.zeros((Nxi, r))

    for i in range(r):
        dUInterpolated = - spline_coeffs_b[:, i] + 2 * zeta * spline_coeffs_c[:, i] - \
                         3 * zeta2 * spline_coeffs_d[:, i]
        dUShifted[:, i] = np.concatenate((dUInterpolated[Nxi - int(q2):], dUInterpolated[:Nxi - int(q2)]), axis=0)

    return dUShifted



def second_derivative_shifted_U(cs, spline_coeffs_b, spline_coeffs_c, spline_coeffs_d, Nxi, dxi):
    """
    This function determines the derivative of the shifted version of a given
    data set with respect to the shift amount, using cubic spline
    interpolation with periodic boundary conditions.

    On each sub-interval [x_{i-1},x_i] the spline has the form
          f_i(x) = a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3
    and its second derivative is
          f_i"(x) = 2*c_i + 6*d_i(x-x_i).

    Inputs:
    - cs: scalar containing the shift amount
    - splineCoeffs_b, splineCoeffs_c, splineCoeffs_d: containing the spline coefficients for
    each data set: splineCoeffs.b, splineCoeffs.c, splineCoeffs.d
    (each of them has size mxr where m is the number of grid
    points and r the number of modes)
    - Nxi: number of grid points
    - dxi: grid size

    Outputs:
    - (T^z)" U: derivative of shifted data set(s) with respect to the shift
    amount
    """

    r = spline_coeffs_b.shape[1]

    # Length of computational domain
    L = Nxi * dxi

    # Compute csTilde = cs modulo L
    q1 = np.floor(cs / L)
    csTilde = cs - q1 * L

    # Compute zeta = csTilde modulo dxi
    q2 = np.floor(csTilde / dxi)
    zeta = csTilde - q2 * dxi

    ddUShifted = np.zeros((Nxi, r))

    for i in range(r):
        ddUInterpolated = 2 * spline_coeffs_c[:, i] - 6 * zeta * spline_coeffs_d[:, i]
        ddUShifted[:, i] = np.concatenate((ddUInterpolated[Nxi - int(q2):], ddUInterpolated[:Nxi - int(q2)]), axis=0)

    return ddUShifted
