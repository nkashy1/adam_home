import numpy as np
from scipy.linalg import cholesky


def sigma_pts_ddt(x, Pxx, h=np.sqrt(3.0)):
    """Get sigma points for the Divided Difference Transform

    DDT is a method for propagating uncertainty of an
    initial Gaussian distribution

    X~N(x, Pxx)

    args:
        x: numpy array of initial state mean
        Pxx: numpy matrix of initial covariance
        h = step size used in DDT (default = sqrt(3))

    out:
        sig_pts: list of numpy arrays
            sigma points are in format (2D case):
                sig_pts[0] = x + h * Sxx[:, 0]
                sig_pts[1] = x - h * Sxx[:, 0]
                sig_pts[2] = x + h * Sxx[:, 1]
                sig_pts[3] = x - h * Sxx[:, 1]
                sig_pts[4] = x

    """
    # Lower triangular matrix such that P = Sxx * transpose(Sxx)
    Sxx = cholesky(Pxx, lower=True)
    sig_pts = []
    for col in Sxx.T:  # loop over columns
        sig_pts.append(x + (h * col))
        sig_pts.append(x - (h * col))
    sig_pts.append(np.array(x))
    return sig_pts


def mean_cov_ddt2(sig_pts, h=np.sqrt(3.0)):
    """Compute mean and covariance from sigma point of the
    second order Divided Difference Transform

    Y = f(X) where f is any nonlinear function acting
    on a multivariate gaussian initial condition
    X~N(x, Pxx)
    Estimate Y~N(y, Pyy)

    This function computes mean and covariance of y
    using sigma points of x propagated through f

    args:
        sig_pts: list of numpy arrays of sigma points
            For a 2d x:
            sig_pts[0] = f(x + h * Sxx[:, 0])
            sig_pts[1] = f(x - h * Sxx[:, 0])
            sig_pts[2] = f(x + h * Sxx[:, 1])
            sig_pts[3] = f(x - h * Sxx[:, 1])
            sig_pts[4] = f(x)
        h = step size used in DDT (default = sqrt(3))
    out:
        y: estimate of mean of Y
        Pyy: estimate of covariance of Y

    """
    x_vars_count = int((len(sig_pts) - 1) / 2)
    y_vars_count = len(sig_pts[0])
    y_x = sig_pts[-1]  # f(x)
    # Covariance related
    Syy_ord_1 = np.zeros([y_vars_count, x_vars_count])
    Syy_ord_2 = np.zeros_like(Syy_ord_1)
    # TODO Easy to read but not computationally efficient or elegent
    # Mean related
    y_mean = ((h ** 2 - x_vars_count) / (h ** 2)) * y_x
    for var in range(0, x_vars_count):
        y_xpdx = sig_pts[2 * var]  # f(x+ dx)
        y_xmdx = sig_pts[2 * var + 1]  # f(x - dx)
        # Mean related
        y_mean += (y_xpdx + y_xmdx) / (2.0 * h ** 2)
        # Covariance related
        Syy_ord_1[:, var] = (y_xpdx - y_xmdx) / (2.0 * h)
        Syy_ord_2[:, var] = (y_xpdx + y_xmdx - (2.0 * y_x)) * (
            np.sqrt(h ** 2 - 1.0) / (2.0 * h ** 2)
        )
    # [Syy_ord_1, Syy_ord_2]
    Syy_rect = np.concatenate((Syy_ord_1, Syy_ord_2), axis=1)
    # TODO compute Sxx using Householder transformation
    # We just want the covariance matrix, so no need to find square-root
    Pyy = np.dot(Syy_rect, Syy_rect.T)
    return y_mean, Pyy


def propagate_mean_cov_ddt(f_x, x, P_xx, h=np.sqrt(3)):
    """Compute mean and covariance using the
    second order Divided Difference Transform

    Y = f(X) where f is any nonlinear function acting
    on a multivariate gaussian initial condition
    X~N(x, Pxx)
    Estimate Y~N(y, Pyy)

    This function computes mean and covariance of y
    using sigma points of x propagated through f

    N.B. f_x is a function that takes vectorial numpy array inputs
    i.e. Y = [[y0], [y1], ..., [yN]] (2D np array where each aray is a sample)
         X = [[x0], [x1], ..., [xN]] (2D np array where each aray is a sample)
         Y = f_x(X)

    args:
        f_x: function input
        h = step size used in DDT (default = sqrt(3))
    out:
        y: estimate of mean of Y
        Pyy: estimate of covariance of Y

    """
    x_sigma_pts = sigma_pts_ddt(x, P_xx, h)
    y_sigma_pts = f_x(x_sigma_pts)
    y, P_yy = mean_cov_ddt2(y_sigma_pts, h)
    return (y, P_yy)


def bhattacharya_distance_mvn(mu_1, mu_2, P_1, P_2):
    """Compute the bhattacharya distance between two multivariate gaussian
    distributions

    x1 ~ N(mu_1, P_1)
    x2 ~ N(mu_2, P_2)

    TODO Test:
    bhattacharya_distance_mvn(
        np.array([1., 1.]), np.array([2., 5.]),
        np.array([np.array([1.,2.]), np.array([2.,5.])]), 
        np.array([np.array([3.,4.]), np.array([4.,7.])])
    ) = 0.7302799995588631

    Args:
        mu_1: Mean 1 (numpy array)
        mu_2: Mean 2 (numpy array)
        P_1: Cov 1 (2D numpy array)
        P_2: Cov 2 (2D numpy array)

    Out:
        distance: Bhattacharya distance (float)

    """
    P = 0.5 * (P_1 + P_2)
    mu = mu_1 - mu_2
    # mu.T * inv(P) * mu
    p1 = np.dot(np.dot(mu.T, np.linalg.inv(P)), mu)
    # ln(P/sqrt(det(P1)*det(P2)))
    p2 = np.log(
        np.linalg.det(P) /
        np.sqrt(np.linalg.det(P_1) * np.linalg.det(P_2))
    )
    # (1/8)*p1 + (1/2)*p2
    distance = 0.125 * p1 + 0.5 * p2
    return distance
