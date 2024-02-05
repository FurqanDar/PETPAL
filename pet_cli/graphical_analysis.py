"""Collection of functions to perform graphical analysis on tissue activity curves (TACs).

TODO:
    * Check if it makes more sense to lift out the more mathy methods out into a separate module.

"""

__version__ = '0.1'

import numpy as np
import numba
import typing

# TODO: Check if documentation is good.
@numba.njit()
def _line_fitting_make_rhs_matrix_from_xdata(xdata: np.ndarray) -> np.ndarray:
    """Generates the RHS matrix for linear least squares fitting

    Args:
        xdata (np.ndarray): array of independent variable values

    Returns:
        np.ndarray: 2D matrix where first column is `xdata` and the second column is 1s.
        
    """
    out_matrix = np.ones((len(xdata), 2), float)
    out_matrix[:, 0] = xdata
    return out_matrix

# TODO: Check if documentation is good.
@numba.njit()
def fit_line_to_data_using_lls(xdata: np.ndarray, ydata: np.ndarray) -> np.ndarray:
    """Find the linear least squares solution given the x and y variables.
    
    Performs a linear least squares fit to the provided data. Explicitly calls numpy's ``linalg.lstsq`` method by
    constructing the matrix equations. We assume that ``xdata`` and ``ydata`` have the same number of elements.
    
    Args:
        xdata: Array of independent variable values
        ydata: Array of dependent variable values

    Returns:
       Linear least squares solution. (m, b) values
       
    """
    make_2d_matrix = _line_fitting_make_rhs_matrix_from_xdata
    matrix = make_2d_matrix(xdata)
    fit_ans = np.linalg.lstsq(matrix, ydata)[0]
    return fit_ans


@numba.njit()
def cumulative_trapezoidal_integral(xdata: np.ndarray, ydata: np.ndarray, initial: float = 0.0) -> np.ndarray:
    """Calculates the cumulative integral of `ydata` over `xdata` using the trapezoidal rule.
    
    This function is based `heavily` on the ``scipy.integrate.cumtrapz`` implementation.
    `source <https://github.com/scipy/scipy/blob/v0.18.1/scipy/integrate/quadrature.py#L206>`_.
    This implementation only works for 1D arrays and was implemented to work with ``numba``.
    
    Args:
        xdata (np.ndarray): Array for the integration coordinate.
        ydata (np.ndarray): Array for the values to integrate

    Returns:
        np.ndarray: Cumulative integral of ``ydata`` over ``xdata``
    """
    dx = np.diff(xdata)
    cum_int = np.zeros(len(xdata))
    cum_int[0] = initial
    cum_int[1:] = np.cumsum(dx * (ydata[1:] + ydata[:-1]) / 2.0)
    
    return cum_int

# TODO: Add references for the TCMs and Patlak. Could maybe rely on Turku PET Center.
@numba.njit()
def calculate_patlak_x(tac_times: np.ndarray, tac_vals: np.ndarray) -> np.ndarray:
    r"""Calculates the x-variable in Patlak analysis (:math:`\frac{\int_{0}^{T}f(t)\mathrm{d}t}{f(T)}`).
    
    Patlak-Gjedde analysis is a linearization of the 2-TCM with irreversible uptake in the second compartment.
    Therefore, we essentially have to fit a line to some data :math:`y = mx+b`. This function calculates the :math:`x`
    variable for Patlak analysis where,
    .. math::
        x = \frac{\int_{0}_{T} C_\mathrm{P}(t) \mathrm{d}t}{C_\mathrm{P}(t)},
    
    where further :math:`C_\mathrm{P}` is usually the plasma TAC.
    
    Args:
        tac_times (np.ndarray): Array containing the sampled times.
        tac_vals (np.ndarray): Array for activity values at the sampled times.

    Returns:
        np.ndarray: Patlak x-variable. Cumulative integral of activity divided by activity.
    """
    cumulative_integral = cumulative_trapezoidal_integral(xdata=tac_times, ydata=tac_vals, initial=0.0)
    
    return cumulative_integral / tac_vals


@numba.njit()
def get_index_from_threshold(times_in_minutes: np.ndarray, t_thresh_in_minutes: float) -> int:
    """Get the index after which all times are greater than the threshold.

    Args:
        times_in_minutes (np.ndarray): Array containing times in minutes.
        t_thresh_in_minutes (float): Threshold time in minutes.

    Returns:
        int: Index for first time greater than or equal to the threshold time.
        
    Notes:
        If the threshold value is larger than the array, we return -1.
    """
    if t_thresh_in_minutes > np.max(times_in_minutes):
        return -1
    else:
        return np.argwhere(times_in_minutes >= t_thresh_in_minutes)[0, 0]