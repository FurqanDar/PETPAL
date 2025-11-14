"""
This module provides functionalities for fitting Tissue Compartment Models (TCM) to Time Activity Curves (TAC)
using various methods.

It includes classes that handle different parts of the TAC fitting process:
    - :class:`TACFitter`: The primary class for fitting TCMs to TACs. It provides utility methods to prepare data,
      set up fitting parameters, and perform the curve fitting. This class allows fitting based on various TCM functions
      such as one-tissue compartment model (1TCM), 2TCM, and others.
    - :class:`TACFitterWithoutBloodVolume`: A subclass of TACFitter designed for scenarios when there is no signal
      contribution from blood volume in the TAC. It utilises the functionalities of :class:`TACFitter` and modifies
      certain methods to exclude the blood volume parameter.

Functions and methods in this module use :mod:`numpy` and :mod:`scipy` packages for data manipulation and optimization
of the fitting process.

Please refer to the documentation of each class for more detailed information.

See Also:
    * :mod:`petpal.kinetic_modeling.tcms_as_convolutions`
    * :mod:`petpal.input_function.blood_input`
    
"""
import inspect
import json
import os
import warnings
from typing import Callable, Union
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit as sp_cv_fit
import lmfit

from . import tcms_as_convolutions as pet_tcms
from ..input_function import blood_input as pet_bld
from ..utils.time_activity_curve import (TimeActivityCurve,
                                         safe_load_tac,
                                         MultiTACAnalysisMixin,
                                         get_frame_index_pairs_from_fine_times)
from ..utils.scan_timing import ScanTimingInfo

def _get_fitting_params_for_tcm_func(f: Callable) -> list:
    r"""
    Fetches the parameter names from the function signature of a given Tissue Compartment Model (TCM) function. The
    functions can be one of the following:
    
    * :func:`gen_tac_1tcm_cpet_from_tac<petpal.tcms_as_convolutions.gen_tac_1tcm_cpet_from_tac>`
    * :func:`gen_tac_2tcm_with_k4zero_cpet_from_tac<petpal.tcms_as_convolutions.gen_tac_2tcm_with_k4zero_cpet_from_tac>`
    * :func:`gen_tac_2tcm_cpet_from_tac<petpal.tcms_as_convolutions.gen_tac_2tcm_cpet_from_tac>`

    Args:
        f (Callable): TCM function.

    Returns:
        list: List of parameter names.
        
    """
    return list(inspect.signature(f).parameters.keys())[2:]


def _get_number_of_fit_params_for_tcm_func(f: Callable) -> int:
    r"""
    Counts the number of fitting parameters for a given Tissue Compartment Model (TCM) function. The
    functions can be one of the following:
    
    * :func:`gen_tac_1tcm_cpet_from_tac<petpal.tcms_as_convolutions.gen_tac_1tcm_cpet_from_tac>`
    * :func:`gen_tac_2tcm_with_k4zero_cpet_from_tac<petpal.tcms_as_convolutions.gen_tac_2tcm_with_k4zero_cpet_from_tac>`
    * :func:`gen_tac_2tcm_cpet_from_tac<petpal.tcms_as_convolutions.gen_tac_2tcm_cpet_from_tac>`

    Args:
        f (Callable): TCM function.

    Returns:
        int: Number of fitting parameters.
    """
    return len(_get_fitting_params_for_tcm_func(f))

class TcmModelConfig:
    r"""
    Base configuration class for Tissue Compartment Model (TCM) specifications.

    This class encapsulates the essential properties of a TCM including the model function,
    parameter names, default bounds, and pretty-printed parameter names for visualization.
    It provides utilities for normalizing model names and resolving model names to their
    corresponding functions.

    Attributes:
        func (Callable): The TCM function implementation.
        param_names (list[str]): List of parameter names (e.g., ['k1', 'k2', 'vb']).
        pretty_param_names (list[str]): List of formatted parameter names for display (e.g., [r'$K_1$', r'$k_2$']).
        default_bounds (np.ndarray): Default bounds for parameters with shape (num_params, 3) where each row
            contains [initial_guess, lower_bound, upper_bound].
        num_params (int): Number of parameters in the model.

    See Also:
        * :class:`~.ConvTcmModelConfig`
        * :class:`~.FrameAvgdTcmModelConfig`
    """
    def __init__(self, func: Callable, param_names: list[str], default_bounds: np.ndarray, pretty_param_names: list[str] | None = None):
        r"""
        Initialize a TCM model configuration.

        Args:
            func (Callable): The tissue compartment model function.
            param_names (list[str]): List of parameter names.
            default_bounds (np.ndarray): Default parameter bounds array with shape (num_params, 3).
            pretty_param_names (list[str] or None, optional): Pretty-printed parameter names for display.
                If None, uses param_names. Defaults to None.
        """
        self.func = func
        self.param_names = param_names
        self.pretty_param_names = pretty_param_names if pretty_param_names is not None else param_names
        self.default_bounds = default_bounds
        self.num_params = len(param_names)

    _NAME_TO_FUNC: dict[str, Callable] = {}

    @staticmethod
    def normalize_name(name: str) -> str:
        r"""
        Normalize a model name to a standard format.

        Converts the name to lowercase and replaces spaces and underscores with hyphens
        for consistent model name lookup.

        Args:
            name (str): The model name to normalize.

        Returns:
            str: The normalized model name.

        Example:
            .. code-block:: python

                from petpal.kinetic_modeling.tac_fitting import TcmModelConfig
                TcmModelConfig.normalize_name("Serial 2TCM")  # Returns "serial-2tcm"
                TcmModelConfig.normalize_name("1TCM")  # Returns "1tcm"
        """
        return name.lower().replace(' ', '_').replace('_', '-')

    @classmethod
    def valid_model_names(cls) -> list[str]:
        r"""
        Get a sorted list of all valid model names.

        Returns:
            list[str]: Sorted list of valid model name strings.
        """
        return sorted(set(cls._NAME_TO_FUNC.keys()))

    @classmethod
    def resolve_model_name(cls, model_name: str) -> Callable:
        r"""
        Resolve a model name string to its corresponding TCM function.

        Args:
            model_name (str): The name of the compartment model.

        Returns:
            Callable: The TCM function corresponding to the model name.

        Raises:
            ValueError: If the model_name does not correspond to a known model.

        Example:
            .. code-block:: python

                from petpal.kinetic_modeling.tac_fitting import TcmModelConfig
                func = TcmModelConfig.resolve_model_name("2tcm")
                # Returns the 2TCM function
        """
        norm_name = cls.normalize_name(model_name)
        try:
            return cls._NAME_TO_FUNC[norm_name]
        except KeyError as err:
            valid_model_names = "', '".join(cls.valid_model_names())
            raise ValueError(f"Unknown compartment model '{model_name}'. Valid options: '{valid_model_names}'") from err


class ConvTcmModelConfig(TcmModelConfig):
    r"""
    Configuration for convolution-based TCM models.

    This subclass of :class:`~.TcmModelConfig` provides configurations specific to TCM models
    that use convolution-based implementations. Supported models include 1TCM, 2TCM with
    k4=0, and serial 2TCM.

    Attributes:
        _NAME_TO_FUNC (dict[str, Callable]): Mapping of normalized model names to their
            corresponding convolution-based TCM functions.

    See Also:
        * :class:`~.TcmModelConfig`
        * :class:`~.FrameAvgdTcmModelConfig`
        * :mod:`petpal.kinetic_modeling.tcms_as_convolutions`
    """
    _NAME_TO_FUNC: dict[str, Callable] = {
        '1tcm'       : pet_tcms.gen_tac_1tcm_cpet_from_tac,
        '2tcm-k4zero': pet_tcms.gen_tac_2tcm_with_k4zero_cpet_from_tac,
        '2tcm'       : pet_tcms.gen_tac_2tcm_cpet_from_tac,
        'serial-2tcm': pet_tcms.gen_tac_2tcm_cpet_from_tac
        }


_CONV_TCM_MODELS_CONFIGS = {
    pet_tcms.gen_tac_1tcm_cpet_from_tac            : ConvTcmModelConfig(
            func=pet_tcms.gen_tac_1tcm_cpet_from_tac,
            param_names=['k1', 'k2', 'vb'],
            default_bounds=np.array([
                [0.2, 1e-8, 0.5],  # k1
                [0.1, 1e-8, 0.5],  # k2
                [0.05, 1e-8, 0.5]  # vb
                ])
            ),
    pet_tcms.gen_tac_2tcm_with_k4zero_cpet_from_tac: ConvTcmModelConfig(
            func=pet_tcms.gen_tac_2tcm_with_k4zero_cpet_from_tac,
            param_names=['k1', 'k2', 'k3', 'vb'],
            default_bounds=np.array([
                [0.2, 1e-8, 0.5],  # k1
                [0.1, 1e-8, 0.5],  # k2
                [0.1, 1e-8, 0.5],  # k3
                [0.05, 1e-8, 0.5]  # vb
                ])
            ),
    pet_tcms.gen_tac_2tcm_cpet_from_tac            : ConvTcmModelConfig(
            func=pet_tcms.gen_tac_2tcm_cpet_from_tac,
            param_names=['k1', 'k2', 'k3', 'k4', 'vb'],
            default_bounds=np.array([
                [0.2, 1e-8, 0.5],  # k1
                [0.1, 1e-8, 0.5],  # k2
                [0.1, 1e-8, 0.5],  # k3,
                [0.01, 1e-8, 0.5], # k4
                [0.05, 1e-8, 0.5]  # vb
                ])
            )
    }

class TACFitter(object):
    r"""
    A class used for fitting Tissue Compartment Models(TCM) to Time Activity Curves (TAC).

    It facilitates and simplifies the curve fitting process of TCM functions to TAC data. The class
    takes in raw TAC data for the plasma and tissue as input, and provides numerous utility methods
    to prepare data, set up fitting parameters, and perform the curve fitting. The resample method ensures
    data is appropriate for curve fitting by interpolating the TAC data over a regular time grid, and includes a
    time=0 data-point to the TACs if necessary.

    The class provides multiple options for setting up weights for the curve fitting residuals and for providing
    initial guesses and setting up bounds for the fitting parameters of the TCM function.

    Allows fitting on the basis of various TCM functions like one-tissue compartment model (1TCM), 2TCM, and others.

    Attributes:
        resample_times (np.ndarray): Times at which TACs are resampled.
        resampled_t_tac (np.ndarray): Tissue TAC values resampled at these times.
        p_tac_vals (np.ndarray): Plasma TAC values used for feeding to TCM function.
        raw_t_tac (np.ndarray): Raw TAC times for tissue, fed at initialization.
        weights (np.ndarray): Weights for handling residuals during the optimization process.
        tgt_tac_vals (np.ndarray): Tissue TAC values to fit TCM model.
        fit_param_number (int): Number of fitting parameters in the TCM function.
        initial_guesses (np.ndarray): Initial guesses for all the parameters for curve fitting.
        bounds_hi (np.ndarray): Upper bounds for all the parameters for curve fitting.
        fit_results (np.optimize.OptimizeResult): The results of the fit, including optimized parameters and covariance
            matrix.
        fit_param_names (List[str]): Names of fitting parameters in the TCM function.
        raw_p_tac (np.ndarray): Raw TAC times for plasma, fed at initialization.
        resampled_p_tac (np.ndarray): Plasma TAC values resampled on these times.
        sanitized_t_tac (np.ndarray): Sanitized version of tissue TAC times.
        bounds_lo (np.ndarray): Lower bounds for all the parameters for curve fitting.
        bounds (np.ndarray): Bounds for each parameter for curve fitting.
        max_func_evals (int): Maximum number of function evaluations (iterations) for the optimization process.
        tcm_func (Callable): The tissue compartment model (TCM) function to fit.
        sanitized_p_tac (np.ndarray): Sanitized version of plasma TAC times.
        delta_t (float): Delta between the newly created time steps in resampled times.
        
    Example:
        In the following quick example, ``tTAC`` represents a tissue TAC (``[times, values]``) and ``pTAC`` represents the
        input function (``[times, values]``). Furthermore, we want to fit the provided ``tTAC`` with a 2TCM.
        
        .. code-block:: python
        
            import petpal.kinetic_modeling.tcms_as_convolutions as pet_tcm
            import petpal.kinetic_modeling.tac_fitting as pet_fit
            import numpy as np
            
            tcm_func = pet_tcm.gen_tac_2tcm_cpet_from_tac
            fit = pet_fit.TACFitter(pTAC=pTAC, tTAC=tTAC, tcm_func=tcm_func, resample_num=512)
            fit.run_fit()
            fit_params = fit.fit_results[0]
            print(fit_params.round(3))
    
        In the following example, we use an FDG input function from the module-provided data, and simulate a noisy 1TCM
        TAC and fit it -- showing a plot of everything at the end.
    
        .. plot::
            :include-source:
            :caption: Fitting a noisy simulated 1TCM TAC.
            
            import numpy as np
            import petpal.kinetic_modeling.tcms_as_convolutions as pet_tcm
            import petpal.kinetic_modeling.tac_fitting as pet_fit
            import matplotlib.pyplot as plt
            import petpal.utils.testing_utils as pet_tst
            import petpal.visualizations.tac_plots as tac_plots
            
            tcm_func = pet_tcm.gen_tac_1tcm_cpet_from_tac
            pTAC = np.asarray(np.loadtxt('../../../../../data/tcm_tacs/fdg_plasma_clamp_evenly_resampled.txt').T)
            tTAC = tcm_func(*pTAC, k1=1.0, k2=0.25, vb=0.05)
            tTAC[1] = pet_tst.add_gaussian_noise_to_tac_based_on_max(tTAC[1])
            
            fitter = pet_fit.TACFitter(pTAC=pTAC, tTAC=tTAC, tcm_func=tcm_func)
            fitter.run_fit()
            fit_params = fitter.fit_results[0]
            fit_tac = pet_tcm.gen_tac_1tcm_cpet_from_tac(*pTAC, *fit_params)
            
            plotter = tac_plots.TacFigure()
            plotter.add_tac(*pTAC, label='Input TAC', color='black', ls='--')
            plotter.add_tac(*tTAC, label='Tissue TAC', color='blue', ls='', marker='o', mec='k')
            plotter.add_tac(*fit_tac, label='Fit TAC', color='red', ls='-', marker='', lw=2.5)
            plt.legend()
            plt.show()
    
    See Also:
        * :class:`TACFitterWithoutBloodVolume` to assume :math:`V_B=0` and only fit the kinetic parameters.
        
    """

    SUPPORTED_MODELS = frozenset(_CONV_TCM_MODELS_CONFIGS.keys())

    def __init__(self,
                 pTAC: np.ndarray,
                 tTAC: np.ndarray,
                 weights: Union[None, float, np.ndarray] = None,
                 tcm_func: Callable = None,
                 fit_bounds: Union[np.ndarray, None] = None,
                 resample_num: int = 512,
                 aif_fit_thresh_in_mins: float = 30.0,
                 max_iters: int = 2500):
        r"""
        Initialize TACFitter with provided arguments.

        The init function performs several important operations:
            1. It sets the maximum number of function evaluations (iterations) for the optimization process.
            2. It sets the TCM function properties and initial bounds with the provided TCM function and fit bounds.
            3. It loads the raw tissue and plasma TACs and then resamples them evenly over a regular time grid.
            4. It determines the weights to be used for handling residuals during the optimization process.
            5. It sets the plasma TAC values and tissue TAC values to fit the TCM model.

        Args:
            pTAC (np.ndarray): The plasma TAC, with the form ``[times, values]``.
            tTAC (np.ndarray): The tissue TAC to which we will fit a TCM, with the form ``[times, values]``.
            weights (float, np.ndarray or None, optional): Weights for handling residuals during the optimization
                process. If None, all residuals are equally weighted. Defaults to None.
            tcm_func (Callable, optional): The specific TCM function to be used for fitting. Defaults to None.
            fit_bounds (np.ndarray or None, optional): Bounds for each parameter for curve fitting.
                If None, they will be guessed. Defaults to None.
            resample_num (int, optional): The number of time points used when resampling TAC data. Defaults to 512.
            aif_fit_thresh_in_mins (float, optional): The threshold in minutes when resampling. Defaults to 30.0.
            max_iters (int, optional): Maximum number of function evaluations (iterations) for the optimization process.
                Defaults to 2500.
                
        """

        self._validate_inputs(input_tac=pTAC, roi_tac=tTAC, tcm_func=tcm_func)

        self.max_func_evals: int = max_iters
        self.model_config = _CONV_TCM_MODELS_CONFIGS[tcm_func]
        self.tcm_func: Callable | None = tcm_func
        self.fit_param_number: int | None = self.model_config.num_params
        self.fit_param_names: list[str] | None = self.model_config.param_names

        self.bounds = self._setup_bounds(fit_bounds=fit_bounds)
        self.initial_guesses = self.bounds[:, 0]
        self.bounds_lo = self.bounds[:, 1]
        self.bounds_hi = self.bounds[:, 2]

        self.raw_p_tac: np.ndarray = pTAC.copy()
        self.raw_t_tac: np.ndarray = tTAC.copy()
        self.sanitized_t_tac: np.ndarray | None = None
        self.sanitized_p_tac: np.ndarray | None = None
        self.resample_times: np.ndarray | None = None
        self.delta_t: float | None = None
        self.resampled_t_tac: np.ndarray | None = None
        self.resampled_p_tac: np.ndarray | None = None

        self.resample_tacs_evenly(aif_fit_thresh_in_mins, resample_num)

        self.weights: np.ndarray | None = None
        self.set_weights(weights)

        self.p_tac_vals: np.ndarray | None = self.resampled_p_tac[1]
        self.tgt_tac_vals: np.ndarray | None = self.resampled_t_tac[1]
        self.fit_results = None

    def _validate_inputs(self, input_tac: np.ndarray, roi_tac: np.ndarray, tcm_func: Callable):
        assert np.asarray(input_tac).ndim == 2, "Input TAC must be a 2D array of times and activity"
        assert np.asarray(roi_tac).ndim == 2, "Input TAC must be a 2D array of times and activity"

        if tcm_func not in self.SUPPORTED_MODELS:
            raise ValueError(
                    f"tcm_model_func must be one of: "
                    f"{', '.join(f.__name__ for f in self.SUPPORTED_MODELS)}"
                    )

    def _setup_bounds(self, fit_bounds: np.ndarray | None) -> np.ndarray:
        expected_shape = (self.model_config.num_params, 3)

        if fit_bounds is not None:
            if fit_bounds.shape != expected_shape:
                raise ValueError(
                        f"fit_bounds has wrong shape {fit_bounds.shape}. "
                        f"Expected {expected_shape} for parameters: "
                        f"{', '.join(self.model_config.param_names)}"
                        )
            return fit_bounds.copy()

        return self.model_config.default_bounds.copy()

    def resample_tacs_evenly(self, fit_thresh_in_mins: float, resample_num: int) -> None:
        r"""
        Resample pTAC and tTAC evenly with respect to time, and at the same times.

        The method takes a threshold in minutes and a resample number as inputs. It starts by sanitizing
        the pTAC and tTAC (prepending a :math:`f(t=0)=0` point to data if necessary). A regularly sampled time is
        then generated using the start, end, and number of samples dictated by resample_num. Following this,
        an interpolation object is created using the :class:`petpal.blood_input.BloodInputFunction` class for the pTAC.
        This allows both interpolation and extrapolation for times beyond the pTAC onto the new tTAC times.

        Finally, the method resamples the sanitized tTAC and pTAC across these new evenly distributed
        times to ensure that they are regularly spaced over time. These resampled values are stored for
        future computations. The :math:`\Delta t` for the regularly sampled times is also stored.

        Args:
            fit_thresh_in_mins (float): Threshold in minutes used for defining how to fit half of the pTAC.
                                        The fitting time threshold determines the point at which the pTAC
                                        switches from interpolation to fitting. It should be a positive float value.

            resample_num (int): Number of samples to generate when resampling the tTAC. This will be the total
                                number of samples in tTAC after it has been resampled. It should be a positive integer.

        Returns:
            None

        Side Effects:
            - sanitized_t_tac (np.ndarray): Sanitized version of the original tTAC given during class initialization.
            - sanitized_p_tac (np.ndarray): Sanitized version of the original pTAC given during class initialization.
            - resample_times (np.ndarray): Regularly sampled time points generated from the start and end of sanitized
              tTAC, and the passed resample_num.
            - delta_t (float): Delta between the newly created time steps in resample_times.
            - resampled_t_tac (np.ndarray): tTAC resampled at the time points defined in resample_times.
            - resampled_p_tac (np.ndarray): pTAC resampled and extrapolated (if necessary) at the time points defined in
              resample_times.
              
        See Also:
            - :class:`petpal.blood_input.BloodInputFunction`
            
        """
        self.sanitized_t_tac = self.sanitize_tac(*self.raw_t_tac)
        self.sanitized_p_tac = self.sanitize_tac(*self.raw_p_tac)
        
        self.resample_times = np.linspace(self.sanitized_t_tac[0][0], self.sanitized_t_tac[0][-1], resample_num)
        self.delta_t = self.resample_times[1] - self.resample_times[0]
        
        p_tac_interp_obj = pet_bld.BloodInputFunction(time=self.sanitized_p_tac[0], activity=self.sanitized_p_tac[1],
                                                      thresh_in_mins=fit_thresh_in_mins)
        
        self.resampled_t_tac = self.resample_tac_on_new_times(*self.sanitized_t_tac, self.resample_times)
        self.resampled_p_tac = np.asarray(
                [self.resample_times[:], p_tac_interp_obj.calc_blood_input_function(t=self.resample_times)])
    
    def set_weights(self, weights: Union[float, str, None]) -> None:
        r"""
        Sets the weights for handling the residuals in the optimization process.

        The ``weights`` parameter determines how weights will be used:
            - It can be a float which will generate the weights based on an exponential decay formula. We assume that
              the passed in float is the decay constant, :math:`\lambda=\ln(2)/T_{1/2}`, where the half-life is in
              minutes. The weights are generated as: :math:`\sigma_i=\sqrt{e^{-\lambda t_i}C(t_i)}`, to be used as the
              ``sigma`` parameter for :func:`scipy.optimize.curve_fit`.
            - If it's a numpy array, the weights are linearly interpolated on the calculated `resample_times`.
            - If no specific value or an array is given, a numpy array of ones is used (i.e., it assumes equal weight).

        The method asserts that ``resampled_t_tac`` has been computed, thus :meth:`resample_tacs_evenly`
        method should be run before this.

        Args:
            weights (Union[float, str, None]): Determines how weights will be computed. If a float, it is used
                                               as the exponential decay constant. If a numpy array, the provided weights
                                               are linearly interpolated on the calculated resampled times. If None,
                                               equal weights are assumed.

        Returns:
            None

        Side Effects:
            weights (np.ndarray): Sets the weights attribute of the class based on logical conditions. Either
                                  they are based on an exponential decay function, directly supplied, or assumed
                                  as equal weights.
                                  
        """
        assert self.resampled_t_tac is not None, 'This method should be run after `resample_tacs_evenly`'
        
        if isinstance(weights, float):
            tmp_ar = np.sqrt(np.exp(-weights * self.resampled_t_tac[0]) * self.resampled_t_tac[1])
            zero_idx = tmp_ar == 0.0
            tmp_ar[zero_idx] = np.inf
            self.weights = tmp_ar
        elif isinstance(weights, np.ndarray):
            self.weights = np.interp(x=self.resampled_t_tac[0], xp=self.raw_t_tac[0], fp=weights)
        else:
            self.weights = np.ones_like(self.resampled_t_tac[1])

    @staticmethod
    def sanitize_tac(tac_times_in_minutes: np.ndarray, tac_vals: np.ndarray) -> np.ndarray:
        r"""
        Makes sure that the Time-Activity Curve (TAC) starts from time zero.

        The method ensures that the TAC starts from time zero by checking the first timestamp. If it's not zero, a zero
        timestamp and value are prepended, otherwise, the first value is set to zero. This method assumes that
        `tac_times_in_minutes` and `tac_vals` arrays have the same shape.

        Args:
            tac_times_in_minutes (numpy.ndarray): The original times of the TAC.
            tac_vals (numpy.ndarray): The original values of the TAC.

        Returns:
            numpy.ndarray: The sanitized TAC: ``[sanitized_times, sanitized_vals]``.
        """
        assert tac_times_in_minutes.shape == tac_vals.shape, ("`tac_times_in_minutes` and "
                                                              "`tac_vals` must have the same shape.")
        if tac_times_in_minutes[0] != 0.0:
            return np.asarray([np.append(0, tac_times_in_minutes), np.append(0, tac_vals)])
        else:
            out_vals = tac_vals[:]
            out_vals[0] = 0.0
            return np.asarray([tac_times_in_minutes, out_vals])
    
    @staticmethod
    def resample_tac_on_new_times(tac_times_in_minutes: np.ndarray, tac_vals: np.ndarray, new_times: np.ndarray) -> np.ndarray:
        r"""
        Resamples the Time-Activity Curve (TAC) on given new time points by linear interpolation.

        The method performs a linear interpolation of `tac_vals` on `new_times` based on `tac_times_in_minutes`.

        Args:
            tac_times_in_minutes (numpy.ndarray): The original times of the TAC.
            tac_vals (numpy.ndarray): The original values of the TAC.
            new_times (numpy.ndarray): The new times to resample the TAC on.

        Returns:
            numpy.ndarray: The resampled TAC: the resampled times and values. ``[new_times, new_vals]``.
            
        See Also:
            :func:`numpy.interp`
            
        """
        return np.asarray([new_times, np.interp(x=new_times, xp=tac_times_in_minutes, fp=tac_vals)])
    
    def fitting_func(self, x: np.ndarray, *params) -> np.ndarray:
        r"""
        A wrapper function to fit the Tissue Compartment Model (TCM) using given parameters.

        It calculates the results of the TCM function with the given times and parameters using the resampled pTAC.

        Args:
            x (np.ndarray): The independent data (time-points for TAC)
            *params: The parameters for the TCM function

        Returns:
            np.ndarray: The values of the TCM function with the given parameters at the given x-values.
        """
        return self.tcm_func(x, self.p_tac_vals, *params)[1]
    
    def run_fit(self) -> None:
        r"""
        Runs the optimization/fitting process on the data, using previously defined function and parameters.

        This method runs the curve fitting process on the TAC data, starting with the initial guesses
        for the parameters and the preset bounds for each. ``fitting_func``, initial guesses and bounds
        should have been set prior to calling this method. Optimized fit results and fit covariances are stored in
        ``fit_results``.

        Returns:
            None

        Side Effects:
            - fit_results (OptimizeResult): The results of the fit, including optimized parameters and covariance matrix.
              Fitted values can be extracted using fit_results.x, among other available attributes (refer to
              :func:`scipy.optimize.curve_fit` documentation for more details).
              
        """
        self.fit_results = sp_cv_fit(f=self.fitting_func, xdata=self.resample_times, ydata=self.tgt_tac_vals,
                                     p0=self.initial_guesses, bounds=(self.bounds_lo, self.bounds_hi),
                                     sigma=self.weights, maxfev=self.max_func_evals)


class TACFitterWithoutBloodVolume(TACFitter):
    r"""

    .. warning::
        This class is now deprecated and will be removed in a future version. Currently just acts like
        :class:`~.TACFitter`, and will attempt to fit for blood volume.


    A sub-class of TACFitter used specifically for fitting Tissue Compartment Models(TCM) to Time Activity Curves (TAC),
    when there is no signal contribution from blood volume in the TAC.

    It uses the functionalities of :class:`TACFitter` and modifies the methods calculating the ``tcm_function``
    properties and the bounds setting, and the wrapped ``fitting_func`` to ignore the blood volume parameter, ``vb``.

    Attributes:
        resample_times (np.ndarray): Times at which TACs are resampled.
        resampled_t_tac (np.ndarray): Tissue TAC values resampled at these times.
        p_tac_vals (np.ndarray): Plasma TAC values used for feeding to TCM function.
        raw_t_tac (np.ndarray): Raw TAC times for tissue, fed at initialization.
        weights (np.ndarray): Weights for handling residuals during the optimization process.
        tgt_tac_vals (np.ndarray): Tissue TAC values to fit TCM model.
        fit_param_number (int): Number of fitting parameters in the TCM function.
        initial_guesses (np.ndarray): Initial guesses for all the parameters for curve fitting.
        bounds_hi (np.ndarray): Upper bounds for all the parameters for curve fitting.
        fit_results (np.optimize.OptimizeResult): The results of the fit, including optimized parameters and covariance
            matrix.
        fit_param_names (List[str]): Names of fitting parameters in the TCM function.
        raw_p_tac (np.ndarray): Raw TAC times for plasma, fed at initialization.
        resampled_p_tac (np.ndarray): Plasma TAC values resampled on these times.
        sanitized_t_tac (np.ndarray): Sanitized version of tissue TAC times.
        bounds_lo (np.ndarray): Lower bounds for all the parameters for curve fitting.
        bounds (np.ndarray): Bounds for each parameter for curve fitting.
        max_func_evals (int): Maximum number of function evaluations (iterations) for the optimization process.
        tcm_func (Callable): The tissue compartment model (TCM) function to fit.
        sanitized_p_tac (np.ndarray): Sanitized version of plasma TAC times.
        delta_t (float): Delta between the newly created time steps in resampled times.
        
    See Also:
        * :class:`~.TACFitter`

    """
    def __init__(self,
                 pTAC: np.ndarray,
                 tTAC: np.ndarray,
                 weights: np.ndarray = None,
                 tcm_func: Callable = None,
                 fit_bounds: np.ndarray = None,
                 resample_num: int = 2048,
                 aif_fit_thresh_in_mins: float = 30.0,
                 max_iters: int = 2500):
        r"""
        Initializes TACFitterWithoutBloodVolume with provided arguments. Inherits all arguments from parent class TACFitter.

        This ``__init__`` method, in comparison to TACFitter's ``__init__``, executes the same initial operations but
        disregards the blood volume parameter. The significant steps are:
        
            1. Calls the TACFitter's __init__ with the provided arguments.
            2. Sets the TCM function properties while eliminating blood volume.
            3. Sets the fitting bounds and initial guesses, again excluding blood volume.

        Args:
            pTAC (np.ndarray): The plasma TAC, with the form [times, values].
            tTAC (np.ndarray): The tissue TAC to which we will fit a TCM, with the form [times, values].
            weights (float, np.ndarray or None, optional): Weights for handling residuals during the optimization process.
                If None, all residuals are equally weighted. Defaults to None.
            tcm_func (Callable, optional): The specific TCM function to be used for fitting. Defaults to None.
            fit_bounds (np.ndarray or None, optional): Bounds for each parameter for curve fitting.
                If None, they will be guessed. Defaults to None.
            resample_num (int, optional): The number of time points used when resampling TAC data. Defaults to 512.
            aif_fit_thresh_in_mins (float, optional): The threshold in minutes when resampling. Defaults to 30.0.
            max_iters (int, optional): Maximum number of function evaluations (iterations) for the optimization process.
                Defaults to 2500.

        Side Effect:
            Sets the TCM function properties and initial bounds while disregarding the blood volume parameter.
            
        See Also
            * :class:`TACFitter`
            
        """
        
        super().__init__(pTAC, tTAC, weights, tcm_func, fit_bounds, resample_num, aif_fit_thresh_in_mins, max_iters)
        warnings.warn("TACFitterWithoutBloodVolume is deprecated and will be removed in"
                      "a future update. Please use TACFitter instead. This class behaves just like"
                      "TACFitter currently.",
                      DeprecationWarning, stacklevel=2)

class TCMAnalysis(object):
    r"""
    A class dedicated to perform Tissue Compartment Model (TCM) fitting to time-activity curves (TACs).

    This class consolidates all related TAC fitting functionalities. It first initializes relevant fitting parameters
    and paths to TAC data from input arguments. After initialization, an analysis with methods like `run_analysis()`
    and `save_analysis()` can be performed. The results, including fit parameters and properties, can be
    accessed after the analysis.

    Example:
        In the proceeding example, we assume that we have two tacs: an input function tac, and a region of interest
        (ROI) tac named 'input_tac.txt' and 'roi_tac.txt', respectively. Here, we are trying to fit the ROI tac to
        a standard serial 2TCM. Note that we are not fitting any time delay or dispersion corrections.
        
        .. code-block:: python
            
            import petpal.kinetic_modeling.tac_fitting as pet_fit
            
            fit_obj = pet_fit.TCMAnalysis(input_tac_path='./input_tac.txt',
                                          roi_tac_path='./roi_tac.txt',
                                          output_directory='./',
                                          output_filename_prefix='fit',
                                          compartment_model='serial-2tcm',
                                          weights=None,
                                          parameter_bounds=np.asarray([[0.1, 0.0, 1.0],
                                                                       [0.1, 0.0, 1.0],
                                                                       [0.1, 0.0, 1.0],
                                                                       [0.1, 0.0, 1.0],
                                                                       [0.1, 0.0, 1.0]]),
                                          resample_num=512
                              )
            fit_obj.run_analysis()
            fit_obj.save_analysis()
    
    See Also:
        * :class:`TACFitter`
        * :class:`TACFitterWithoutBloodVolume`
    
    """
    def __init__(self,
                 input_tac_path: str,
                 roi_tac_path: str,
                 output_directory: str,
                 output_filename_prefix: str,
                 compartment_model: str,
                 parameter_bounds: Union[None, np.ndarray] = None,
                 weights: Union[float, None, np.ndarray] = None,
                 resample_num: int = 512,
                 aif_fit_thresh_in_mins: float = 40.0,
                 max_func_iters: int = 2500,
                 ignore_blood_volume: bool = False):
        r"""
        Initializes an instance of the TCMAnalysis class.

        The initialization follows these steps:
            1. Saves absolute paths of the TAC files and output directory.
            2. Validates and stores the compartment model and the corresponding fitting function.
            3. Stores the other fitting parameters passed as arguments.
            4. Initializes the appropriate fitting object, depending on whether blood volume is considered or not.
            5. Initializes the analysis properties structure.

        After initialization, you can directly run and save the analysis using bundled methods :meth:`run_analysis` and
        :meth:`save_analysis`, respectively.
        
        See Also:
            * :meth:`validated_tcm`
            
        """
        self.input_tac_path: str = os.path.abspath(input_tac_path)
        self.roi_tac_path: str = os.path.abspath(roi_tac_path)
        self.output_directory: str = os.path.abspath(output_directory)
        self.output_filename_prefix: str = output_filename_prefix
        self.compartment_model: str = self.validated_tcm(compartment_model)
        self.short_tcm_name = "".join(self.compartment_model.split("-"))
        self._tcm_func: Callable = self._get_tcm_function(self.compartment_model)
        self.bounds: Union[None, np.ndarray] = parameter_bounds
        self.tac_resample_num: int = resample_num
        self.input_tac_fitting_thresh_in_mins: float = aif_fit_thresh_in_mins
        self.max_func_iters: int = max_func_iters
        self.ignore_blood_volume = ignore_blood_volume
        self.weights: Union[float, None, np.ndarray] = weights
        if self.ignore_blood_volume:
            self.fitter_class = TACFitterWithoutBloodVolume
        else:
            self.fitter_class = TACFitter
        self.analysis_props: dict = self.init_analysis_props()
        self.fit_results: Union[None, tuple[np.ndarray, np.ndarray]] = None
        self._has_analysis_been_run: bool = False

    def init_analysis_props(self):
        r"""
        Initialize the structure of the analysis properties dictionary. This dictionary can be saved as a JSON file for
        parsing the results.
    
        The dictionary structure is as follows:
            - FilePathPTAC -> path of the input TAC file
            - FilePathTTAC -> path of the ROI TAC file
            - TissueCompartmentModel -> type of tissue compartment model used
            - IgnoreBloodVolume -> flag indicating whether blood volume is being considered or not
            - PTACFittingThresholdTime -> threshold time for the AIF fitting
            - FitProperties -> an inner dictionary with empty lists/arrays as placeholders
              for FitValues, FitStdErr and Bounds. Also contains ResampleNum and MaxIterations.
    
        FitProperties will be updated during the analysis process.
    
        Returns:
            dict: Initialized dictionary with analysis properties structure.
            
        See Also:
            * :meth:`calculate_fit_properties`
            
        """
        props = {
            'FilePathPTAC': self.input_tac_path,
            'FilePathTTAC': self.roi_tac_path,
            'TissueCompartmentModel': self.compartment_model,
            'IgnoreBloodVolume': self.ignore_blood_volume,
            'PTACFittingThresholdTime': self.input_tac_fitting_thresh_in_mins,
            'FitProperties': {
                'FitValues': [],
                'FitStdErr': [],
                'Bounds': [],
                'ResampleNum': self.tac_resample_num,
                'MaxIterations': self.max_func_iters,
                }
            }

        return props

    @staticmethod
    def validated_tcm(compartment_model: str) -> str:
        r"""
        Validates the type of tissue compartment model.

        Runs :meth:`~.ConvTcmModel.resolve_model_name` for validation, and returns
        the normalized string by running :meth:`~.ConvTcmModel.normalize_name`.

        Args:
            compartment_model (str): The name of the compartment model.

        Returns:
            str: The transformed name of the validated compartment model if it was valid.

        Raises:
            KeyError: If the provided compartment model is not one of '1tcm', '2tcm-k4zero', 'serial-2tcm' or '2tcm'.
            
        """
        ConvTcmModelConfig.resolve_model_name(compartment_model)
        return ConvTcmModelConfig.normalize_name(compartment_model)

    @staticmethod
    def _get_tcm_function(compartment_model: str) -> Callable:
        """
        Returns the corresponding function for the provided tissue compartment model used for the fitting class.
        Runs :meth:`~.ConvTcmModel.resolve_model_name` for validation, and returns the appropriate function.

        Args:
            compartment_model: The name of the tissue compartment model.

        Returns:
            function: The corresponding function for the tissue compartment model.

        Raises:
            KeyError: If the provided tissue compartment model name does not correspond to any known models.
        
        See Also:
            * :func:`gen_tac_1tcm_cpet_from_tac<petpal.tcms_as_convolutions.gen_tac_1tcm_cpet_from_tac>`
            * :func:`gen_tac_2tcm_with_k4zero_cpet_from_tac<petpal.tcms_as_convolutions.gen_tac_2tcm_with_k4zero_cpet_from_tac>`
            * :func:`gen_tac_2tcm_cpet_from_tac<petpal.tcms_as_convolutions.gen_tac_2tcm_cpet_from_tac>`
            * :class:`~.TACFitter`

        """
        return ConvTcmModelConfig.resolve_model_name(compartment_model)

    def run_analysis(self):
        r"""
        Runs the fitting analysis given the file-paths and method.
        
        :meth:`calculate_fit` and :meth:`calculate_fit_properties` are run, and the analysis-has-been-fun flag is set to
        true.

        This method first calculates the fit, then updates the analysis properties with the
        fit results and finally sets the flag denoting that analysis has been successfully run.
        
        
        """
        self.calculate_fit()
        self.calculate_fit_properties()
        self._has_analysis_been_run = True

    def save_analysis(self):
        r"""
        Saves the analysis properties to a json file in the prescribed output directory.

        The saved filename is constructed as a combination of the output directory, filename prefix,
        "analysis", and the used TissueCompartmentModel.
        If the analysis has not been run before calling this method, a RuntimeError is raised.

        Raises:
            RuntimeError: If the method is called before the analysis has been run.
        """
        if not self._has_analysis_been_run:
            raise RuntimeError("'run_analysis' method must be run before running this method.")

        file_name_prefix = os.path.join(self.output_directory,
                                        f"{self.output_filename_prefix}_desc"
                                        f"-{self.short_tcm_name}")
        analysis_props_file = f"{file_name_prefix}_fitprops.json"
        with open(analysis_props_file, 'w', encoding='utf-8') as f:
            json.dump(obj=self.analysis_props, fp=f, indent=4)

    def update_props_with_formatted_fit_values(self, fit_results, fit_props_dict: dict):
        r"""
        Update the analysis properties dictionary with formatted fit results.

        Extracts fit parameters and standard errors from fit results, formats them using
        pretty parameter names, and updates the provided properties dictionary.

        Args:
            fit_results (tuple): Tuple containing (fit_params, fit_covariances) from the fitting process.
            fit_props_dict (dict): Dictionary to update with formatted fit values and errors.

        Side Effects:
            Updates the FitProperties section of fit_props_dict with formatted FitValues, FitStdErr, and Bounds.
        """
        fit_params, fit_covariances = fit_results
        try:
            fit_stderr = np.sqrt(np.diagonal(fit_covariances))
        except ValueError:
            fit_stderr = np.nan * np.ones_like(fit_params)
        format_func = self._generate_pretty_params
        
        fit_props_dict["FitProperties"]["FitValues"] = format_func(fit_params.round(5))
        fit_props_dict["FitProperties"]["FitStdErr"] = format_func(fit_stderr.round(5))
        
        format_func = self._generate_pretty_bounds
        fit_props_dict["FitProperties"]["Bounds"] = format_func(self.bounds.round(5))

    def calculate_fit_properties(self):
        r"""
        Calculates the fit properties and updates the analysis properties.

        This method retrieves the fitting parameters and their standard errors from the fitting
        results, formats them for readability, and stores them in the analysis properties dictionary.
        Bounds to the fitting parameters are also formatted and stored.
        """
        self.update_props_with_formatted_fit_values(fit_results=self.fit_results,
                                                    fit_props_dict=self.analysis_props)
    
    def calculate_fit(self):
        r"""
        Performs the fit and stores results to the instance.

        This method has a series of actions which it performs in the following order:
            1. Loads TAC files using the :func:`safe_load_tac` method, specific to the paths stored in instance
               variables.
            2. Creates a fitting object with the relevant parameters and TACs.
            3. Runs the fit using the fitting object's ``run_fit`` method.
            4. Stores the results of the fit in the ``fit_results`` instance variable.

        As a result of this method, ``fit_results`` instance variable will hold the results of the fit, that
        can be used for further analysis or calculations.
        
        See Also:
            * :class:`TACFitter`
            * :class:`TACFitterWithoutBloodVolume`
        
        """
        p_tac = safe_load_tac(self.input_tac_path)
        t_tac = safe_load_tac(self.roi_tac_path)
        self.fitter_class = self.fitter_class(pTAC=p_tac, tTAC=t_tac,
                                              weights=self.weights,
                                              tcm_func=self._tcm_func,
                                              fit_bounds=self.bounds,
                                              max_iters=self.max_func_iters,
                                              aif_fit_thresh_in_mins=self.input_tac_fitting_thresh_in_mins,
                                              resample_num=self.tac_resample_num)
        self.fitter_class.run_fit()
        self.fit_results = self.fitter_class.fit_results
        if self.bounds is None:
            self.bounds = self.fitter_class.bounds

    def _generate_pretty_params(self, results: np.ndarray) -> dict:
        r"""
        Transforms array of results into a formatted dictionary.

        This method formats the fitting results into a more human-readable form.
        If the fitting was done without blood volume, it formats all values as 'k_i': value.
        Otherwise it formats all but the last as 'k_i': value and the last as 'vb': value.

        Args:
            results (np.ndarray): The array of fitting results.

        Returns:
            dict: The formatted fitting results as {param: value} pairs. In the case of
                  TACFitterWithBloodVolume, the last parameter will be named 'vb', others 'k_i'.
                  In the case of TACFitterWithoutBloodVolume, parameters will be named 'k_i'.
        """
        k_vals = {f'k_{n + 1}': val for n, val in enumerate(results[:-1])}
        vb = {'vb': results[-1]}
        return {**k_vals, **vb}

    def _generate_pretty_bounds(self, bounds: Union[np.ndarray, None]) -> dict:
        r"""
        Transforms array of bounds into a formatted dictionary.

        This method creates a dictionary of the fitting parameters and their corresponding
        initial values and lower and upper bounds. The keys of the dictionary are the names
        of the fitting parameters from :meth:`_generate_pretty_params` method.

        Args:
            bounds (np.ndarray): The array of parameter bounds.

        Returns:
            dict: The fitting parameters with their corresponding initial values, and lower and
                  upper bounds, in the following format: {param: {'initial': val, 'lo': lo, 'hi': hi}}
        """
        param_names = list(self._generate_pretty_params(bounds).keys())
        param_bounds = {f'{param}': {'initial': val[0],
                                     'lo': val[1],
                                     'hi': val[2]} for param, val in
                        zip(param_names, bounds)}
        return param_bounds
    
    def __call__(self):
        self.run_analysis()
        self.save_analysis()


class MultiTACTCMAnalysis(TCMAnalysis, MultiTACAnalysisMixin):
    """
    A class for performing tissue compartment model (TCM) analysis on multiple tissue TACs.


    Attributes:
        input_tac_path (str): Path to the input TAC file.
        roi_tacs_dir (str): Directory containing region of interest TAC files.
        output_directory (str): Directory for saving analysis results.
        output_filename_prefix (str): Prefix for output filenames.
        compartment_model (str): Name of the compartment model to use.
        parameter_bounds (Union[None, np.ndarray], optional): Bounds for the fitting parameters. Defaults to None.
        weights (Union[float, None, np.ndarray], optional): Weights for fitting. Defaults to None.
        resample_num (int): Number of resampling points. Defaults to 512.
        aif_fit_thresh_in_mins (float): Threshold in minutes for AIF fitting. Defaults to 40.0.
        max_func_iters (int): Maximum number of iterations for the fitting function. Defaults to 2500.
        ignore_blood_volume (bool): Whether to ignore blood volume in the analysis. Defaults to False.
    """
    def __init__(self,
                 input_tac_path: str,
                 roi_tacs_dir: str,
                 output_directory: str,
                 output_filename_prefix: str,
                 compartment_model: str,
                 parameter_bounds: Union[None, np.ndarray] = None,
                 weights: Union[float, None, np.ndarray] = None,
                 resample_num: int = 512,
                 aif_fit_thresh_in_mins: float = 40.0,
                 max_func_iters: int = 2500,
                 ignore_blood_volume: bool = False):
        """
        Initializes the MultiTACTCMAnalysis object with required paths, model parameters, and fitting options.

        Args:
            input_tac_path (str): Path to the input TAC file.
            roi_tacs_dir (str): Directory containing region of interest TAC files.
            output_directory (str): Directory for saving analysis results.
            output_filename_prefix (str): Prefix for output filenames.
            compartment_model (str): Name of the compartment model to use.
            parameter_bounds (Union[None, np.ndarray], optional): Bounds for the fitting parameters. Defaults to None.
            weights (Union[float, None, np.ndarray], optional): Weights for fitting. Defaults to None.
            resample_num (int, optional): Number of resampling points. Defaults to 512.
            aif_fit_thresh_in_mins (float, optional): Threshold in minutes for AIF fitting. Defaults to 40.0.
            max_func_iters (int, optional): Maximum number of iterations for the fitting function. Defaults to 2500.
            ignore_blood_volume (bool, optional): Whether to ignore blood volume in the analysis. Defaults to False.
        """
        MultiTACAnalysisMixin.__init__(self,
                                       input_tac_path=input_tac_path,
                                       tacs_dir=roi_tacs_dir)
        TCMAnalysis.__init__(self,
                             input_tac_path=input_tac_path,
                             roi_tac_path=roi_tacs_dir,
                             output_directory=output_directory,
                             output_filename_prefix=output_filename_prefix,
                             compartment_model=compartment_model,
                             parameter_bounds=parameter_bounds,
                             weights=weights,
                             resample_num=resample_num,
                             aif_fit_thresh_in_mins=aif_fit_thresh_in_mins,
                             max_func_iters=max_func_iters,
                             ignore_blood_volume=ignore_blood_volume)
        self.fit_results = []
        
    def init_analysis_props(self):
        """
        Initializes analysis properties for each tissue TAC. Overrides :meth:`TCMAnalysis.init_analysis_props`.

        Returns:
            list[dict]: A list of analysis property dictionaries for each TAC.
        """
        num_of_tacs = self.num_of_tacs
        analysis_props = [TCMAnalysis.init_analysis_props(self) for a_tac in range(num_of_tacs)]
        for tac_id, a_prop_dict in enumerate(analysis_props):
            a_prop_dict['FilePathTTAC'] = os.path.abspath(self.tacs_files_list[tac_id])
        return analysis_props

    def calculate_fit(self):
        """
        Calculates the fit for each TAC, updating the analysis properties with model fit results.
        Overrides :meth:`TCMAnalysis.calculate_fit`.
        """
        p_tac = safe_load_tac(self.input_tac_path)
        fit_obj = None
        for a_tac in self.tacs_files_list:
            t_tac = safe_load_tac(a_tac)
            fit_obj = self.fitter_class(pTAC=p_tac,
                                        tTAC=t_tac,
                                        weights=self.weights,
                                        tcm_func=self._tcm_func,
                                        fit_bounds=self.bounds,
                                        max_iters=self.max_func_iters,
                                        aif_fit_thresh_in_mins=self.input_tac_fitting_thresh_in_mins,
                                        resample_num=self.tac_resample_num)
            fit_obj.run_fit()
            self.fit_results.append(fit_obj.fit_results)
        if (self.bounds is None) and (fit_obj is not None):
            self.bounds = fit_obj.bounds

    def calculate_fit_properties(self):
        """
        Updates analysis properties with formatted fit values for each TAC.
        Overrides :meth:`TCMAnalysis.calculate_fit_properties`.
        """
        for fit_results, fit_props, tac_path in zip(self.fit_results,
                                                    self.analysis_props,
                                                    self.tacs_files_list):
            self.update_props_with_formatted_fit_values(fit_results=fit_results, fit_props_dict=fit_props)

    def save_analysis(self):
        """
        Saves the analysis results to a JSON file for each TAC. Overrides :meth:`TCMAnalysis.save_analysis`.

        Raises:
            RuntimeError: If 'run_analysis' method has not been run before 'save_analysis'.
        """
        if not self._has_analysis_been_run:
            raise RuntimeError("'run_analysis' method must be run before running this method.")
        
        for seg_name, fit_props in zip(self.inferred_seg_labels, self.analysis_props):
            
            filename = [self.output_filename_prefix,
                        f'desc-{self.short_tcm_name}',
                        f'seg-{seg_name}',
                        'fitprops.json']
            filename='_'.join(filename)
            filepath = os.path.join(self.output_directory, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(obj=fit_props, fp=f, indent=4)


class FrameAvgdTcmModelConfig(TcmModelConfig):
    r"""
    Configuration for frame-averaged TCM models.

    This subclass of :class:`~.TcmModelConfig` provides configurations specific to TCM models
    that operate on frame-averaged data. These models internally use convolution-based models
    as in :class:`~.ConvTcmModelConfig`, but expose a different API optimized for frame-averaged fitting.

    Attributes:
        _NAME_TO_FUNC (dict[str, Callable]): Mapping of normalized model names to their
            corresponding frame-averaged TCM functions.

    Note:
        These models internally use convolution-based models as in :class:`~.ConvTcmModelConfig`,
        but expose a different API for frame-averaged fitting.

    See Also:
        * :class:`~.TcmModelConfig`
        * :class:`~.ConvTcmModelConfig`
        * :mod:`petpal.kinetic_modeling.tcms_as_convolutions`
    """
    _NAME_TO_FUNC: dict[str, Callable] = {
        '1tcm'       : pet_tcms.model_serial_1tcm_frame_avgd,
        'serial-2tcm': pet_tcms.model_serial_2tcm_frame_avgd
        }


_FRAME_AVGD_TCM_CONFIGS = {
    pet_tcms.model_serial_1tcm_frame_avgd: FrameAvgdTcmModelConfig(
            func=pet_tcms.model_serial_1tcm_frame_avgd,
            param_names=['k1', 'k2', 'vb'],
            default_bounds=np.array([
                [0.2, 1e-8, 0.5],  # k1
                [0.1, 1e-8, 0.5],  # k2
                [0.05, 1e-8, 0.5]  # vb
                ]),
            pretty_param_names=[r'$K_1$', r'$k_2$', r'$V_B$'],
            ),
    pet_tcms.model_serial_2tcm_frame_avgd: FrameAvgdTcmModelConfig(
            func=pet_tcms.model_serial_2tcm_frame_avgd,
            param_names=['k1', 'k2', 'k3', 'k4', 'vb'],
            default_bounds=np.array([
                [0.2, 1e-8, 0.5],  # k1
                [0.1, 1e-8, 0.5],  # k2
                [0.1, 1e-8, 0.5],  # k3
                [0.01, 1e-8, 0.1], # k4
                [0.05, 1e-8, 0.5]  # vb
                ]),
            pretty_param_names=[r'$K_1$', r'$k_2$', r'$k_3$', r'$k_4$', r'$V_B$'],
            )
    }


class FrameAveragedTACFitter():
    r"""
    A class for fitting Tissue Compartment Models (TCM) to frame-averaged Time Activity Curves (TAC).

    This fitter is designed specifically for PET data where TAC values represent frame-averaged
    measurements rather than instantaneous samples. It uses the :mod:`lmfit` package for robust
    parameter estimation and handles frame timing information explicitly through :class:`~.ScanTimingInfo`.

    The class performs high-resolution resampling of input TACs internally, then averages the model
    predictions over each frame's duration to match the frame-averaged measurements.

    Attributes:
        input_tac (TimeActivityCurve): Input function (plasma) TAC.
        roi_tac (TimeActivityCurve): Region of interest (tissue) TAC to fit.
        roi_has_err (bool): Whether the ROI TAC has uncertainty estimates.
        frame_durations (np.ndarray): Duration of each frame in minutes.
        frame_starts (np.ndarray): Start time of each frame in minutes.
        frame_ends (np.ndarray): End time of each frame in minutes.
        model_config (FrameAvgdTcmModelConfig): Configuration for the TCM model being fitted.
        tcm_func (Callable): The tissue compartment model function.
        bounds (np.ndarray): Parameter bounds with shape (num_params, 3).
        initial_guesses (np.ndarray): Initial parameter guesses.
        bounds_lo (np.ndarray): Lower parameter bounds.
        bounds_hi (np.ndarray): Upper parameter bounds.
        tcm_fit_params (lmfit.Parameters): lmfit Parameters object for fitting.
        tac_resample_num (int): Number of points for high-resolution TAC resampling.
        fine_roi_tac (TimeActivityCurve): High-resolution resampled ROI TAC.
        fine_input_tac (TimeActivityCurve): High-resolution resampled input TAC.
        frame_idx_pairs (np.ndarray): Index pairs for frame averaging operations.
        weights (np.ndarray or None): Weights for weighted least squares fitting.
        result_obj (lmfit.minimizer.MinimizerResult or None): Results from lmfit optimization.
        fit_results (tuple[np.ndarray, np.ndarray] or None): Fitted parameters and covariance matrix.
        fit_residuals (np.ndarray or None): Residuals from the fit.
        fit_sum_of_square_residuals (float or None): Sum of squared residuals.
        fit_tac (TimeActivityCurve or None): Model-predicted TAC using fitted parameters.

    Example:
        .. code-block:: python

            from petpal.utils.time_activity_curve import TimeActivityCurve
            from petpal.utils.scan_timing import ScanTimingInfo
            import petpal.kinetic_modeling.tcms_as_convolutions as pet_tcm
            import petpal.kinetic_modeling.tac_fitting as pet_fit

            # Load TACs and scan timing
            input_tac = TimeActivityCurve.from_tsv('input.tsv')
            roi_tac = TimeActivityCurve.from_tsv('roi.tsv')
            scan_info = ScanTimingInfo.from_nifti('pet_image.nii')

            # Initialize and run fitter
            fitter = pet_fit.FrameAveragedTACFitter(
                input_tac=input_tac,
                roi_tac=roi_tac,
                scan_info=scan_info,
                tcm_model_func=pet_tcm.model_serial_2tcm_frame_avgd
            )
            fitter.run_fit()

            # Access results
            fit_params = fitter.fit_results[0]
            fit_tac = fitter.fit_tac

    See Also:
        * :class:`~.TACFitter` for non-frame-averaged fitting
        * :class:`~.FrameAveragedTCMAnalysis` for complete analysis workflow
        * :class:`~.TimeActivityCurve`
        * :class:`~.ScanTimingInfo`
    """

    SUPPORTED_MODELS = frozenset(_FRAME_AVGD_TCM_CONFIGS.keys())

    def __init__(self,
                 input_tac: TimeActivityCurve,
                 roi_tac: TimeActivityCurve,
                 scan_info: ScanTimingInfo,
                 tcm_model_func: Callable = pet_tcms.model_serial_2tcm_frame_avgd,
                 fit_bounds: None | np.ndarray = None,
                 fit_weights: None | np.ndarray | str = None,
                 tac_resample_num: int = 8192,
                 **leastsq_kwargs):
        r"""
        Initialize FrameAveragedTACFitter with TACs, scan timing, and fitting parameters.

        Args:
            input_tac (TimeActivityCurve): Input function (plasma) TAC.
            roi_tac (TimeActivityCurve): Region of interest (tissue) TAC to fit.
            scan_info (ScanTimingInfo): Scan timing information including frame starts and durations.
            tcm_model_func (Callable, optional): The frame-averaged TCM function to use for fitting.
                Defaults to :func:`model_serial_2tcm_frame_avgd<~.pet_tcms.model_serial_2tcm_frame_avgd>`.
            fit_bounds (np.ndarray or None, optional): Parameter bounds with shape (num_params, 3).
                If None, default bounds are used. Defaults to None.
            fit_weights (np.ndarray, str, or None, optional): Weights for fitting. Can be:
                None: No weighting (uniform weights);
                'roi' or 'roi-tac' or 'tac-err' or 'stderr': Use ROI TAC uncertainties;
                np.ndarray: Custom weight array Defaults to None.
            tac_resample_num (int, optional): Number of points for high-resolution resampling.
                Defaults to 8192.
            **leastsq_kwargs: Additional keyword arguments passed to :meth:`lmfit.Minimizer.leastsq`.

        Raises:
            AssertionError: If input_tac or roi_tac are not :class:`~.TimeActivityCurve` objects, or if
                scan_info is not a :class:`~.ScanTimingInfo` object.
            ValueError: If tcm_model_func is not in the supported models list.
        """
        self._validate_inputs(input_tac=input_tac, roi_tac=roi_tac,
                              scan_info=scan_info, tcm_model_func=tcm_model_func, )

        self.input_tac = TimeActivityCurve(*input_tac.tac_werr)
        self.roi_tac = TimeActivityCurve(*roi_tac.tac_werr)
        self.roi_has_err = not np.all(np.isnan(self.roi_tac.uncertainty))
        self.roi_tac.uncertainty[self.roi_tac.uncertainty == 0] = np.inf

        self.frame_durations = scan_info.duration_in_mins
        self.frame_starts = scan_info.start_in_mins
        self.frame_ends = scan_info.end_in_mins

        self.model_config = _FRAME_AVGD_TCM_CONFIGS[tcm_model_func]
        self.tcm_func = tcm_model_func

        self.bounds = self._setup_bounds(fit_bounds=fit_bounds)
        self.initial_guesses = self.bounds[:, 0]
        self.bounds_lo = self.bounds[:, 1]
        self.bounds_hi = self.bounds[:, 2]
        self.tcm_fit_params = self.gen_fit_params()

        self.tac_resample_num = tac_resample_num
        self.fine_roi_tac = self.roi_tac.evenly_resampled_tac(self.tac_resample_num)
        self.fine_input_tac = self.input_tac.resampled_tac_on_times(self.fine_roi_tac.times_in_mins)
        self.frame_idx_pairs = get_frame_index_pairs_from_fine_times(fine_times=self.fine_roi_tac.times_in_mins,
                                                                     frame_starts=self.frame_starts,
                                                                     frame_ends=self.frame_ends)
        self.weights = self._setup_weights(fit_weights=fit_weights)

        self._fit_obj = lmfit.Minimizer(userfcn=self.tcm_func,
                                        params=self.tcm_fit_params,
                                        fcn_args=(*self.fine_input_tac.tac,
                                                  self.frame_idx_pairs,
                                                  self.roi_tac.activity,
                                                  self.weights))
        self.leastsq_kwargs = leastsq_kwargs
        self.result_obj: None | lmfit.minimizer.MinimizerResult = None
        self.fit_results: None | tuple[np.ndarray, np.ndarray] = None
        self.fit_residuals: None | np.ndarray = None
        self.fit_sum_of_square_residuals: None | np.ndarray = None
        self.fit_tac: None | TimeActivityCurve = None


    def _validate_inputs(self,
                         input_tac: TimeActivityCurve,
                         roi_tac: TimeActivityCurve,
                         scan_info: ScanTimingInfo,
                         tcm_model_func: Callable):
        r"""
        Validate input parameters for the fitter.

        Args:
            input_tac (TimeActivityCurve): Input function TAC to validate.
            roi_tac (TimeActivityCurve): ROI TAC to validate.
            scan_info (ScanTimingInfo): Scan timing information to validate.
            tcm_model_func (Callable): TCM function to validate.

        Raises:
            AssertionError: If TACs are not :class:`~.TimeActivityCurve` objects or scan_info is not :class:`~.ScanTimingInfo`.
            ValueError: If tcm_model_func is not supported.
        """
        assert isinstance(input_tac, TimeActivityCurve), "Input TAC must be a TimeActivityCurve object."
        assert isinstance(roi_tac, TimeActivityCurve), "ROI TAC must be a TimeActivityCurve object."
        assert isinstance(scan_info, ScanTimingInfo), "Scan timing information must be a ScanTimingInfo object."

        if tcm_model_func not in self.SUPPORTED_MODELS:
            raise ValueError(
                    f"tcm_model_func must be one of: "
                    f"{', '.join(f.__name__ for f in self.SUPPORTED_MODELS)}"
                    )

    def _setup_bounds(self, fit_bounds: np.ndarray | None) -> np.ndarray:
        r"""
        Set up parameter bounds for curve fitting.

        If bounds are provided, validates their shape. Otherwise, uses default bounds from the model configuration.

        Args:
            fit_bounds (np.ndarray or None): User-provided bounds with shape (num_params, 3) where each row
                contains [initial_guess, lower_bound, upper_bound]. If None, default bounds are used.

        Returns:
            np.ndarray: Parameter bounds array with shape (num_params, 3).

        Raises:
            ValueError: If fit_bounds has incorrect shape for the model.

        """
        expected_shape = (self.model_config.num_params, 3)

        if fit_bounds is not None:
            if fit_bounds.shape != expected_shape:
                raise ValueError(
                        f"fit_bounds has wrong shape {fit_bounds.shape}. "
                        f"Expected {expected_shape} for parameters: "
                        f"{', '.join(self.model_config.param_names)}"
                        )
            return fit_bounds.copy()

        return self.model_config.default_bounds.copy()

    def gen_fit_params(self):
        r"""
        Generate lmfit Parameters object from bounds.

        Creates an :class:`lmfit.Parameters` object with parameter names, initial values,
        and bounds configured from the model configuration and bounds arrays.

        Returns:
            lmfit.Parameters: Configured parameters object for fitting.

        See Also:
            * :func:`lmfit.create_params`
        """
        params_dict = {name: {'vary': True, 'value': guess, "min": min, "max": max} for
                       name, guess, min, max in
                       zip(self.model_config.param_names,
                           self.initial_guesses,
                           self.bounds_lo,
                           self.bounds_hi)}
        return lmfit.create_params(**params_dict)

    def _setup_weights(self, fit_weights: np.ndarray | str | None) -> np.ndarray | None:
        r"""
        Set up weights for weighted least squares fitting.

        Args:
            fit_weights (np.ndarray, str, or None): Specification for weights. Can be: None: No weighting;
                'roi', 'roi-tac', 'tac-err', 'stderr': Use ROI TAC uncertainties;
                 np.ndarray: Custom weight array

        Returns:
            np.ndarray or None: Weight array for fitting, or None for uniform weighting.

        Side Effects:
            Issues a warning if a string weight specification is not recognized.
        """
        if fit_weights is None:
            return None
        elif isinstance(fit_weights, str):
            normalized = fit_weights.lower().replace(" ", "-")
            if normalized in {'roi', 'roi-tac', 'tac-err', 'stderr'}:
                    return self.roi_tac.uncertainty
            warnings.warn(
                    f"Unrecognized weights='{fit_weights}'. "
                    f"Valid options: 'roi', 'roi-tac', 'tac-err', 'stderr'. "
                    f"Setting weights to None.",
                    stacklevel=2
                    )
            return None
        else:
            cleaned_weights = fit_weights.copy()
            cleaned_weights[cleaned_weights == 0] = np.inf
            return cleaned_weights

    def run_fit(self) -> None:
        r"""
        Run the optimization/fitting process on the frame-averaged data.

        Performs the least-squares fitting using :mod:`lmfit`, extracts the fitted parameters
        and covariance matrix, calculates residuals, and generates the fitted TAC.

        Returns:
            None

        Side Effects:
            - result_obj (lmfit.minimizer.MinimizerResult): Stores the complete lmfit result object.
            - fit_results (tuple[np.ndarray, np.ndarray]): Stores fitted parameters and covariance matrix.
            - fit_residuals (np.ndarray): Stores the residuals from the fit.
            - fit_sum_of_square_residuals (float): Stores the sum of squared residuals.
            - fit_tac (TimeActivityCurve): Stores the model-predicted TAC using fitted parameters.

        See Also:
            * :meth:`lmfit.Minimizer.leastsq`
        """
        self._fit_obj.leastsq(**self.leastsq_kwargs)
        self.result_obj = self._fit_obj.result

        _fit_vals = np.asarray([val.value for _, val in self.result_obj.params.items()])
        self.fit_results = _fit_vals, self.result_obj.covar

        self.fit_residuals = self.result_obj.residual.copy()
        self.fit_sum_of_square_residuals = np.sum(self.fit_residuals ** 2)

        _fit_tac_activity = self.tcm_func(self.result_obj.params, *self.fine_input_tac.tac, self.frame_idx_pairs)
        self.fit_tac = TimeActivityCurve(self.roi_tac.times_in_mins, _fit_tac_activity)

    def __call__(self):
        r"""
        Execute the fit by calling the instance.

        Convenience method that runs :meth:`run_fit`.

        Side Effects:
            Runs the fitting process and populates result attributes.
        """
        self.run_fit()


class FrameAveragedTCMAnalysis():
    r"""
    A class for complete TCM analysis using files on frame-averaged TAC data.

    This class provides a high-level interface for performing tissue compartment model fitting
    on frame-averaged PET data. It handles file I/O, fitting, and result export including
    JSON parameter files. It wraps :class:`FrameAveragedTACFitter` with additional functionality
    for loading data from files and saving results.

    Attributes:
        input_tac_path (str): Absolute path to input TAC file.
        roi_tac_path (str): Absolute path to ROI TAC file.
        scan_info_path (str): Absolute path to scan timing information. Typically a JSON file with metadata from a
            NIfTI file. Can also be the path to a NIfTI file if the metadata have the same name as the NIfTI file, but
            the extension is json.
        output_directory (str): Absolute path to output directory for results.
        output_filename_prefix (str): Prefix for output filenames.
        compartment_model (str): Normalized name of the compartment model.
        short_tcm_name (str): Shortened model name without hyphens.
        _tcm_func (Callable): The TCM function corresponding to the compartment model.
        _model_config (FrameAvgdTcmModelConfig): Configuration for the TCM model.
        bounds (np.ndarray or None): Parameter bounds for fitting.
        weights (float, np.ndarray, or None): Weights for fitting.
        resample_num (int): Number of points for TAC resampling.
        fitter_class (type): The fitter class to use (FrameAveragedTACFitter).
        analysis_props (dict): Dictionary containing analysis properties and results.
        fit_results (tuple or None): Fitted parameters and covariance matrix.
        _has_analysis_been_run (bool): Flag indicating if analysis has been executed.

    Example:
        .. code-block:: python

            import petpal.kinetic_modeling.tac_fitting as pet_fit

            analysis = pet_fit.FrameAveragedTCMAnalysis(
                input_tac_path='./input.tsv',
                roi_tac_path='./roi.tsv',
                scan_info_path='./pet_image.nii.gz', # Assumes that ./pet_image.json is the metadata file
                output_directory='./results',
                output_filename_prefix='subject01',
                compartment_model='serial-2tcm'
            )

            # Run and save
            analysis()

            # Or step by step
            analysis.run_analysis()
            analysis.save_analysis()

    See Also:
        * :class:`FrameAveragedTACFitter`
        * :class:`FrameAveragedMultiTACTCMAnalysis` for multiple ROIs
        * :class:`TCMAnalysis` for non-frame-averaged analysis
    """
    def __init__(self,
                 input_tac_path: str,
                 roi_tac_path: str,
                 scan_info_path: str,
                 output_directory: str,
                 output_filename_prefix: str,
                 compartment_model: str,
                 parameter_bounds: None | np.ndarray = None,
                 weights: float | None | np.ndarray = None,
                 resample_num: int = 8192):
        r"""
        Initialize a FrameAveragedTCMAnalysis instance.

        Args:
            input_tac_path (str): Path to the input/plasma TAC file.
            roi_tac_path (str): Path to the ROI/tissue TAC file.
            scan_info_path (str): Path to scan timing information file. Typically a JSON file with metadata from a
                NIfTI file. Can also be the path to a NIfTI file if the metadata have the same name as the NIfTI file, but
                the extension is json.
            output_directory (str): Directory for saving analysis results.
            output_filename_prefix (str): Prefix for output filenames.
            compartment_model (str): Name of the compartment model (e.g., '1tcm', 'serial-2tcm').
            parameter_bounds (np.ndarray or None, optional): Parameter bounds with shape (num_params, 3).
                Defaults to None.
            weights (float, np.ndarray, or None, optional): Weights for fitting. Defaults to None.
            resample_num (int, optional): Number of points for TAC resampling. Defaults to 8192.
        """
        self.input_tac_path = os.path.abspath(input_tac_path)
        self.roi_tac_path = os.path.abspath(roi_tac_path)
        self.scan_info_path = os.path.abspath(scan_info_path)
        self.output_directory = os.path.abspath(output_directory)
        self.output_filename_prefix = output_filename_prefix
        self.compartment_model = self.validated_tcm_name(compartment_model)
        self.short_tcm_name = "".join(self.compartment_model.split("-"))
        self._tcm_func = self.validated_tcm_function(self.compartment_model)
        self._model_config = _FRAME_AVGD_TCM_CONFIGS[self._tcm_func]
        self.bounds = parameter_bounds
        self.weights = weights
        self.resample_num = resample_num
        self.fitter_class = FrameAveragedTACFitter
        self.analysis_props: dict = self.init_analysis_props()
        self.fit_results: None | tuple | list= None
        self._has_analysis_been_run: bool = False

    def init_analysis_props(self) -> dict:
        r"""
        Initialize the analysis properties dictionary structure.

        Creates a dictionary to store metadata, file paths, model information, and fit results.
        This dictionary can be saved as a JSON file for parsing results.

        Returns:
            dict: Initialized dictionary with analysis properties structure including:
                - FilePathPTAC: Path to input TAC
                - FilePathTTAC: Path to ROI TAC
                - TissueCompartmentModel: Model name
                - FitProperties: Nested dict with FitValues, FitStdErr, Bounds, and ResampleNum

        See Also:
            * :meth:`calculate_fit_properties`
        """
        props = {
            'FilePathPTAC'            : self.input_tac_path,
            'FilePathTTAC'            : self.roi_tac_path,
            'TissueCompartmentModel'  : self.compartment_model,
            'FitProperties'           : {
                'FitValues'    : [],
                'FitStdErr'    : [],
                'Bounds'       : [],
                'ResampleNum'  : self.resample_num,
                }
            }

        return props

    @staticmethod
    def validated_tcm_name(compartment_model: str) -> str:
        r"""
        Validate and normalize a tissue compartment model name.

        Args:
            compartment_model (str): The compartment model name to validate.

        Returns:
            str: The normalized model name.

        Raises:
            ValueError: If the model name is not valid for frame-averaged models.

        See Also:
            * :meth:`~.FrameAvgdTcmModelConfig.normalize_name`
        """
        FrameAvgdTcmModelConfig.resolve_model_name(compartment_model)
        return FrameAvgdTcmModelConfig.normalize_name(compartment_model)

    @staticmethod
    def validated_tcm_function(compartment_model: str) -> Callable:
        r"""
        Get the TCM function corresponding to a model name.

        Args:
            compartment_model (str): The compartment model name.

        Returns:
            Callable: The frame-averaged TCM function.

        Raises:
            ValueError: If the model name does not correspond to a known frame-averaged model.

        See Also:
            * :meth:`FrameAvgdTcmModelConfig.resolve_model_name`
        """
        return FrameAvgdTcmModelConfig.resolve_model_name(compartment_model)

    def __call__(self, pretty_params: bool = False):
        r"""Execute the complete analysis workflow.

        Convenience method that runs the analysis and saves results.

        Args:
            pretty_params (bool, optional): If True, use LaTeX-formatted parameter names in output.
                Defaults to False.

        Side Effects:
            Runs the fit analysis and saves results to the output directory.

        See Also:
            * :meth:`run_analysis`
            * :meth:`save_analysis`
        """
        self.run_analysis(pretty_params=pretty_params)
        self.save_analysis()

    def run_analysis(self, pretty_params: bool = False):
        r"""Run the fitting analysis workflow.

        Executes :meth:`calculate_fit` and :meth:`calculate_fit_properties`, then sets
        the analysis-has-been-run flag.

        Args:
            pretty_params (bool, optional): If True, use LaTeX-formatted parameter names.
                Defaults to False.

        Side Effects:
            - Populates fit_results with fitted parameters and covariances.
            - Updates analysis_props with formatted fit values.
            - Sets _has_analysis_been_run to True.
        """
        self.calculate_fit()
        self.calculate_fit_properties(pretty_params=pretty_params)
        self._has_analysis_been_run = True

    def save_analysis(self):
        r"""Save analysis results to a JSON file.

        Saves the analysis_props dictionary containing fit results and metadata to a JSON file
        in the output directory. The filename is constructed from the output prefix, model name,
        and '_fitprops.json' suffix.

        Raises:
            RuntimeError: If :meth:`run_analysis` has not been called before this method.

        Side Effects:
            Writes a JSON file to the output directory.
        """
        if not self._has_analysis_been_run:
            raise RuntimeError("`run_analysis` must be called before saving analysis.")
        file_name_prefix = os.path.join(self.output_directory,
                                        f"{self.output_filename_prefix}_desc"
                                        f"-{self.short_tcm_name}")
        analysis_props_file = f"{file_name_prefix}_fitprops.json"
        with open(analysis_props_file, 'w', encoding='utf-8') as f:
            json.dump(obj=self.analysis_props, fp=f, indent=4)

    def calculate_fit(self):
        r"""Perform the TCM fit on the loaded TAC data.

        Loads TACs and scan timing from files, creates a :class:`FrameAveragedTACFitter` instance,
        runs the fit, and stores results.

        Side Effects:
            - Populates fit_results with fitted parameters and covariance matrix.
            - Sets bounds if they were None initially.

        See Also:
            * :class:`~.FrameAveragedTACFitter`
            * :class:`~.TimeActivityCurve`
            * :class:`~.ScanTimingInfo`
        """
        fitter_cls = self.fitter_class(input_tac=TimeActivityCurve.from_tsv(self.input_tac_path),
                                       roi_tac=TimeActivityCurve.from_tsv(self.roi_tac_path),
                                       scan_info=ScanTimingInfo.from_nifti(self.scan_info_path),
                                       tcm_model_func=self._tcm_func,
                                       fit_bounds=self.bounds,
                                       fit_weights=self.weights,
                                       tac_resample_num=self.resample_num,
                                       )
        fitter_cls()
        self.fit_results = fitter_cls.fit_results
        self.bounds = fitter_cls.bounds if self.bounds is None else self.bounds

    def calculate_fit_properties(self, pretty_params: bool = False):
        r"""Calculate and format fit properties for output.

        Extracts fitted parameters, standard errors, and bounds, then formats them
        into the analysis_props dictionary.

        Args:
            pretty_params (bool, optional): If True, use LaTeX-formatted parameter names.
                Defaults to False.

        Side Effects:
            Updates the analysis_props dictionary with formatted fit results.

        See Also:
            * :meth:`update_props_with_formatted_fit_values`
        """
        self.update_props_with_formatted_fit_values(fit_values=self.fit_results[0],
                                                    fit_covars=self.fit_results[1],
                                                    param_bounds=self.bounds,
                                                    fit_props_dict=self.analysis_props,
                                                    pretty_params=pretty_params)

    def update_props_with_formatted_fit_values(self,
                                               fit_values: np.ndarray,
                                               fit_covars: np.ndarray,
                                               param_bounds: np.ndarray,
                                               fit_props_dict: dict,
                                               pretty_params: bool = False) -> None:
        r"""Update properties dictionary with formatted fit results.

        Computes standard errors from covariances, formats parameter names (optionally with LaTeX),
        and updates the fit properties dictionary.

        Args:
            fit_values (np.ndarray): Array of fitted parameter values.
            fit_covars (np.ndarray): Covariance matrix from the fit.
            param_bounds (np.ndarray): Parameter bounds array.
            fit_props_dict (dict): Dictionary to update with formatted results.
            pretty_params (bool, optional): If True, use LaTeX-formatted parameter names.
                Defaults to False.

        Side Effects:
            Updates the FitProperties section of fit_props_dict with FitValues, FitStdErr, and Bounds.
        """
        try:
            fit_stderr = np.sqrt(np.diagonal(fit_covars))
        except ValueError:
            fit_stderr = np.nan * np.ones_like(fit_values)

        param_names = self._model_config.pretty_param_names if pretty_params else self._model_config.param_names

        formatted_values = {name:value for name, value in zip(param_names, fit_values.round(5).tolist())}
        formatted_stderr = {name: value for name, value in zip(param_names, fit_stderr.round(5).tolist())}

        fit_props_dict["FitProperties"]["FitValues"] = formatted_values
        fit_props_dict["FitProperties"]["FitStdErr"] = formatted_stderr

        _format_func = self._generate_formatted_bounds
        fit_props_dict["FitProperties"]["Bounds"] = _format_func(param_bounds=param_bounds,
                                                                 pretty_params=pretty_params)

    def _generate_formatted_bounds(self, param_bounds: np.ndarray, pretty_params: bool = False) -> dict:
        r"""Format parameter bounds into a dictionary structure.

        Args:
            param_bounds (np.ndarray): Parameter bounds array with shape (num_params, 3).
            pretty_params (bool, optional): If True, use LaTeX-formatted parameter names.
                Defaults to False.

        Returns:
            dict: Formatted bounds dictionary with structure:
                {param_name: {'initial': val, 'lo': lo_bound, 'hi': hi_bound}}
        """
        param_names = self._model_config.pretty_param_names if pretty_params else self._model_config.param_names
        formatted_bounds = {param: {'initial': val[0], 'lo': val[1], 'hi': val[2]} for param, val in
                            zip(param_names, param_bounds)}
        return formatted_bounds

class FrameAveragedMultiTACTCMAnalysis(FrameAveragedTCMAnalysis, MultiTACAnalysisMixin):
    r"""
    A class for performing frame-averaged TCM analysis on multiple tissue TACs.

    This class extends :class:`~.FrameAveragedTCMAnalysis` to handle multiple ROI TACs simultaneously,
    fitting each to the same compartment model. It inherits functionality from both
    :class:`~.FrameAveragedTCMAnalysis` and :class:`~.MultiTACAnalysisMixin`, and produces both individual
    JSON files for each ROI and consolidated TSV tables summarizing all results.

    Attributes:
        input_tac_path (str): Path to the input TAC file.
        roi_tacs_dir (str): Directory containing multiple ROI TAC files.
        scan_info_path (str): Absolute path to scan timing information. Typically a JSON file with metadata from a
            NIfTI file. Can also be the path to a NIfTI file if the metadata have the same name as the NIfTI file, but
            the extension is json.
        output_directory (str): Directory for saving analysis results.
        output_filename_prefix (str): Prefix for output filenames.
        compartment_model (str): Name of the compartment model to use.
        parameter_bounds (np.ndarray or None): Bounds for the fitting parameters.
        weights (float, np.ndarray, or None): Weights for fitting.
        resample_num (int): Number of resampling points.
        fit_results (list): List of fit results for each TAC.
        fit_tacs (list[TimeActivityCurve]): List of fitted TAC curves for each ROI.

    Example:
        .. code-block:: python

            import petpal.kinetic_modeling.tac_fitting as pet_fit

            analysis = pet_fit.FrameAveragedMultiTACTCMAnalysis(
                input_tac_path='./input.tsv',
                roi_tacs_dir='./roi_tacs/',
                scan_info_path='./pet_image.nii.gz', # Assuming that './pet_image.json/' exists.
                output_directory='./results',
                output_filename_prefix='sub-0001_ses-01',
                compartment_model='serial-2tcm'
            )

            analysis()  # Run and save all results

    See Also:
        * :class:`~.FrameAveragedTCMAnalysis`
        * :class:`~.MultiTACAnalysisMixin`
        * :class:`~.MultiTACTCMAnalysis` for non-frame-averaged multi-TAC analysis
    """
    def __init__(self,
                 input_tac_path: str,
                 roi_tacs_dir: str,
                 scan_info_path: str,
                 output_directory: str,
                 output_filename_prefix: str,
                 compartment_model: str,
                 parameter_bounds: None | np.ndarray = None,
                 weights: float | None | np.ndarray = None,
                 resample_num: int = 4096):
        r"""
        Initialize a FrameAveragedMultiTACTCMAnalysis instance.

        Args:
            input_tac_path (str): Path to the input/plasma TAC file.
            roi_tacs_dir (str): Directory containing multiple ROI TAC files.
            scan_info_path (str): Path to scan timing information. Typically a JSON file with metadata from a
                NIfTI file. Can also be the path to a NIfTI file if the metadata have the same name as the NIfTI file, but
                the extension is json.
            output_directory (str): Directory for saving analysis results.
            output_filename_prefix (str): Prefix for output filenames. Typically something like 'sub-XXXX_ses-XX'.
            compartment_model (str): Name of the compartment model.
            parameter_bounds (np.ndarray or None, optional): Parameter bounds. Defaults to None.
            weights (float, np.ndarray, or None, optional): Weights for fitting. Defaults to None.
            resample_num (int, optional): Number of resampling points. Defaults to 4096.
        """
        MultiTACAnalysisMixin.__init__(self,
                                       input_tac_path=input_tac_path,
                                       tacs_dir=roi_tacs_dir, )
        FrameAveragedTCMAnalysis.__init__(self,
                                          input_tac_path=input_tac_path,
                                          roi_tac_path=roi_tacs_dir,
                                          scan_info_path=scan_info_path,
                                          output_directory=output_directory,
                                          output_filename_prefix=output_filename_prefix,
                                          compartment_model=compartment_model,
                                          parameter_bounds=parameter_bounds,
                                          weights=weights,
                                          resample_num=resample_num)
        self.fit_results = []
        self.fit_tacs: list[TimeActivityCurve] = []

    def init_analysis_props(self) -> list[dict]:
        r"""
        Initialize analysis properties for each tissue TAC.

        Creates a list of analysis property dictionaries, one for each TAC file found in the
        ROI TACs directory. Updates each dictionary with the appropriate file paths.

        Returns:
            list[dict]: A list of analysis property dictionaries for each TAC.

        Side Effects:
            Updates FilePathTTAC and FilePathImageOrMetadata for each properties dictionary.
        """
        num_of_tacs = self.num_of_tacs
        analysis_props = [FrameAveragedTCMAnalysis.init_analysis_props(self) for a_tac in range(num_of_tacs)]
        for tac_id, a_prop_dict in enumerate(analysis_props):
            a_prop_dict['FilePathTTAC'] = os.path.abspath(self.tacs_files_list[tac_id])
            a_prop_dict['FilePathImageOrMetadata'] = self.scan_info_path
        return analysis_props

    def calculate_fit(self):
        r"""
        Calculate the fit for each TAC in the directory.

        Loads the input TAC and scan timing once, then iterates through all ROI TAC files,
        fitting each one and storing results.

        Side Effects:
            - Populates fit_results list with fit results for each TAC.
            - Populates fit_tacs list with fitted TAC curves.
            - Sets bounds if they were None initially.

        See Also:
            * :class:`~.FrameAveragedTACFitter`
        """
        p_tac = TimeActivityCurve.from_tsv(self.input_tac_path)
        scan_info = ScanTimingInfo.from_nifti(self.scan_info_path)
        fitter_cls = None
        for tac_id, a_tac in enumerate(self.tacs_files_list):
            t_tac = TimeActivityCurve.from_tsv(a_tac)
            fitter_cls = self.fitter_class(input_tac=p_tac,
                                           roi_tac=t_tac,
                                           scan_info=scan_info,
                                           tcm_model_func=self._tcm_func,
                                           tac_resample_num=self.resample_num,
                                           )
            fitter_cls.run_fit()
            self.fit_results.append(fitter_cls.fit_results)
            self.fit_tacs.append(fitter_cls.fit_tac)
            # self.analysis_props[tac_id]['FitProperties']['Sum Of Squared Residuals'] = fitter_cls.fit_sum_of_square_residuals
        if (self.bounds is None) and (fitter_cls is not None):
            self.bounds = fitter_cls.bounds

    def calculate_fit_properties(self, pretty_params: bool = False):
        r"""
        Calculate and format fit properties for all TACs.

        Iterates through fit results and updates each ROI's analysis properties dictionary
        with formatted fit values, standard errors, and bounds.

        Args:
            pretty_params (bool, optional): If True, use LaTeX-formatted parameter names.
                Defaults to False.

        Side Effects:
            Updates each dictionary in analysis_props with formatted fit results.
        """
        for fit_results, fit_props in zip(self.fit_results, self.analysis_props):
            self.update_props_with_formatted_fit_values(fit_values=fit_results[0],
                                                        fit_covars=fit_results[1],
                                                        fit_props_dict=fit_props,
                                                        param_bounds=self.bounds,
                                                        pretty_params=pretty_params)


    def save_analysis(self):
        r"""
        Save all analysis results to files.

        Saves three types of outputs:
            1. Individual JSON files for each ROI's fit properties
            2. A consolidated TSV table with all TACs and fits
            3. A consolidated TSV table with all fit parameters

        Side Effects:
            Writes multiple files to the output directory.

        See Also:
            * :meth:`_save_fit_props`
            * :meth:`_save_multitacs_table`
            * :meth:`_save_multifitprops_table`
        """
        self._save_fit_props()
        self._save_multitacs_table()
        self._save_multifitprops_table()

    def _save_fit_props(self):
        r"""
        Save individual JSON fit properties files for each ROI.

        Creates a separate JSON file for each ROI containing its fit properties,
        with filenames including the ROI segment label.

        Side Effects:
            Writes multiple JSON files to the output directory, one per ROI.
        """
        for seg_name, fit_props in zip(self.inferred_seg_labels, self.analysis_props):
            filename = [self.output_filename_prefix,
                        f'desc-{self.short_tcm_name}',
                        f'seg-{seg_name}',
                        'fitprops.json']
            filename = '_'.join(filename)
            filepath = os.path.join(self.output_directory, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(obj=fit_props, fp=f, indent=4)

    def _save_multitacs_table(self):
        r"""
        Save a consolidated table of all TACs, fits, and residuals.

        Creates a tab-separated values (TSV) file containing time points and, for each ROI:
        measured activity, uncertainty, fitted activity, and residuals.

        Side Effects:
            Writes a TSV file named with pattern: {prefix}_desc-{model}_multitacs.tsv

        See Also:
            * :class:`TimeActivityCurve`
        """
        tacs_header: list[str] | str = ['Time(mins)']
        tacs_table: list[np.ndarray] | np.ndarray = [self.fit_tacs[0].times_in_mins]
        for seg_id, seg_name in enumerate(self.inferred_seg_labels):
            tacs_header.append(f'seg-{seg_name}_activity')
            tacs_header.append(f'seg-{seg_name}_uncertainty')
            tacs_header.append(f'seg-{seg_name}_fit')
            tacs_header.append(f'seg-{seg_name}_residuals')
            roi_tac = TimeActivityCurve.from_tsv(self.tacs_files_list[seg_id])
            fit_tac = self.fit_tacs[seg_id]
            tacs_table.append(roi_tac.activity)
            tacs_table.append(roi_tac.uncertainty)
            tacs_table.append(fit_tac.activity)
            tacs_table.append(roi_tac.activity - fit_tac.activity)

        tacs_header = "\t".join(tacs_header)
        tacs_table = np.asarray(tacs_table)

        filename = [self.output_filename_prefix,
                    f'desc-{self.short_tcm_name}',
                    'multitacs.tsv']
        filename = '_'.join(filename)
        filepath = os.path.join(self.output_directory, filename)
        np.savetxt(fname=filepath, X=tacs_table.T, delimiter='\t',
                   fmt='%.8e', header=tacs_header, comments='')

    def _save_multifitprops_table(self):
        r"""
        Save a consolidated table of all fit parameters.

        Creates a tab-separated values (TSV) file containing fitted parameter values and
        standard errors for all ROIs in a long-format table.

        The output table has columns: seg-name, param, value, stderr

        Side Effects:
            Writes a TSV file named with pattern: {prefix}_desc-{model}_multifitprops.tsv

        See Also:
            * :mod:`pandas`
        """
        _segs = []
        _params = []
        _vals = []
        _errs = []
        for seg_id, (seg_name, seg_analysis) in enumerate(zip(self.inferred_seg_labels, self.analysis_props)):
            _fit_vals = seg_analysis['FitProperties']['FitValues']
            _fit_errs = seg_analysis['FitProperties']['FitStdErr']
            for param, val, err in zip(_fit_vals, _fit_vals.values(), _fit_errs.values()):
                _segs.append(seg_name)
                _params.append(param)
                _vals.append(val)
                _errs.append(err)

        _tmp_km_df = pd.DataFrame.from_dict({'seg-name': _segs, 'param': _params, 'value': _vals, 'stderr': _errs})

        filename = [self.output_filename_prefix,
                    f'desc-{self.short_tcm_name}',
                    'multifitprops.tsv']
        filename = '_'.join(filename)
        filepath = os.path.join(self.output_directory, filename)
        _tmp_km_df.to_csv(path_or_buf=filepath, sep='\t', index=False)