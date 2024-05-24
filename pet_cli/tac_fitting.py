import inspect
from typing import Callable, Union
import numpy as np
from scipy.optimize import curve_fit as sp_cv_fit
from . import tcms_as_convolutions as pet_tcms
from . import blood_input as pet_bld
import os


def get_fitting_params_for_tcm_func(f: Callable) -> list:
    r"""
    Fetches the parameter names from the function signature of a given Tissue Compartment Model (TCM) function. The
    functions can be one of the following:
        * :func:`generate_tac_1tcm_c1_from_tac<pet_cli.tcms_as_convolutions.generate_tac_1tcm_c1_from_tac>`
        * :func:`generate_tac_2tcm_with_k4zero_cpet_from_tac<pet_cli.tcms_as_convolutions.generate_tac_2tcm_with_k4zero_cpet_from_tac>`
        * :func:`generate_tac_serial_2tcm_cpet_from_tac<pet_cli.tcms_as_convolutions.generate_tac_serial_2tcm_cpet_from_tac>`

    Args:
        f (Callable): TCM function.

    Returns:
        list: List of parameter names.
    """
    return list(inspect.signature(f).parameters.keys())[2:]


def get_number_of_fit_params_for_tcm_func(f: Callable) -> int:
    r"""
    Counts the number of fitting parameters for a given Tissue Compartment Model (TCM) function. The
    functions can be one of the following:
        * :func:`generate_tac_1tcm_c1_from_tac<pet_cli.tcms_as_convolutions.generate_tac_1tcm_c1_from_tac>`
        * :func:`generate_tac_2tcm_with_k4zero_cpet_from_tac<pet_cli.tcms_as_convolutions.generate_tac_2tcm_with_k4zero_cpet_from_tac>`
        * :func:`generate_tac_serial_2tcm_cpet_from_tac<pet_cli.tcms_as_convolutions.generate_tac_serial_2tcm_cpet_from_tac>`

    Args:
        f (Callable): TCM function.

    Returns:
        int: Number of fitting parameters.
    """
    return len(get_fitting_params_for_tcm_func(f))


class TACFitter(object):
    def __init__(self,
                 pTAC: np.ndarray,
                 tTAC: np.ndarray,
                 weights: Union[None, float, np.ndarray] = None,
                 tcm_func: Callable = None,
                 fit_bounds: Union[np.ndarray, None] = None,
                 resample_num: int = 512,
                 aif_fit_thresh_in_mins: float = 30.0,
                 max_iters: int = 2500):
        
        self.max_func_evals: int = max_iters
        self.tcm_func: Callable = None
        self.fit_param_number: int = None
        self.fit_param_names: list = None
        
        self.bounds: np.ndarray = None
        self.initial_guesses: np.ndarray = None
        self.bounds_lo: np.ndarray = None
        self.bounds_hi: np.ndarray = None
        
        self.get_tcm_func_properties(tcm_func)
        self.set_bounds_and_initial_guesses(fit_bounds)
        
        self.raw_p_tac: np.ndarray = pTAC.copy()
        self.raw_t_tac: np.ndarray = tTAC.copy()
        self.sanitized_t_tac: np.ndarray = None
        self.sanitized_p_tac: np.ndarray = None
        self.resample_times: np.ndarray = None
        self.delta_t: float = None
        self.resampled_t_tac: np.ndarray = None
        self.resampled_p_tac: np.ndarray = None
        
        self.resample_tacs_evenly(aif_fit_thresh_in_mins, resample_num)
        
        self.weights: np.ndarray = None
        self.set_weights(weights)
        
        self.p_tac_vals: np.ndarray = self.resampled_p_tac[1]
        self.tgt_tac_vals: np.ndarray = self.resampled_t_tac[1]
        self.fit_results = None
    
    def set_bounds_and_initial_guesses(self, fit_bounds: np.ndarray) -> None:
        assert self.tcm_func is not None, "This method should be run after `get_tcm_func_properties`"
        if fit_bounds is not None:
            assert fit_bounds.shape == (self.fit_param_number, 3), ("Fit bounds has the wrong shape. For each potential"
                                                                    " fitting parameter in `tcm_func`, we require the "
                                                                    "tuple: `(initial, lower, upper)`.")
            self.bounds = fit_bounds.copy()
        else:
            bounds = np.zeros((self.fit_param_number, 3), float)
            for pid, param in enumerate(bounds[:-1]):
                bounds[pid] = [0.1, 1.0e-8, 5.0]
            bounds[-1] = [0.1, 0.0, 1.0]
            self.bounds = bounds.copy()
        
        self.initial_guesses = self.bounds[:, 0]
        self.bounds_lo = self.bounds[:, 1]
        self.bounds_hi = self.bounds[:, 2]
    
    def resample_tacs_evenly(self, fit_thresh_in_mins: float, resample_num: int) -> None:
        r"""
        Resample pTAC and tTAC evenly with respect to time, and at the same times.

        The method takes a threshold in minutes and a resample number as inputs. It starts by sanitizing
        the pTAC and tTAC (prepending a :math:`f(t=0)=0` point to data if necessary). A regularly sampled time is
        then generated using the start, end, and number of samples dictated by resample_num. Following this,
        an interpolation object is created using the :class:`pet_cli.blood_input.BloodInputFunction` class for the pTAC.
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
    
    def get_tcm_func_properties(self, tcm_func: Callable) -> None:
        assert tcm_func in [pet_tcms.generate_tac_1tcm_c1_from_tac,
                            pet_tcms.generate_tac_2tcm_with_k4zero_cpet_from_tac,
                            pet_tcms.generate_tac_serial_2tcm_cpet_from_tac], (
            "`tcm_func should be one of `pet_tcms.generate_tac_1tcm_c1_from_tac`, "
            "`pet_tcms.generate_tac_2tcm_with_k4zero_cpet_from_tac`, "
            "`pet_tcms.generate_tac_serial_2tcm_cpet_from_tac`")
        
        self.tcm_func = tcm_func
        self.fit_param_names = get_fitting_params_for_tcm_func(self.tcm_func)
        self.fit_param_number = len(self.fit_param_names)
    
    @staticmethod
    def sanitize_tac(tac_times: np.ndarray, tac_vals: np.ndarray) -> np.ndarray:
        assert tac_times.shape == tac_vals.shape, "`tac_times` and `tac_vals` must have the same shape."
        if tac_times[0] != 0.0:
            return np.asarray([np.append(0, tac_times), np.append(0, tac_vals)])
        else:
            out_vals = tac_vals[:]
            out_vals[0] = 0.0
            return np.asarray([tac_times, out_vals])
    
    @staticmethod
    def resample_tac_on_new_times(tac_times: np.ndarray, tac_vals: np.ndarray, new_times: np.ndarray) -> np.ndarray:
        return np.asarray([new_times, np.interp(x=new_times, xp=tac_times, fp=tac_vals)])
    
    def fitting_func(self, x: np.ndarray, *params) -> np.ndarray:
        return self.tcm_func(x, self.p_tac_vals, *params)[1]
    
    def run_fit(self) -> None:
        self.fit_results = sp_cv_fit(f=self.fitting_func, xdata=self.resample_times, ydata=self.tgt_tac_vals,
                                     p0=self.initial_guesses, bounds=(self.bounds_lo, self.bounds_hi),
                                     sigma=self.weights, maxfev=self.max_func_evals)


class TACFitterWithoutBloodVolume(TACFitter):
    def __init__(self,
                 pTAC: np.ndarray,
                 tTAC: np.ndarray,
                 weights: np.ndarray = None,
                 tcm_func: Callable = None,
                 fit_bounds: np.ndarray = None,
                 resample_num: int = 2048,
                 aif_fit_thresh_in_mins: float = 30.0,
                 max_iters: int = 2500):
        
        super().__init__(pTAC, tTAC, weights, tcm_func, fit_bounds, resample_num, aif_fit_thresh_in_mins, max_iters)
        self.get_tcm_func_properties(tcm_func)
        self.set_bounds_and_initial_guesses(fit_bounds)
    
    def get_tcm_func_properties(self, tcm_func: Callable) -> None:
        assert tcm_func in [pet_tcms.generate_tac_1tcm_c1_from_tac,
                            pet_tcms.generate_tac_2tcm_with_k4zero_cpet_from_tac,
                            pet_tcms.generate_tac_serial_2tcm_cpet_from_tac], (
            "`tcm_func should be one of `pet_tcms.generate_tac_1tcm_c1_from_tac`, "
            "`pet_tcms.generate_tac_2tcm_with_k4zero_cpet_from_tac`, "
            "`pet_tcms.generate_tac_serial_2tcm_cpet_from_tac`")
        
        self.tcm_func = tcm_func
        self.fit_param_names = get_fitting_params_for_tcm_func(self.tcm_func)[:-1]
        self.fit_param_number = len(self.fit_param_names)
    
    def set_bounds_and_initial_guesses(self, fit_bounds: np.ndarray) -> None:
        assert self.tcm_func is not None, "This method should be run after `get_tcm_func_properties`"
        if fit_bounds is not None:
            assert fit_bounds.shape == (self.fit_param_number, 3), ("Fit bounds has the wrong shape. For each potential"
                                                                    " fitting parameter in `tcm_func`, we require the "
                                                                    "tuple: `(initial, lower, upper)`.")
            self.bounds = fit_bounds.copy()
        else:
            bounds = np.zeros((self.fit_param_number, 3), float)
            for pid, param in enumerate(bounds[:]):
                bounds[pid] = [0.1, 1.0e-8, 5.0]
            self.bounds = bounds.copy()
        
        self.initial_guesses = self.bounds[:, 0]
        self.bounds_lo = self.bounds[:, 1]
        self.bounds_hi = self.bounds[:, 2]
    
    def fitting_func(self, x: np.ndarray, *params) -> np.ndarray:
        return self.tcm_func(x, self.p_tac_vals, *params, vb=0.0)[1]
