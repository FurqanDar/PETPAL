import numpy as np
from scipy.optimize import curve_fit as sp_fit
from . import tcms_as_convolutions as tcms_conv


def calc_srtm_tac(tac_times: np.ndarray, r1: float, k2: float, bp: float, ref_tac_vals: np.ndarray) -> np.ndarray:
    first_term = r1 * ref_tac_vals
    bp_coeff = k2 / (1.0 + bp)
    exp_term = np.exp(-bp_coeff * tac_times)
    dt = tac_times[1] - tac_times[0]
    second_term = (k2 - r1 * bp_coeff) * tcms_conv.calc_convolution_with_check(f=exp_term, g=ref_tac_vals, dt=dt)
    
    return first_term + second_term


def calc_frtm_tac(tac_times: np.ndarray,
                  r1: float,
                  a1: float,
                  a2: float,
                  alpha_1: float,
                  alpha_2: float,
                  ref_tac_vals: np.ndarray) -> np.ndarray:
    first_term = r1 * ref_tac_vals
    exp_funcs = a1 * np.exp(-alpha_1 * tac_times) + a2 * np.exp(-alpha_2 * tac_times)
    dt = tac_times[1] - tac_times[0]
    second_term = tcms_conv.calc_convolution_with_check(f=exp_funcs, g=ref_tac_vals, dt=dt)
    return first_term + second_term

