r"""
Command-line interface (CLI) for fitting Tissue Compartment Models (TCM) to PET Time-Activity Curves (TACs).

This module provides a CLI to interact with the tac_fitting module. It utilizes argparse to handle command-line
arguments.

The user must provide:
    * Input TAC file path
    * Region of Interest (ROI) TAC file path
    * Compartment model name for fitting. Supported models are '1tcm', '2tcm-k4zero', or 'serial-2tcm'.
    * Filename prefix for the output files
    * Output directory where the analysis results will be saved
    * Whether to ignore blood volume contributions while fitting
    * Threshold in minutes (input fitting threshold)

User can optionally provide:
    * Initial guesses for the fitting parameters
    * Lower and upper bounds for the fitting parameters
    * Maximum number of function iterations
    * Decay constant (in minutes) for per-frame weighting

This script utilizes the :class:`TCMAnalysis<petpal.tac_fitting.TCMAnalysis>` class to perform the TAC fitting and save
 the results accordingly.

Examples:
    In the proceeding example, we assume that we have an input TAC named 'input_tac.txt', and an ROI TAC named
    'roi_tac.tsv'. We want to try fitting a serial 2TCM to the ROI tac.
    
    .. code-block:: bash

        petpal-tcm-fit interp -i "input_tac.tsv"\
        -r "roi_tac.tsv"\
        -m "serial-2tcm"\
        -o "./" -p "cli_"\
        -t 35.0\
        -g 0.1 0.1 0.1 0.1 0.1\
        -l 0.0 0.0 0.0 0.0 0.0\
        -u 5.0 5.0 5.0 5.0 5.0\
        -f 2500 -n 4096 -b --print

    In this next example, we assume that we have an input TAC named 'input_tac.txt', an ROI TACs directory
     named './roi_tas/' which contains TACs for different ROIs, and a scan/image metadata file called 'scan_info.json'.
     We want to try fitting serial 2TCM to the ROI tacs, with the default bounds.

    .. code-block:: bash

        petpal-tcm-fit frame_avgd -i "input_tac.tsv"\
        -r "./roi_tacs/"\
        -s "scan_info.json"\
        -m "serial-2tcm"\
        -o "./" -p "cli_"\
        -f 2500 -n 4096 -b --print

See Also:
    :mod:`petpal.tac_fitting` - module for fitting TACs with TCMs.

"""

from typing import Union
import os
import numpy as np
import argparse
from ..kinetic_modeling import tac_fitting as pet_fit

_EXAMPLE_ = ('Fitting a TAC to the serial 2TCM using the interpolation strategy:\n\t'
             'petpal-tcm-fit interp -i "input_tac.txt"'
             ' -r "2tcm_tac.txt" '
             '-m "serial-2tcm" '
             '-o "./" -p "cli_" -t 35.0 '
             '-g 0.1 0.1 0.1 0.1 0.1 '
             '-l 0.0 0.0 0.0 0.0 0.0 '
             '-u 5.0 5.0 5.0 5.0 5.0 '
             '-f 2500 -n 4096 -b '
             '--print\n\n'
             'Fitting a TAC to the serial 2TCM using the frame averaged strategy:\n\t'
             'petpal-tcm-fit frame_avgd -i "input_tac.txt"'
             ' -r "2tcm_tac.txt" '
             '-s scan_metadata.json '
             '-m "serial-2tcm" '
             '-o "./" -p "cli_" '
             '-g 0.1 0.1 0.1 0.1 0.1 '
             '-l 0.0 0.0 0.0 0.0 0.0 '
             '-u 5.0 5.0 5.0 5.0 5.0 '
             '-f 2500 -n 4096 -b '
             '--print'
             )

def add_common_io_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    r"""Adds common input/output arguments to the provided argument parser.

    This helper function configures arguments that are shared between all subcommands
    for specifying input TAC paths, ROI TAC paths, output directory and output filename prefix.

    Args:
        parser (argparse.ArgumentParser): Argument parser (or subparser) to which the
            common I/O arguments will be added.

    Returns:
        argparse._ArgumentGroup: The argument group containing the I/O arguments.
    """
    grp_io = parser.add_argument_group('IO Paths and Prefixes')
    grp_io.add_argument("-i", "--input-tac-path", required=True, help="Path to the input TAC file.")
    grp_io.add_argument("-r", "--roi-tac-path", required=True, help="Path to the ROI TAC file. "
                                                                    "Can also be the path to a directory where each "
                                                                    "TAC in the directory corresponds to an ROI to be "
                                                                    "analyzed.")
    grp_io.add_argument("-o", "--output-directory", required=True, help="Path to the output directory.")
    grp_io.add_argument("-p", "--output-filename-prefix", required=True,
                        help="Prefix for the output filenames. Typically 'sub-xxxx_ses-xx'")
    return grp_io


def add_common_analysis_args(parser: argparse.ArgumentParser):
    r"""Adds common analysis-related arguments to the provided argument parser.

    This helper function configures arguments controlling model selection, parameter
    initialization, parameter bounds, maximum number of fitting iterations and
    resampling behavior.

    Args:
        parser (argparse.ArgumentParser): Argument parser (or subparser) to which the
            analysis arguments will be added.
    """
    grp_analysis = parser.add_argument_group('Analysis Parameters')
    grp_analysis.add_argument("-m", "--model", required=True,
                              choices=['1tcm', '2tcm-k4zero', 'serial-2tcm', '2tcm'],
                              help="Analysis model to fit.")
    grp_analysis.add_argument("-g", "--initial-guesses", required=False, nargs='+', type=float,
                              help="Initial guesses for each fitting parameter.")
    grp_analysis.add_argument("-l", "--lower-bounds", required=False, nargs='+', type=float,
                              help="Lower bounds for each fitting parameter.")
    # grp_analysis.add_argument("-w", "--weighting-decay-constant", required=False, type=float, default=None,
    #                           help="Decay constant for computing per-frame weighting for fits.")
    grp_analysis.add_argument("-u", "--upper-bounds", required=False, nargs='+', type=float,
                              help="Upper bounds for each fitting parameter.")
    grp_analysis.add_argument("-f", "--max-fit-iterations", required=False, default=2500, type=int,
                              help="Maximum number of function iterations")
    grp_analysis.add_argument("-n", "--resample-num", required=False, default=4096, type=int,
                              help="Number of samples for uniform linear interpolation of provided TACs.")


def add_common_print_args(parser: argparse.ArgumentParser):
    r"""Adds common verbosity/printing arguments to the provided argument parser.

    This helper function configures arguments related to optional printing of analysis
    results to the console.

    Args:
        parser (argparse.ArgumentParser): Argument parser (or subparser) to which the
            printing/verbosity arguments will be added.
    """
    grp_verbose = parser.add_argument_group('Additional Options')
    grp_verbose.add_argument("--print", action="store_true", help="Whether to print the analysis results.")


def _generate_args() -> argparse.Namespace:
    r"""Generates and handles the arguments for the command-line interface.

    This function sets up the argument parser, adds required and optional arguments, and parses input arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Raises:
        argparse.ArgumentError: If necessary arguments are missing or invalid arguments are provided.
    """
    parser = argparse.ArgumentParser(prog='petpal-tcm-fit',
                                     description='Command line interface for fitting Tissue Compartment Models (TCM) '
                                                 'to PET Time Activity Curves (TACs).',
                                     formatter_class=argparse.RawTextHelpFormatter, epilog=_EXAMPLE_)

    subparsers = parser.add_subparsers(dest='strategy', description='Strategy for fitting TACs.',
                                       help='Strategy for fitting TACs.')

    frame_avgd_parser = subparsers.add_parser('frame_avgd', help='Perform analysis on properly frame averaged '
                                                                 'TACs using scan information. Usually more stable to '
                                                                 'noise, and the preferred strategy for non-simulated TACs')
    interpolated_parser = subparsers.add_parser('interp', help='Perform analysis on interpolated TACs '
                                                               'without using scan information.')

    sub_parser_list = [frame_avgd_parser, interpolated_parser]

    for a_parser in sub_parser_list:
        match a_parser.prog.split(' ')[-1]:
            case 'interp':
                grp_io = add_common_io_args(a_parser)
                grp_io.add_argument("-t", "--input-fitting-threshold-in-mins", required=False, type=float, default=30.0,
                                    help="Threshold in minutes for fitting the later half of the input function.")
                grp_io.add_argument("-b", "--ignore-blood-volume", required=False,
                                    default=False, action='store_true',
                                    help="Whether to ignore any blood volume contributions while fitting.")
            case 'frame_avgd':
                grp_io = add_common_io_args(a_parser)
                grp_io.add_argument('-s', '--scan-metadata-path', required=True, type=str,
                                    help="Path to the scan metadata file. Can also be the path to the .nii.gz file"
                                         "if the metadata shares the scan name: `*.nii.gz` -> `*json`.")
            case _:
                pass
        add_common_analysis_args(a_parser)
        add_common_print_args(a_parser)

    return parser.parse_args()


def _generate_bounds(initial: Union[list, None],
                     lower: Union[list, None],
                     upper: Union[list, None]) -> Union[np.ndarray, None]:
    r"""Generates the bounds for the fitting parameters.

    This function takes lists of initial fitting parameters, lower bounds, and upper bounds.
    All lists must have the same length. If no initial parameters are provided, the function returns None.

    Args:
        initial (list, optional): List of initial guesses for fitting parameters. If None, no bounds are generated.
        lower (list, optional): List of lower bounds for fitting parameters.
        upper (list, optional): List of upper bounds for fitting parameters.

    Returns:
        (np.ndarray, optional): If initial is not None, then we return a numpy array of shape [n, 3],
        where n is the number of parameters, where column 0 has the initial guesses, column 1 has lower bounds, and
        column 2 has upper bounds. If initial is None, function will return None.

    Raises:
        ValueError: If initial is not None and the length of initial, lower, and upper are not the same.
    """
    if initial is not None:
        if (len(initial) != len(lower)) or (len(initial) != len(upper)) or (len(upper) != len(lower)):
            raise ValueError("The number of initial guesses, lower bounds and upper bounds must be the same.")
        return np.asarray(np.asarray([initial, lower, upper]).T)
    else:
        return None


def fit_props_printer(fit_props: dict, segment: str | None = None) -> None:
    r"""Nicely formats and prints fit parameter estimates and uncertainties.

    This function prints a table of fitted parameter values, standard errors and
    percent errors. If a segment label is provided, it is printed as a header
    before the table, which is useful when printing results for multiple ROIs.

    Args:
        fit_props (dict): Dictionary containing the fit results. It is expected to
            contain a ``"FitProperties"`` entry with nested ``"FitValues"`` and
            ``"FitStdErr"`` mappings of parameter names to numerical values.
        segment (str, optional): Optional label for the TAC segment (for example,
            ROI name). If provided, the label is printed above the table.
    """
    title_str = f"{'Param':<5} {'FitVal':<6}    {'StdErr':<8} ({'%Err':>6})|"
    title_len = len(title_str)
    if segment :
        print(f'{"Segment: " + segment:#^{title_len}}')
    print("-" * title_len)
    print(title_str)
    print("-" * title_len)
    vals = fit_props["FitProperties"]["FitValues"]
    errs = fit_props["FitProperties"]["FitStdErr"]
    for param_name in vals:
        val = vals[param_name]
        err = errs[param_name]
        percent_err = f"{err / val * 100:>5.2f}%" if val > 0 else "NAN"
        print(f"{param_name:<5} {val:<6.4f} +- {err:<8.4f} ({percent_err:>6})|")

    print("-" * title_len)

def main():
    args = _generate_args()
    
    bounds = _generate_bounds(initial=args.initial_guesses, lower=args.lower_bounds, upper=args.upper_bounds)

    if args.strategy is None:
        args.print_help()
        raise ValueError("No fitting strategy specified!")

    common_kwargs = dict(input_tac_path=args.input_tac_path,
                         roi_tac_path=args.roi_tac_path,
                         roi_tacs_dir=args.roi_tac_path,
                         output_directory=args.output_directory,
                         output_filename_prefix=args.output_filename_prefix,
                         compartment_model=args.model,
                         parameter_bounds=bounds,
                         weights=None,
                         resample_num=args.resample_num, )

    is_single_tac = os.path.isfile(args.roi_tac_path)
    common_kwargs.pop('roi_tacs_dir') if is_single_tac else common_kwargs.pop('roi_tac_path')


    if args.strategy == 'frame_avgd':
        strategy_kwargs = common_kwargs | dict(scan_info_path=args.scan_metadata_path)
        AnalysisClass = pet_fit.FrameAveragedTCMAnalysis if is_single_tac else pet_fit.FrameAveragedMultiTACTCMAnalysis
    else:
        strategy_kwargs = common_kwargs | dict(aif_fit_thresh_in_mins=args.input_fitting_threshold_in_mins,
                                               max_func_iters=args.max_fit_iterations,
                                               ignore_blood_volume=args.ignore_blood_volume)
        AnalysisClass = pet_fit.TCMAnalysis if is_single_tac else pet_fit.MultiTACTCMAnalysis

    os.makedirs(args.output_directory, exist_ok=True)
    tac_fitting = AnalysisClass(**strategy_kwargs)
    tac_fitting.run_analysis()
    tac_fitting.save_analysis()
    
    if args.print:
        if os.path.isfile(args.roi_tac_path):
            fit_props_printer(fit_props=tac_fitting.analysis_props)
        else:
            for seg_name, seg_props in zip(tac_fitting.inferred_seg_labels, tac_fitting.analysis_props):
                fit_props_printer(fit_props=seg_props, segment=seg_name)


if __name__ == "__main__":
    main()
