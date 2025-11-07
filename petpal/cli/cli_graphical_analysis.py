"""
Command-line interface (CLI) for conducting graphical analysis of PET Time-Activity Curves (TACs).

This module provides a CLI to interact with the graphical_analysis module. It leverages argparse to handle command-line
arguments.

The user must provide:
    * Input TAC file path
    * Region of Interest (ROI) TAC file path
    * Threshold in minutes (below which data points will not be considered for fitting)
    * The method name for conducting the analysis. Supported methods are 'patlak', 'logan', or 'alt-logan'.
    * Output directory where the analysis results will be saved

An optional filename prefix for the output files can also be supplied.

This script utilizes the :class:`petpal.kinetic_modeling.graphical_analysis.GraphicalAnalysis` class to perform the graphical analysis
and save the results accordingly.

Example:
    .. code-block:: bash
    
        petpal-graph-analysis --input-tac-path /path/to/input.tac --roi-tac-path /path/to/roi.tac --threshold-in-mins 30.0 --method-name patlak --output-directory ./analysis --output-filename-prefix analysis

See Also:
    :mod:`petpal.kinetic_modeling.graphical_analysis` - module responsible for conducting and saving graphical analysis of PET TACs.

"""

import argparse
from ..kinetic_modeling import graphical_analysis as pet_ga


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-i", "--input-tac-path", required=True, help="Path to the input TAC file.")
    parser.add_argument("-o", "--output-directory", required=True, help="Path to the output directory.")
    parser.add_argument("-p", "--output-filename-prefix", required=True, help="Prefix for the output filenames.")
    parser.add_argument("-t", "--threshold-in-mins", required=True, type=float,
                           help="Threshold in minutes for the analysis.")
    parser.add_argument("-m", "--method-name", required=True, choices=['patlak', 'logan', 'alt-logan', 'logan-ref'],
                           help="Analysis method to be used.")
    parser.add_argument("-k","--k2-prime",required=False,help="k2-prime value used for logan-ref only",type=float)
    parser.add_argument("--print", action="store_true", help="Whether to print the analysis results.",default=False)


def main():
    parser = argparse.ArgumentParser(prog="Graphical Analysis", description="Perform graphical analysis on TAC data.",
                                     epilog="Example: petpal-graph-analysis "
                                            "graphical-analysis "
                                            "--input-tac-path /path/to/input.tac "
                                            "--pet4D-img-path /path/to/pet4D.img "
                                            "--output-directory /path/to/output "
                                            "--output-filename-prefix graph_ana "
                                            "--method-name patlak --threshold-in-mins 30.0 ")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help.")
    parser_single_roi = subparsers.add_parser('graphical-analysis')
    _add_common_args(parser_single_roi)
    parser_single_roi.add_argument("-r", "--roi-tac-path", required=True, help="Path to the ROI TAC file.")

    parser_multitac = subparsers.add_parser('graphical-analysis-multitac')
    _add_common_args(parser_multitac)
    parser_multitac.add_argument("-r", "--roi-tacs-dir", required=True, help="Path to directory containing ROI TTACs")
    parser_multitac.add_argument("-x","--excel", action='store_true',help='Set to output an excel-compatible table in a single file.',default=False)

    args = parser.parse_args()
    command = str(args.command).replace('-','_')

    if args.command is None:
        parser.print_help()
        raise SystemExit('Exiting without command')

    run_kwargs = {}
    if args.k2_prime is not None:
        run_kwargs['k2_prime'] = args.k2_prime

    method = args.method_name.replace('-','_')

    if command=='graphical_analysis':
        graphical_analysis = pet_ga.GraphicalAnalysis(input_tac_path=args.input_tac_path,
                                                    roi_tac_path=args.roi_tac_path,
                                                    output_directory=args.output_directory,
                                                    output_filename_prefix=args.output_filename_prefix,
                                                    method=method,
                                                    fit_thresh_in_mins=args.threshold_in_mins)
        graphical_analysis.run_analysis(**run_kwargs)
        graphical_analysis.save_analysis()
    if command=='graphical_analysis_multitac':
        graphical_analysis = pet_ga.MultiTACGraphicalAnalysis(input_tac_path=args.input_tac_path,
                                                              roi_tacs_dir=args.roi_tacs_dir,
                                                              output_directory=args.output_directory,
                                                              output_filename_prefix=args.output_filename_prefix,
                                                              method=method,
                                                              fit_thresh_in_mins=args.threshold_in_mins)
        graphical_analysis(one_file_per_region=not args.excel, **run_kwargs)

    if args.print:
        for key, val in graphical_analysis.analysis_props.items():
            print(f"{key:<20}:  {val}")


if __name__ == "__main__":
    main()
