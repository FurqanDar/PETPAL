"""
Command-line interface (CLI) for generating PET parametric images using graphical analysis of Time-Activity Curves (TACs).

This module provides a CLI to work with the parametric_images module. It uses argparse to handle command-line arguments.

The user must provide:
    * Input TAC file path
    * Path to the 4D PET image file
    * Threshold in minutes (below which data points will be not be used for fitting)
    * The method name for generating the images. Supported methods are 'patlak', 'logan', or 'alt-logan'.
    * Output directory where the parametric images will be saved

An optional filename prefix for the output files can also be supplied.

This script uses the :class:'petpal.parametric_images.GraphicalAnalysisParametricImage' class to 
calculate and save the images.

Example:
    .. code-block:: bash
    
         petpal-parametric-image --input-tac-path /path/to/input.tac --input-image-path /path/to/pet4D.img --threshold-in-mins 30.0 --method-name patlak --output-directory ./images --output-filename-prefix image

See Also:
    :mod:`petpal.parametric_images` - module for initiating and saving the graphical analysis of PET parametric images.

"""

import argparse
from ..kinetic_modeling.parametric_images import (GraphicalAnalysisParametricImage,
                                                  ReferenceTissueParametricImage)


def main():
    """
    Parametric image command line interface
    """
    parser = argparse.ArgumentParser(prog="Parametric Images With Graphical Analyses",
                                     description="Generate parametric images using graphical or "
                                                 "reference tissue methods on PET data.",
                                     epilog="Example usage: petpal-parametric-image "
                                            "--input-tac-path /path/to/input.tac "
                                            "--input-image-path /path/to/image4D.pet "
                                            "--output-directory /path/to/output"
                                            " --output-filename-prefix param_image "
                                            "--method-name patlak --threshold-in-mins 30.0")
    subparsers = parser.add_subparsers(dest="command",
                                       help="Generate parametric image with graphical method.")

    parser_graphical = subparsers.add_parser("graphical-analysis",help="Parametric image with "
                                             "graphical methods, e.g. Logan or Patlak")
    grp_io = parser_graphical.add_argument_group('I/O Paths')
    grp_io.add_argument("-i", "--input-tac-path", required=True,
                        help="Path to the input Time-Activity Curve (TAC) file.")
    grp_io.add_argument("-p", "--input-image-path", required=True,
                        help="Path to the 4D PET image file.")
    grp_io.add_argument("-o", "--output-directory", required=True,
                        help="Directory where the output parametric images will be saved.")
    grp_io.add_argument("-f", "--output-filename-prefix", default="",
                        help="Optional prefix for the output filenames.")


    grp_params = parser_graphical.add_argument_group('Method Parameters')
    grp_params.add_argument("-t", "--threshold-in-mins", required=True, type=float,
                            help="Threshold in minutes below which data points will be discarded.")
    grp_params.add_argument("-m", "--method-name", required=True,
                            choices=['patlak', 'logan', 'alt_logan', 'logan_ref'],
                            help="Name of the method for generating the plot.")
    grp_params.add_argument("-k", "--k2-prime", required=False, type=float, default=None,
                            help="k2_prime in minutes for Logan reference plot.")


    parser_reference = subparsers.add_parser("reference-tissue",help="Parametric image with "
                                             "reference tissue (RTM) methods, e.g. SRTM or MRTM")
    grp_io = parser_reference.add_argument_group('I/O Paths')
    grp_io.add_argument("-i", "--reference-tac-path", required=True,
                        help="Path to the reference region Time-Activity Curve (TAC) file.")
    grp_io.add_argument("-p", "--input-image-path", required=True,
                        help="Path to the 4D PET image file.")
    grp_io.add_argument("--mask-img-path", required=True,
                        help="Path to the mask of 4D PET image.")
    grp_io.add_argument("-o", "--output-directory", required=True,
                        help="Directory where the output parametric images will be saved.")
    grp_io.add_argument("-f", "--output-filename-prefix", default="",
                        help="Optional prefix for the output filenames.")


    grp_params = parser_reference.add_argument_group('Method Parameters')
    grp_params.add_argument("-t", "--threshold-in-mins", required=False, type=float,default=None,
                            help="Threshold in minutes below which data points will be discarded.")
    grp_params.add_argument("-m", "--method-name", required=True,
                            help="Name of the RTM method for kinetic modeling.")
    grp_params.add_argument("-b", "--bounds",required=False,nargs='+',default=None,type=float,
                            help="Fit parameter bounds.")
    grp_params.add_argument("-k", "--k2-prime",required=False,default=None,type=float,
                            help="Set k2_prime for RTM2 type methods.")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        raise SystemExit('Exiting without command')

    if args.command=='graphical-analysis':
        run_kwargs = {}
        if args.k2_prime is not None:
            run_kwargs['k2_prime'] = args.k2_prime

        param_img = GraphicalAnalysisParametricImage(input_tac_path=args.input_tac_path,
                                                    input_image_path=args.input_image_path,
                                                    output_directory=args.output_directory,
                                                    output_filename_prefix=args.output_filename_prefix)
        param_img.run_analysis(method_name=args.method_name,
                               t_thresh_in_mins=args.threshold_in_mins,
                               **run_kwargs)
        param_img.save_analysis()

    if args.command=='reference-tissue':
        param_img = ReferenceTissueParametricImage(reference_tac_path=args.reference_tac_path,
                                                   pet_image_path=args.input_image_path,
                                                   mask_image_path=args.mask_img_path,
                                                   method=args.method_name,
                                                   output_directory=args.output_directory,
                                                   output_filename_prefix=args.output_filename_prefix)
        param_img.run_parametric_analysis(bounds=args.bounds,
                                          k2_prime=args.k2_prime,
                                          t_thresh_in_mins=args.threshold_in_mins)
        param_img.save_parametric_images()
        param_img.save_analysis_properties()

if __name__ == "__main__":
    main()
