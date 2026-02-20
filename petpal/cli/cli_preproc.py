"""
This module provides a Command-line interface (CLI) for preprocessing imaging data to
produce regional PET Time-Activity Curves (TACs) and prepare data for parametric imaging analysis.

The user must provide:
    * The sub-command. Options: 'weighted-sum', 'motion-correct', 'register-pet', or 'write-tacs'.
    * Path to PET input data in NIfTI format. This can be source data, or with some preprocessing
      such as registration or motion correction, depending on the chosen operation.
    * Directory to which the output is written.
    * The name of the subject being processed, for the purpose of naming output files.
    * 3D imaging data, such as anatomical, segmentation, or PET sum, depending on the desired
      preprocessing operation.
    * Additional information needed for preprocessing, such as color table or half-life.
    

Examples:

   * Auto Crop:

   .. code-block:: bash

       petpal-preproc auto-crop -i /path/to/input_img.nii.gz -o petpal_crop.nii.gz -t 0.05


   * Windowed Motion Correction:

   .. code-block:: bash

       petpal-preproc motion-correct -i /path/to/input_img.nii.gz -o petpal_moco.nii.gz --window-size 120 --transform-type QuickRigid


   * Register to anatomical:

   .. code-block:: bash

       petpal-preproc register-pet -i /path/to/input_img.nii.gz -o petpal_reg.nii.gz --motion-target 0 600 --anatomical /path/to/anat.nii.gz


    * Write regional tacs:

   .. code-block:: bash

    petpal-preproc write-tacs -i /path/to/input_img.nii.gz -p sub-001 -o /tmp/petpal_tacs -s /path/to/segmentation.nii.gz -l perlcyno -x


   * Write regional tacs, deprecated:

   .. code-block:: bash

       petpal-preproc write-tacs -i /path/to/input_img.nii.gz -o /tmp/petpal_tacs --segmentation /path/to/segmentation.nii.gz --label-map-path /path/to/dseg.tsv


   * Half life weighted sum of series:

   .. code-block:: bash

       petpal-preproc weighted-series-sum -i /path/to/input_img.nii.gz -o petpal_wss.nii.gz --start-time 1800 --end-time 7200


   * SUVR Image:

   .. code-block:: bash

       petpal-preproc suvr -i /path/to/input_img.nii.gz -o petpal_suvr.nii.gz --segmentation /path/to/segmentation.nii.gz --ref-region 1


   * Gaussian blur image:

   .. code-block:: bash

       petpal-preproc gauss-blur -i /path/to/input_img.nii.gz -o petpal_blur.nii.gz --blur-size-mm 8


   * Divide image by scale factor:

   .. code-block:: bash

       petpal-preproc rescale-image -i /path/to/input_img.nii.gz -o petpal_rescale.nii.gz --scale-factor 1000


   * Warp image to atlas:

   .. code-block:: bash

       petpal-preproc warp-pet-atlas -i /path/to/input_img.nii.gz -o petpal_reg-atlas.nii.gz --anatomical /path/to/anat.nii.gz --reference-atlas /path/to/atlas.nii.gz

    * Crop segmentation to PET FOV:

    .. code-block:: bash

        petpal-preproc seg-crop -i /path/to/input_img.nii.gz -o petpal_cropped_seg.nii.gz --segmentation /path/to/segmentation.nii.gz

See Also:
    * :mod:`~petpal.preproc.image_operations_4d` - module used for operations on 4D images.
    * :mod:`~petpal.preproc.motion_corr` - module for motion correction tools.
    * :mod:`~petpal.preproc.register` - module for MRI and atlas registration.

"""
import argparse
import ants


from ..preproc import (image_operations_4d,
                       motion_corr,
                       register,
                       regional_tac_extraction,
                       standard_uptake_value,
                       segmentation_tools,
                       brain_mask_pet)


_PREPROC_EXAMPLES_ = r"""
Examples:
  - Auto crop:
    petpal-preproc auto-crop -i /path/to/input_img.nii.gz -o petpal_crop.nii.gz -t 0.05
  - Windowed moco:
    petpal-preproc motion-correct -i /path/to/input_img.nii.gz -o petpal_moco.nii.gz --window-size 120 --transform-type QuickRigid
  - Register to anatomical:
    petpal-preproc register-pet -i /path/to/input_img.nii.gz -o petpal_reg.nii.gz --motion-target 0 600 --anatomical /path/to/anat.nii.gz
  - Write regional tacs:
    petpal-preproc write-tacs -i /path/to/input_img.nii.gz -p sub-001 -o /tmp/petpal_tacs -s /path/to/segmentation.nii.gz -l perlcyno -x
  - Write tacs, deprecated:
    petpal-preproc write-tacs-old -i /path/to/input_img.nii.gz -o /tmp/petpal_tacs --segmentation /path/to/segmentation.nii.gz --label-map-path /path/to/dseg.tsv
  - Half life weighted sum of series:
    petpal-preproc weighted-series-sum -i /path/to/input_img.nii.gz -o petpal_wss.nii.gz --start-time 1800 --end-time 7200
  - SUVR:
    petpal-preproc suvr -i /path/to/input_img.nii.gz -o petpal_suvr.nii.gz --segmentation /path/to/segmentation.nii.gz --ref-region 1
  - Gauss blur:
    petpal-preproc gauss-blur -i /path/to/input_img.nii.gz -o petpal_blur.nii.gz --blur-size-mm 8
  - Divide image by scale factor:s
    petpal-preproc rescale-image -i /path/to/input_img.nii.gz -o petpal_rescale.nii.gz --scale-factor 1000
  - Warp to atlas:
    petpal-preproc warp-pet-atlas -i /path/to/input_img.nii.gz -o petpal_reg-atlas.nii.gz --anatomical /path/to/anat.nii.gz --reference-atlas /path/to/atlas.nii.gz
  - SUV:
    petpal-preproc suv -i /path/to/input_img.nii.gz -o petpal_suv.nii.gz --weight 75 --dose 250 --start-time 1200 --end-time 3600
  - Crop segmentation image to PET FOV:
    petpal-preproc seg-crop -i /path/to/input_img.nii.gz -o petpal_cropped_seg.nii.gz --segmentation /path/to/segmentation.nii.gz
  - Add eroded white matter region to segmentation image:
    petpal-preproc eroded-wm -i /path/to/input_segmentation.nii.gz -o petpal_seg_with_eroded_wm.nii.gz
  - Apply brain mask to PET image:
    petpal-preproc brain-mask-pet -i /path/to/input_img.nii.gz -o petpal_brain_masked.nii.gz --atlas $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz --mask $FSLDIR/data/standard/MNI152_T1_1mm_brain_mask.nii.gz
"""


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Adds common arguments ('--input-img', '--out-img') to a provided ArgumentParser
    object.

    This function modifies the passed ArgumentParser object by adding two arguments commonly
    used for the command line scripts. It uses the add_argument method of the ArgumentParser class.
    After running this function, the parser will be able to accept and parse these additional
    arguments from the command line when run.
    
    .. note::
        This function modifies the passed `parser` object in-place and does not return anything.

    Args:
        parser (argparse.ArgumentParser): The argument parser object to which the arguments are
            added.

    Raises:
        argparse.ArgumentError: If a duplicate argument tries to be added.

    Side Effects:
        Modifies the ArgumentParser object by adding new arguments.

    Example:
        .. code-block:: python

            parser = argparse.ArgumentParser()
            _add_common_args(parser)
            args = parser.parse_args(['--pet', 'pet_file', '--out-dir', './', '--prefix', 'prefix'])
            print(args.pet)
            print(args.out_dir)
            print(args.prefix)

    """
    parser.add_argument('-o',
                        '--out-img',
                        default='petpal_preproc_output.nii.gz',
                        help='Output image filename')
    parser.add_argument('-i', '--input-img',required=True,help='Path to input image.',type=str)


def _generate_args() -> argparse.ArgumentParser:
    """
    Generates command line arguments for method :func:`main`.

    Returns:
        args (argparse.ArgumentParser): Arguments used in the command line and their corresponding
            values.
    """
    parser = argparse.ArgumentParser(prog='petpal-preproc',
                                     description="Command line interface for running PET "
                                                 "pre-processing steps.",
                                     epilog=_PREPROC_EXAMPLES_,
                                     formatter_class=argparse.RawTextHelpFormatter)

    subparsers = parser.add_subparsers(dest="command", help="Sub-command help.")

    parser_wss = subparsers.add_parser('weighted-series-sum',
                                       help='Half-life weighted sum of 4D PET series.')
    _add_common_args(parser_wss)
    parser_wss.add_argument('--start-time',
                            required=False,
                            help='Start time of sum in seconds.',
                            type=float,
                            default=0)
    parser_wss.add_argument('--end-time',
                            required=False,
                            help='End time of sum in seconds.',
                            type=float,
                            default=-1)

    parser_crop = subparsers.add_parser('auto-crop',
                                        help='Automatically crop 4D PET image using threshold.')
    _add_common_args(parser_crop)
    parser_crop.add_argument('-t','--thresh-val', required=True,default=0.01,
                            help='Fractional threshold to crop image projections.',type=float)

    parser_tac = subparsers.add_parser('write-tacs',
                                       help='Write ROI TACs from 4D PET using segmentation masks.')
    parser_tac.add_argument('-i', '--input-img',required=True,help='Path to input image.',type=str)
    parser_tac.add_argument('-p',
                            '--patid',
                            help='Name of participant, appended to beginning of TAC output files.',
                            required=True,
                            type=str)
    parser_tac.add_argument('-o',
                            '--out-tac-dir',
                            required=True,
                            help='Output TAC folder dir')
    parser_tac.add_argument('-s', '--segmentation', required=True,
                            help='Path to segmentation image in anatomical space.')
    parser_tac.add_argument('-l',
                            '--label-map',
                            required=True,
                            help='Label map for the seg image, either a preset option or path to a '
                                 'json file. E.g. freesurfer, freesurfer_merge_lr, perlcyno, '
                                 'perlcyno_merge_lr, /path/to/my_label_map.json.')
    parser_tac.add_argument('-x',
                            '--excel',
                            action='store_true',
                            required=False,
                            default=False,
                            help='Option to store results as a single file table instead of one '
                                 'TAC file per region.')

    parser_oldtac = subparsers.add_parser('write-tacs-old',
                                          help='DEPRECATED Write ROI TACs from 4D PET using '
                                               'segmentation masks. Uses `dseg.tsv`file as label '
                                               'map.')
    parser_oldtac.add_argument('-i',
                               '--input-img',
                               required=True,
                               help='Path to input image.',
                               type=str)
    parser_oldtac.add_argument('-o',
                            '--out-tac-dir',
                            default='petpal_tacs',
                            help='Output TAC folder dir')
    parser_oldtac.add_argument('-s', '--segmentation', required=True,
                            help='Path to segmentation image in anatomical space.')
    parser_oldtac.add_argument('-l',
                            '--label-map-path',
                            required=True,
                            help='Path to label map dseg.tsv')

    parser_warp = subparsers.add_parser('warp-pet-atlas',
                                        help='Perform nonlinear warp on PET to atlas.')
    _add_common_args(parser_warp)
    parser_warp.add_argument('-a',
                             '--anatomical',
                             required=True,
                             help='Path to 3D anatomical image (T1w or T2w).',
                            type=str)
    parser_warp.add_argument('-r',
                             '--reference-atlas',
                             required=True,
                             help='Path to anatomical atlas.',
                             type=str)

    parser_suvr = subparsers.add_parser('suvr',help='Compute SUVR on a parametric PET image.')
    _add_common_args(parser_suvr)
    parser_suvr.add_argument('-s', '--segmentation', required=True,
                            help='Path to segmentation image in anatomical space.')
    parser_suvr.add_argument('-r',
                             '--ref-region',
                             help='Reference region to normalize SUVR to. List multiple regions to '
                                  'combine as one reference.',
                             required=True,
                             type=int,
                             nargs='+')
    parser_suvr.add_argument('--start-time',
                            required=True,
                            help='Start time for SUVR calculation in seconds from scan start',
                            type=float)
    parser_suvr.add_argument('--end-time',
                            required=True,
                            help='End time for SUVR calculation in seconds from scan start',
                            type=float)

    parser_blur = subparsers.add_parser('gauss-blur',help='Perform 3D gaussian blurring.')
    _add_common_args(parser_blur)
    parser_blur.add_argument('-b',
                             '--blur-size-mm',
                             help='Size of gaussian kernal with which to blur image.',
                             type=float)

    parser_rescale = subparsers.add_parser('rescale-image',help='Divide an image by a scalar.')
    _add_common_args(parser_rescale)
    parser_rescale.add_argument('-r',
                                '--scale-factor',
                                help='Divides image by this number',
                                type=float,
                                required=True)


    parser_window_moco = subparsers.add_parser('motion-correct',
                                               help='Correct for inter-frame motion in dynamic PET '
                                                    'images')
    _add_common_args(parser_window_moco)
    parser_window_moco.add_argument('--motion-target', default=None, nargs='+',
                            help="Motion target option. Can be an image path, "
                                 "'weighted_series_sum' or a tuple "
                                 "(i.e. '--motion-target 0 600' for first ten minutes).",
                            required=True)
    parser_window_moco.add_argument('-w', '--window-size', default=60.0, type=float,
                                    help="Window size in seconds.",)
    xfm_types = ['QuickRigid', 'Rigid', 'DenseRigid', 'Affine', 'AffineFast']
    parser_window_moco.add_argument('-y', '--transform-type', default='QuickRigid', type=str,
                                    choices=xfm_types,
                                    help="Type of ANTs transformation to apply when registering.")

    parser_reg = subparsers.add_parser('register-pet',
                                       help='Register 4D PET to MRI anatomical space.')
    _add_common_args(parser_reg)
    parser_reg.add_argument('-a',
                            '--anatomical',
                            required=True,
                            help='Path to 3D anatomical image (T1w or T2w).',
                            type=str)
    parser_reg.add_argument('-t', '--motion-target', default=None, nargs='+',
                            help="Motion target option. Can be an image path, "
                                 "'weighted_series_sum' or a tuple (i.e. '-t 0 600' for first "
                                 "ten minutes).")

    parser_suv = subparsers.add_parser('suv',help='Standard Uptake Value (SUV) calculation')
    _add_common_args(parser_suv)
    parser_suv.add_argument('-w',
                            '--weight',
                            required=True,
                            help='Weight of the participant in kg',
                            type=float)
    parser_suv.add_argument('-d',
                            '--dose',
                            required=True,
                            help='Dose of radiotracer injected in MBq',
                            type=float)
    parser_suv.add_argument('--start-time',
                            required=True,
                            help='Start time for SUV calculation in seconds from scan start',
                            type=float)
    parser_suv.add_argument('--end-time',
                            required=True,
                            help='End time for SUV calculation in seconds from scan start',
                            type=float)

    parser_seg_crop = subparsers.add_parser('seg-crop',help='Crop segmentation image to PET FOV')
    _add_common_args(parser_seg_crop)
    parser_seg_crop.add_argument('-s',
                                 '--segmentation',
                                 required=True,
                                 help='Path to segmentation image',
                                 type=str)

    parser_eroded_wm = subparsers.add_parser('eroded-wm',
                                             help='Add eroded white matter region to segmentation '
                                                  'image')
    _add_common_args(parser_eroded_wm)

    parser_brain_mask = subparsers.add_parser('brain-mask-pet',
                                              help='Apply brain mask to dynamic PET image')
    _add_common_args(parser_brain_mask)
    parser_brain_mask.add_argument('--atlas',
                                   required=True,
                                   help='Path to atlas template image. Can be PET or anatomical '
                                         'based.',
                                   type=str)
    parser_brain_mask.add_argument('--mask',
                                   required=True,
                                   help='Path to brain mask in atlas space.',
                                   type=str)

    return parser


def main():
    """
    Preproc command line interface
    """
    preproc_parser = _generate_args()
    args = preproc_parser.parse_args()

    if args.command is None:
        preproc_parser.print_help()
        raise SystemExit('Exiting without command')

    if 'motion_target' in dir(args):
        if len(args.motion_target)==1:
            motion_target = args.motion_target[0]
        else:
            motion_target = args.motion_target
    else:
        motion_target = None

    command = str(args.command).replace('-','_')

    match command:
        case 'weighted_series_sum':
            standard_uptake_value.weighted_sum_for_suv(input_image_path=args.input_img,
                                                       output_image_path=args.out_img,
                                                       start_time=args.start_time,
                                                       end_time=args.end_time)
        case 'auto_crop':
            image_operations_4d.SimpleAutoImageCropper(input_image_path=args.input_img,
                                                    out_image_path=args.out_img,
                                                    thresh_val=args.thresh_val,
                                                    verbose=True)
        case 'register_pet':
            register.register_pet(input_reg_image_path=args.input_img,
                                out_image_path=args.out_img,
                                reference_image_path=args.anatomical,
                                motion_target_option=motion_target,
                                verbose=True)
        case 'write_tacs_old':
            regional_tac_extraction.write_tacs(input_image_path=args.input_img,
                                            out_tac_dir=args.out_tac_dir,
                                            segmentation_image_path=args.segmentation,
                                            label_map_path=args.label_map_path,
                                            verbose=True)
        case 'write_tacs':
            tac_obj = regional_tac_extraction.WriteRegionalTacs(input_image_path=args.input_img,
                                                    segmentation_path=args.segmentation,
                                                    label_map=args.label_map)
            tac_obj(out_tac_prefix=args.patid,
                    out_tac_dir=args.out_tac_dir,
                    one_tsv_per_region=not args.excel)
        case 'warp_pet_atlas':
            register.warp_pet_to_atlas(input_image_path=args.input_img,
                                       anat_image_path=args.anatomical,
                                       atlas_image_path=args.reference_atlas,
                                       out_image_path=args.out_img,
                                       verbose=True)
        case 'gauss_blur':
            image_operations_4d.gauss_blur(input_image_path=args.input_img,
                                        blur_size_mm=args.blur_size_mm,
                                        out_image_path=args.out_img,
                                        verbose=True,
                                        use_fwhm=True)
        case 'suvr':
            standard_uptake_value.suvr(input_image_path=args.input_img,
                                       output_image_path=args.out_img,
                                       segmentation_image_path=args.segmentation,
                                       ref_region=args.ref_region,
                                       start_time=args.start_time,
                                       end_time=args.end_time)
        case 'motion_correct':
            motion_corrector = motion_corr.MotionCorrect()
            motion_corrector(input_image_path=args.input_img,
                             output_image_path=args.out_img,
                             motion_target_option=motion_target,
                             window_duration=args.window_size)
        case 'rescale_image':
            input_img = ants.image_read(filename=args.input_img)
            out_img = image_operations_4d.rescale_image(input_image=input_img,
                                                        rescale_constant=args.scale_factor)
            ants.image_write(image=out_img, filename=args.out_img)
        case 'suv':
            standard_uptake_value.suv(input_image_path=args.input_img,
                                      output_image_path=args.out_img,
                                      start_time=args.start_time,
                                      end_time=args.end_time,
                                      weight=args.weight,
                                      dose=args.dose)
        case 'seg_crop':
            input_img = ants.image_read(filename=args.input_img)
            seg_img = ants.image_read(filename=args.segmentation)
            seg_cropped = segmentation_tools.seg_crop_to_pet_fov(pet_img=input_img,
                                                                 segmentation_img=seg_img)
            ants.image_write(seg_cropped,args.out_img)
        case 'eroded_wm':
            segmentation_tools.eroded_wm_segmentation(input_segmentation_path=args.input_img,
                                                      out_segmentation_path=args.out_img)
        case 'brain_mask_pet':
            brain_mask_pet.brain_mask_pet(input_image_path=args.input_img,
                                          out_image_path=args.out_img,
                                          atlas_image_path=args.atlas,
                                          atlas_mask_path=args.mask)

if __name__ == "__main__":
    main()
