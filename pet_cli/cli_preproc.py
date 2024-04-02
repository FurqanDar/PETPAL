"""
This module provides a Command-line interface (CLI) for preprocessing imaging data to
produce regional PET Time-Activity Curves (TACs) and prepare data for parametric imaging analysis.

The user must provide:
    * Path to PET input data in NIfTI format. This can be source data, or with some preprocessing
      such as registration or motion correction, depending on the chosen operation.
    * Directory to which the output is written.
    * The name of the subject being processed, for the purpose of naming output files.
    * 3D imaging data, such as anatomical, segmentation, or PET sum, depending on the desired
      preprocessing operation.
    * Additional information needed for preprocessing, such as color table or half-life.
    * The operation to be performed on input data. Options: 'weighted_sum', 'motion_correct',
      'register', or 'write_tacs'.

Examples:
    * Half-life Weighted Sum:
    
        .. code-block:: bash
    
            pet-cli-preproc --operation weighted_sum --pet /path/to/pet.nii --out_dir /path/to/output
    
    * Image Registration:
    
        .. code-block:: bash
    
            pet-cli-preproc --operation register --pet /path/to/pet.nii --anatomical /path/to/mri.nii --pet_reference /path/to/pet_sum.nii --out_dir /path/to/output
            
    * Motion Correction:
    
        .. code-block:: bash
            
            pet-cli-preproc --operation motion_correct --pet /path/to/pet.nii --pet-reference /path/to/sum.nii --out_dir /path/to/output
            
    * Extracting TACs Using A Mask And Color-Table:
    
        .. code-block:: bash
            
            pet-cli-preproc --operation write_tacs --pet /path/to/pet.nii --segmentation /path/to/seg_masks.nii --color_table_path /path/to/color_table.json --out_dir /path/to/output

See Also:
    * :mod:`pet_cli.image_operations_4d` - module used to preprocess PET imaging data.

"""
import os
import argparse
from . import image_operations_4d


def _generate_image_path_and_directory(main_dir, ops_dir_name, file_prefix, ops_desc) -> str:
    """
    Generates the full path of an image file based on given parameters and creates the necessary directories.

    This function takes in four arguments: the main directory (main_dir), the operations directory (ops_dir),
    the subject ID (sub_id), and the operations extension (ops_ext). It joins these to generate the full path
    for an image file. The generated directories are created if they do not already exist.

    Args:
        main_dir (str): The main directory path.
        ops_dir_name (str): The operations (ops) directory. This is a directory inside `main_dir`.
        file_prefix (str): The prefix for the file name. Usually sub-XXXX if following BIDS.
        ops_desc (str): The operations (ops) extension to append to the filename.

    Returns:
        str: The full path of the image file with '.nii.gz' extension.

    Side Effects:
        Creates directories denoted by `main_dir`/`ops_dir_name` if they do not exist.

    Example:
        
        .. code-block:: python
        
            _generate_image_path_and_directory('/home/images', 'ops', '123', 'preprocessed')
            # '/home/images/ops/123_desc-preprocessed.nii.gz'
            # Directories '/home/images/ops' are created if they do not exist.
            
    """
    image_dir = os.path.join(os.path.abspath(main_dir), ops_dir_name)
    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(f'{image_dir}', f'{file_prefix}_desc-{ops_desc}.nii.gz')
    return str(image_path)


def _generate_args() -> argparse.Namespace:
    """
    Generates command line arguments for method :func:`main`.

    Returns:
        args (argparse.Namespace): Arguments used in the command line and their corresponding values.
    """
    parser = argparse.ArgumentParser(prog='PET Preprocessing',
                                     description='Command line interface for running PET preprocessing steps.',
                                     epilog='Example: pet-cli-preproc '
                                            '--pet /path/to/pet.nii '
                                            '--anatomical /path/to/mri.nii '
                                            '--pet_reference /path/to/pet_sum.nii '
                                            '--out_dir /path/to/output '
                                            '--operation register')
    io_grp = parser.add_argument_group('I/O')
    io_grp.add_argument('--pet', required=True, help='Path to PET series')
    io_grp.add_argument('--anatomical', required=False, help='Path to 3D anatomical image')
    io_grp.add_argument('--segmentation', required=False, help='Path to segmentation image\
        in anatomical space')
    io_grp.add_argument('--pet_reference', required=False, help='Path to reference image\
        for motion correction, if not weighted_sum.')
    io_grp.add_argument('--color_table_path', required=False, help='Path to color table')
    io_grp.add_argument('--half_life', required=False, help='Half life of radioisotope in seconds.', type=float)
    io_grp.add_argument('--out_dir', required=True, help='Directory to write results to')
    io_grp.add_argument('--subject_id', required=False, help='Subject ID to name files with', default='sub')
    
    ops_grp = parser.add_argument_group('Operations')
    ops_grp.add_argument('--operation', required=True, help='Preprocessing operation to perform',
                         choices=['weighted_sum', 'motion_correct', 'register', 'write_tacs'])
    
    verb_group = parser.add_argument_group('Additional information')
    verb_group.add_argument('-v', '--verbose', action='store_true',
                            help='Print processing information during computation.', required=False)
    
    args = parser.parse_args()
    return args


def _check_ref(args) -> str:
    """
    Check if pet_reference is being used and set variable
    """
    if args.pet_reference is not None:
        ref_image = args.pet_reference
    else:
        ref_image = _generate_image_path_and_directory(main_dir=args.out_dir,
                                                       ops_dir_name='sum_image',
                                                       file_prefix=args.subject_id,
                                                       ops_desc='sum')
    return ref_image


def main():
    """
    Preprocessing command line interface
    """
    args = _generate_args()
    
    if args.operation == 'weighted_sum':
        image_write = _generate_image_path_and_directory(main_dir=args.out_dir,
                                                         ops_dir_name='sum_image',
                                                         file_prefix=args.subject_id,
                                                         ops_desc='sum')
        image_operations_4d.weighted_series_sum(input_image_4d_path=args.pet,
                                                out_image_path=image_write,
                                                half_life=args.half_life,
                                                verbose=args.verbose)
    
    if args.operation == 'motion_correct':
        image_write = _generate_image_path_and_directory(main_dir=args.out_dir,
                                                         ops_dir_name='motion-correction',
                                                         file_prefix=args.subject_id,
                                                         ops_desc='moco')
        ref_image = _check_ref(args=args)
        image_operations_4d.motion_correction(input_image_4d_path=args.pet,
                                              reference_image_path=ref_image,
                                              out_image_path=image_write,
                                              verbose=args.verbose)
    
    if args.operation == 'register':
        image_write = _generate_image_path_and_directory(main_dir=args.out_dir,
                                                         ops_dir_name='registration',
                                                         file_prefix=args.subject_id,
                                                         ops_desc='reg')
        ref_image = _check_ref(args=args)
        image_operations_4d.register_pet(input_calc_image_path=ref_image,
                                         input_reg_image_path=args.pet,
                                         reference_image_path=args.anatomical,
                                         out_image_path=image_write,
                                         verbose=args.verbose)
    
    if args.operation == 'write_tacs':
        tac_write = os.path.join(args.out_dir, 'tacs')
        os.makedirs(tac_write, exist_ok=True)
        image_write = _generate_image_path_and_directory(main_dir=args.out_dir,
                                                         ops_dir_name='segmentation',
                                                         file_prefix=args.subject_id,
                                                         ops_desc='seg')
        image_operations_4d.resample_segmentation(input_image_4d_path=args.pet,
                                                  segmentation_image_path=args.segmentation,
                                                  out_seg_path=image_write,
                                                  verbose=args.verbose)
        
        image_operations_4d.write_tacs(input_image_4d_path=args.pet,
                                       color_table_path=args.color_table_path,
                                       segmentation_image_path=image_write,
                                       out_tac_path=tac_write,
                                       verbose=args.verbose)


if __name__ == "__main__":
    main()
