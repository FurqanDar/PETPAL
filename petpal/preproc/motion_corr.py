"""
Provides methods to motion correct 4D PET data. Includes method
:meth:`determine_motion_target`, which produces a flexible target based on the
4D input data to optimize contrast when computing motion correction or
registration.
"""
from typing import Optional
import ants
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from petpal.utils.useful_functions import gen_nd_image_based_on_image_list
from .motion_target import determine_motion_target
from ..utils import image_io
from ..utils.scan_timing import (ScanTimingInfo,
                                 get_window_index_pairs_from_durations,
                                 get_window_index_pairs_for_image)
from ..utils.useful_functions import (weighted_series_sum_over_window_indices,
                                      coerce_outpath_extension)
from ..utils.image_io import get_half_life_from_nifti, safe_copy_meta
from ..io.table import TableSaver
from ..io.image import ImageLoader


def motion_corr(input_image_path: str,
                motion_target_option: str | tuple,
                out_image_path: str,
                verbose: bool,
                type_of_transform: str = 'DenseRigid',
                **kwargs) -> tuple[np.ndarray, list[str], list[float]]:
    """
    Correct PET image series for inter-frame motion. Runs rigid motion
    correction module from Advanced Normalisation Tools (ANTs) with default
    inputs.

    Args:
        input_image_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image to be motion corrected.
        motion_target_option (str | tuple): Target image for computing
            transformation. See :meth:`determine_motion_target`.
        out_image_path (str): Path to a .nii or .nii.gz file to which the
            motion corrected PET series is written.
        verbose (bool): Set to ``True`` to output processing information.
        type_of_transform (str): Type of transform to perform on the PET image,
            must be one of antspy's transformation types, i.e. 'DenseRigid' or
            'Translation'. Any transformation type that uses >6 degrees of
            freedom is not recommended, use with caution. See
            :py:func:`ants.registration`.
        kwargs (keyword arguments): Additional arguments passed to
            :py:func:`ants.motion_correction`.

    Returns:
        pet_moco_np (np.ndarray): Motion corrected PET image series as a numpy
            array.
        pet_moco_params (list[str]): List of ANTS registration files applied to
            each frame.
        pet_moco_fd (list[float]): List of framewise displacement measure
            corresponding to each frame transform.
    """
    pet_ants = ants.image_read(input_image_path)
    motion_target_image_path = determine_motion_target(motion_target_option=motion_target_option,
                                                       input_image_path=input_image_path)

    motion_target_image = ants.image_read(motion_target_image_path)
    pet_moco_ants_dict = ants.motion_correction(image=pet_ants,
                                                fixed=motion_target_image,
                                                type_of_transform=type_of_transform,
                                                **kwargs)
    if verbose:
        print('(ImageOps4D): motion correction finished.')

    pet_moco_ants = pet_moco_ants_dict['motion_corrected']
    pet_moco_params = pet_moco_ants_dict['motion_parameters']
    pet_moco_fd = pet_moco_ants_dict['FD']
    pet_moco_np = pet_moco_ants.numpy()
    ants.image_write(image=pet_moco_ants,filename=out_image_path)
    image_io.safe_copy_meta(input_image_path=input_image_path, out_image_path=out_image_path)

    if verbose:
        print(f"(ImageOps4d): motion corrected image saved to {out_image_path}")
    return pet_moco_np, pet_moco_params, pet_moco_fd


def motion_corr_frame_list(input_image_path: str,
                           motion_target_option: str | tuple,
                           out_image_path: str,
                           verbose: bool,
                           frames_list: list = None,
                           type_of_transform: str = 'Affine',
                           transform_metric: str = 'mattes',
                           **kwargs):
    r"""
    Perform per-frame motion correction on a 4D PET image.

    This function applies motion correction to each frame of a 4D PET image based on a specified
    motion target. Only the frames in ``frames_list`` are motion corrected, all else are kept as is.

    Args:
        input_image_path (str): Path to the input 4D PET image file.
        motion_target_option (str | tuple): Option to determine the motion target. This can
            be a path to a specific image file, a tuple of frame indices to generate a target, or
            specific options recognized by :func:`determine_motion_target`.
        out_image_path (str): Path to save the motion-corrected output image.
        verbose (bool): Whether to print verbose output during processing.
        frames_list (list, optional): List of frame indices to correct. If None, corrects all
            frames. Default is None.
        type_of_transform (str, optional): Type of transformation to use for registration. Default
            is 'Affine'.
        transform_metric (str, optional): Metric to use for the transformation. Default is
            'mattes'.
        **kwargs: Additional arguments passed to the `ants.registration` method.

    Returns:
        None

    Example:

        .. code-block:: python

            from petpal.preproc.motion_corr import motion_corr_frame_list

            motion_corr_frame_list(input_image_path='/path/to/image.nii.gz',
                                  motion_target_option='/path/to/target_image.nii.gz',
                                  out_image_path='/path/to/output_motion_corrected.nii.gz',
                                  verbose=True)

    Notes:
        - The :func:`determine_motion_target` function is used to derive the motion target image
            based on the specified option.
        - If `frames_list` is not provided, all frames of the 4D image will be corrected.
        - Motion correction is performed using the :py:func:`ants.registration` method from the
            ANTsPy library.
        - The corrected frames are reassembled into a 4D image and saved to the specified output
            path.

    """
    input_image = ants.image_read(input_image_path)

    motion_target_path = determine_motion_target(motion_target_option=motion_target_option,
                                                 input_image_path=input_image_path)
    motion_target = ants.image_read(motion_target_path)

    frames_to_correct = np.zeros(input_image.shape[-1], dtype=bool)

    if frames_list is None:
        _correct_these_frames = np.ones(input_image.shape[-1], dtype=int)
        frames_to_correct[list(_correct_these_frames)] = True
    else:
        assert max(frames_list) < input_image.shape[-1]
        frames_to_correct[list(frames_list)] = True

    out_image = []
    input_image_list = input_image.ndimage_to_list()

    if verbose:
        print("(Info): On frame:", end=' ')

    for frame_id, moco_this_frame in enumerate(frames_to_correct):
        if verbose:
            print(f"{frame_id:>02}", end=' ')
        this_frame = input_image_list[frame_id]
        if moco_this_frame:
            tmp_reg = ants.registration(fixed=motion_target,
                                        moving=this_frame,
                                        type_of_transform=type_of_transform,
                                        aff_metric=transform_metric,
                                        interpolator='linear',
                                        reg_iterations=(),
                                        **kwargs)
            out_image.append(tmp_reg['warpedmovout'])
        else:
            out_image.append(this_frame)

    if verbose:
        print("... done!\n")
    tmp_image = gen_nd_image_based_on_image_list(out_image)
    out_image = ants.list_to_ndimage(tmp_image, out_image)
    ants.image_write(image=out_image,filename=out_image_path)

    if verbose:
        print(f"(ImageOps4d): motion corrected image saved to {out_image_path}")


def motion_corr_frame_list_to_t1(input_image_path: str,
                                 t1_image_path: str,
                                 motion_target_option: str | tuple,
                                 out_image_path: str,
                                 verbose: bool,
                                 frames_list: list = None,
                                 type_of_transform: str = 'AffineFast',
                                 transform_metric: str = "mattes"):
    r"""
    Perform motion correction of a 4D PET image to a T1 anatomical image.

    This function corrects motion in a 4D PET image by registering it to a T1 anatomical
    image. The method uses a two-step process: first registering an intermediate motion
    target to the T1 image (either the time-averaged image or a weighted-series-sum), and
    then using the calculated transform to correct motion in individual frames of the PET series.
    The motion-target-option is registered to the T1 anatomical image. Then, given the frames in
    the frame list, the frames are registered to the T1 image, and all other frames are simply
    transformed to the motion-target in T1-space.

    Args:
        input_image_path (str): Path to the 4D PET image to be corrected.
        t1_image_path (str): Path to the 3D T1 anatomical image.
        motion_target_option (str | tuple): Option for selecting the motion target image.
            Can be a path to a file or a tuple range. If None, the average of the PET timeseries
            is used.
        out_image_path (str): Path to save the motion-corrected 4D image.
        verbose (bool): Set to True to print verbose output during processing.
        frames_list (list, optional): List of frame indices to correct. If None, all frames
            are corrected & registered.
        type_of_transform (str): Type of transformation used in registration. Default is
            'AffineFast'.
        transform_metric (str): Metric for transformation optimization. Default is 'mattes'.

    Returns:
        None

    Raises:
        AssertionError: If maximum frame index in `frames_list` exceeds the number of frames in the
            PET image.

    Example:

        .. code-block:: python


            motion_corr_frame_list_to_t1(input_image_path='pet_timeseries.nii.gz',
                              t1_image_path='t1_image.nii.gz',
                              motion_target_option='mean_image',
                              out_image_path='pet_corrected.nii.gz',
                              verbose=True)

    """

    input_image = ants.image_read(input_image_path)
    t1_image = ants.image_read(t1_image_path)

    motion_target_path = determine_motion_target(motion_target_option=motion_target_option,
                                                 input_image_path=input_image_path)
    motion_target = ants.image_read(motion_target_path)

    motion_target_to_mpr_reg = ants.registration(fixed=t1_image,
                                                 moving=motion_target,
                                                 type_of_transform=type_of_transform,
                                                 aff_metric=transform_metric, )

    motion_target_in_t1 = motion_target_to_mpr_reg['warpedmovout']
    motion_transform_matrix = motion_target_to_mpr_reg['fwdtransforms']

    frames_to_correct = np.zeros(input_image.shape[-1], dtype=bool)

    if frames_list is None:
        _correct_these_frames = np.ones(input_image.shape[-1], dtype=int)
        frames_to_correct[list(_correct_these_frames)] = True
    else:
        assert max(frames_list) < input_image.shape[-1]
        frames_to_correct[list(frames_list)] = True

    out_image = []
    input_image_list = input_image.ndimage_to_list()

    if verbose:
        print("(Info): On frame:", end=' ')

    for frame_id, moco_this_frame in enumerate(frames_to_correct):
        if verbose:
            print(f"{frame_id:>02}", end=' ')
        this_frame = input_image_list[frame_id]
        if moco_this_frame:
            tmp_reg = ants.registration(fixed=motion_target_in_t1,
                                        moving=this_frame,
                                        type_of_transform=type_of_transform,
                                        aff_metric=transform_metric,
                                        interpolator='linear')
            out_image.append(tmp_reg['warpedmovout'])
        else:
            tmp_transform = ants.apply_transforms(fixed=motion_target_in_t1,
                                                  moving=this_frame,
                                                  transformlist=motion_transform_matrix,
                                                  interpolator='linear')
            out_image.append(tmp_transform)

    if verbose:
        print("... done!\n")
    tmp_image = gen_nd_image_based_on_image_list(out_image)
    out_image = ants.list_to_ndimage(tmp_image, out_image)
    ants.image_write(image=out_image,filename=out_image_path)

    if verbose:
        print(f"(ImageOps4d): motion corrected image saved to {out_image_path}")


def motion_corr_frames_above_mean_value(input_image_path: str,
                                        out_image_path: str,
                                        motion_target_option: str | tuple,
                                        verbose: bool,
                                        type_of_transform: str = 'Affine',
                                        transform_metric: str = 'mattes',
                                        scale_factor=1.0,
                                        **kwargs):
    r"""
    Perform motion correction on frames with mean values above the mean of a 4D PET image.

    This function applies motion correction only to the frames in a 4D PET image whose mean voxel
    values are greater than the overall mean voxel value of the entire image. It internally
    utilizes the :func:`motion_corr_frame_list` function to perform the motion correction.

    Args:
        input_image_path (str): Path to the input 4D PET image file.
        motion_target_option (str | tuple): Option to determine the motion target. This can
            be a path to a specific image file, a tuple of frame indices to generate a target, or
            specific options recognized by :func:`determine_motion_target`.
        out_image_path (str): Path to save the motion-corrected output image.
        verbose (bool): Whether to print verbose output during processing.
        type_of_transform (str, optional): Type of transformation to use for registration.
            Default is 'Affine'.
        transform_metric (str, optional): Metric to use for the transformation. Default is
            'mattes'.
        scale_factor (float, optional): Scale factor to apply to frame mean values before
            comparison. Default is 1.0.
        **kwargs: Additional arguments passed to the `ants.registration` method.

    Returns:
        None

    Example:

        .. code-block:: python

            from petpal.preproc.motion_corr import motion_corr_frames_above_mean_value

            motion_corr_frames_above_mean_value(input_image_path='/path/to/image.nii.gz',
                                                motion_target_option='/path/to/target_image.nii.gz',
                                                out_image_path='/path/to/output_motion_corrected.nii.gz',
                                                verbose=True,
                                                type_of_transform='Affine',
                                                transform_metric='mattes',
                                                scale_factor=1.2)

    Notes:
        - Uses :func:`motion_corr_frame_list` for the actual motion correction of specified frames.
        - Frames with mean voxel values greater than the total mean voxel value (optionally scaled
            by `scale_factor`) are selected for motion correction.
        - The :func:`_get_list_of_frames_above_total_mean` function is used to
            identify the frames to be motion corrected based on their mean voxel values.

    """

    frames_list = _get_list_of_frames_above_total_mean(image_4d_path=input_image_path,
                                                       scale_factor=scale_factor)

    motion_corr_frame_list(input_image_path=input_image_path,
                           motion_target_option=motion_target_option,
                           out_image_path=out_image_path,
                           verbose=verbose,
                           frames_list=frames_list,
                           type_of_transform=type_of_transform,
                           transform_metric=transform_metric,
                           **kwargs)


def motion_corr_frames_above_mean_value_to_t1(input_image_path: str,
                                              t1_image_path: str,
                                              motion_target_option: str | tuple,
                                              out_image_path: str,
                                              verbose: bool,
                                              type_of_transform: str = 'AffineFast',
                                              transform_metric: str = "mattes",
                                              scale_factor: float = 1.0):
    """
    Perform motion correction on frames with mean values above the mean of a 4D PET image to a T1
    anatomical image.

    This function applies motion correction only to the frames in a 4D PET image whose mean voxel
    values are greater than the overall mean voxel value of the entire image. It corrects these
    frames by registering them to a T1 anatomical image, using the `motion_corr_frame_list_to_t1`
    function.

    Args:
        input_image_path (str): Path to the input 4D PET image file.
        t1_image_path (str): Path to the 3D T1 anatomical image.
        motion_target_option (str | tuple): Option to determine the motion target. This can
            be a path to a specific image file, a tuple of frame indices to generate a target, or
            specific options recognized by :func:`determine_motion_target`.
        out_image_path (str): Path to save the motion-corrected output image.
        verbose (bool): Whether to print verbose output during processing.
        type_of_transform (str, optional): Type of transformation to use for registration. Default
            is 'AffineFast'.
        transform_metric (str, optional): Metric to use for the transformation. Default is 'mattes'.
        scale_factor (float, optional): Scale factor applied to the mean voxel value of the entire
            image for comparison. Must be greater than 0. Default is 1.0.

    Returns:
        None

    Example:

        .. code-block:: python

            from petpal.preproc.motion_corr import motion_corr_frames_above_mean_value_to_t1

            motion_corr_frames_above_mean_value_to_t1(input_image_path='/path/to/image.nii.gz',
                                                      t1_image_path='/path/to/t1_image.nii.gz',
                                                      motion_target_option='/path/to/target_image.nii.gz',
                                                      out_image_path='/path/to/output_motion_corrected.nii.gz',
                                                      verbose=True,
                                                      type_of_transform='AffineFast',
                                                      transform_metric='mattes',
                                                      scale_factor=1.2)

    Notes:
        - This function internally uses :func:`motion_corr_frame_list_to_t1` for the actual motion
            correction of specified frames.
        - Frames with mean voxel values greater than the total mean voxel value (optionally scaled
            by `scale_factor`) are selected for motion correction.
        - The :func:`_get_list_of_frames_above_total_mean` function identifies
            the frames to be motion corrected based on their mean voxel values.
    """
    frames_list = _get_list_of_frames_above_total_mean(image_4d_path=input_image_path,
                                                       scale_factor=scale_factor)

    motion_corr_frame_list_to_t1(input_image_path=input_image_path,
                                 t1_image_path=t1_image_path,
                                 motion_target_option=motion_target_option,
                                 out_image_path=out_image_path,
                                 verbose=verbose,
                                 frames_list=frames_list,
                                 type_of_transform=type_of_transform,
                                 transform_metric=transform_metric)

class MotionCorrect:
    """Run windowed motion correction on an image and save the result"""
    def __init__(self,
                 image_loader: Optional[ImageLoader] = None,
                 table_saver: Optional[TableSaver] = None):
        self.image_loader = image_loader or ImageLoader()
        self.table_saver = table_saver or TableSaver()
        self.input_img = None
        self.target_img = None
        self.scan_timing = None
        self.half_life = None
        self.reg_kwargs = self.default_reg_kwargs

    @property
    def default_reg_kwargs(self) -> dict:
        """Default registration arguments passed on to :py:func:`~ants.registration`."""
        reg_kwargs_default = {'aff_metric'               : 'mattes',
                              'write_composite_transform': True,
                              'interpolator'             : 'linear',
                              'type_of_transform'        : 'DenseRigid'}
        return reg_kwargs_default

    def set_reg_kwargs(self, **reg_kwargs):
        """Modify the registration arguments passed on to :py:func:`~ants.registration`."""
        self.reg_kwargs.update(**reg_kwargs)

    def set_input_scan_properties(self, input_image_path: str):
        """Load input image and get half life and scan timing. Set as MotionCorrect attributes.
        
        Args:
            input_image_path (str): Path to dynamic PET image."""
        self.input_img = self.image_loader.load(filename=input_image_path)
        self.half_life = get_half_life_from_nifti(image_path=input_image_path)
        self.scan_timing = ScanTimingInfo.from_nifti(image_path=input_image_path)

    def set_target_img(self, input_image_path: str, motion_target_option: str | tuple):
        """Get the motion target, load it as an image, and set as an attribute.
        
        Args:
            input_image_path (str): Path to dynamic PET image.
            motion_target_option (str | tuple): Option for motion target. See
                :meth:`~petpal.preproc.motion_target.determine_motion_target.` for details."""
        motion_target_path = determine_motion_target(motion_target_option=motion_target_option,
                                                     input_image_path=input_image_path)
        self.target_img = self.image_loader.load(filename=motion_target_path)

    def window_index_pairs(self, window_duration: float=300) -> np.ndarray:
        """The pair of indices corresponding to each window in the image.
        
        Args:
            window_duration (float): Scan will be divided into windows of this duration in
                seconds.
            
        Returns:
            window_index_pairs (np.ndarray): Array of start and end frame indices for each
                window.
        
        See also:
            - :meth:`~petpal.utils.scan_timing.get_window_index_pairs_from_durations`
        """
        return get_window_index_pairs_from_durations(frame_durations=self.scan_timing.duration,
                                                     window_duration=window_duration)

    def window_target_img(self, start_index: int, end_index: int):
        """Calculates the sum over frames in the target image within the provided time window."""
        return weighted_series_sum_over_window_indices(input_image_4d=self.input_img,
                                                        output_image_path=None,
                                                        window_start_id=start_index,
                                                        window_end_id=end_index,
                                                        half_life=self.half_life,
                                                        image_frame_info=self.scan_timing)

    @staticmethod
    def ants_xfm_to_rigid_pars(ants_xfm: ants.ANTsTransform):
        """Convert an ants transform object to six parameters (3 translation, 3 rotation) and the
        center reference point."""
        xfm_in = np.reshape(ants_xfm.parameters,(4,3))
        rot_matrix = xfm_in[:3,:]
        translate_matrix = xfm_in[3,:]

        scipy_rotation = Rotation.from_matrix(rot_matrix)
        rot_pars = -scipy_rotation.as_euler('xyz',degrees=True)

        xfm_out = list(rot_pars)+list(translate_matrix)+list(ants_xfm.fixed_parameters)
        return xfm_out

    def register_windows(self, window_duration: float=300):
        """Run motion correction on the input image to the target image."""
        window_xfm_stack = []
        window_index_pairs = self.window_index_pairs(window_duration=window_duration)

        for _, (start_index, end_index) in enumerate(zip(*window_index_pairs)):
            window_target_img = self.window_target_img(start_index=start_index,
                                                       end_index=end_index)
            window_registration = ants.registration(fixed=self.target_img,
                                                    moving=window_target_img,
                                                    **self.reg_kwargs)
            window_xfm = ants.read_transform(window_registration['fwdtransforms'])
            for _ in range(start_index, end_index):
                window_xfm_stack.append(window_xfm)

        return window_xfm_stack

    def apply_motion_correction(self, frame_xfms: list[ants.ANTsTransform]) -> ants.ANTsImage:
        """Apply transforms to input image.
        
        Args:
            frame_xfms (list[ants.ANTsTransform]): Transforms for each frame in the input image.

        Returns:
            moco_img (ants.ANTsImage): Motion corrected dynamic PET image.
        """
        input_img_list = ants.ndimage_to_list(self.input_img)
        moco_img_stack = []

        for frame_index, frame_img in enumerate(input_img_list):
            frame_xfm = frame_xfms[frame_index]
            moco_frame_img = frame_xfm.apply_to_image(image=frame_img,
                                                      reference=self.target_img)
            moco_img_stack.append(moco_frame_img)

        moco_img = gen_timeseries_from_image_list(moco_img_stack)
        return moco_img

    def save_xfm_parameters(self, frame_xfms: list[ants.ANTsTransform], filename: str):
        """Save frame transform parameters as a table.

        Args:
            frame_xfms (np.ndarray): Rigid transform parameters ordered as rotation, translation,
                centerpoint, then X, Y, Z axis, totalling 9 parameters for each frame.
            filename (str): Path to where table will be saved, including extension.

        Raises:
            ValueError: If transform type does not containt 'Rigid'. Saving transform parameters is
                currently only available for rigid transforms."""
        frame_xfm_pars = [self.ants_xfm_to_rigid_pars(ants_xfm=xfm) for xfm in frame_xfms]

        if 'Rigid' not in self.reg_kwargs['type_of_transform']:
            raise ValueError("Saving transform parameters is only available for rigid "
                             "registrations. Current transform type: "
                             f"{self.reg_kwargs['type_of_transform']}")
        xfm_columns = ['rot_x',
                       'rot_y',
                       'rot_z',
                       'tra_x',
                       'tra_y',
                       'tra_z',
                       'cen_x',
                       'cen_y',
                       'cen_z']
        xfms_df = pd.DataFrame(data=frame_xfm_pars,
                               columns=xfm_columns)
        xfms_df.index.name = 'frame'
        csv_filename = coerce_outpath_extension(path=filename, ext='.csv')
        self.table_saver.save(xfms_df,csv_filename)

    def __call__(self, input_image_path: str,
                 output_image_path: str,
                 motion_target_option: str | tuple,
                 window_duration: float = 300,
                 copy_metadata: bool = True,
                 save_xfm: bool = True,
                 **reg_kwargs):
        """Motion correct a dynamic PET image.

        Divides image into segments of duration in seconds `window_duration` and register each frame
        to a target image, using the same transformation on for every frame in each window.

        Args:
            input_image_path (str): Path to dynamic PET image.
            output_image_path (str): Path to which motion corrected image is saved.
            motion_target_option (str | tuple): Path to motion target image, or specify time window
                such as (0,600) or preset option such as 'mean_image'. See
                :py:func:`~petpal.preproc.motion_target.determine_motion_target`.
            window_duration (float): Duration of each window in seconds. Default 300.
            copy_metadata (bool): Copies metadata info from input image to output image. Default
                True.
            save_xfm (bool): Saves motion correction transform parameters for translation,
                rotation, and rotation center point. Only compatible with rigid transforms. Default
                True.
            reg_kwargs (keyword arguments): Keyword arguments to pass on to the registration
                function. See :py:func:`~ants.registration`.
        """
        self.set_input_scan_properties(input_image_path=input_image_path)
        self.set_target_img(input_image_path=input_image_path,
                            motion_target_option=motion_target_option)

        self.set_reg_kwargs(**reg_kwargs)

        frame_xfms = self.register_windows(window_duration=window_duration)
        moco_img = self.apply_motion_correction(frame_xfms=frame_xfms)

        if save_xfm:
            self.save_xfm_parameters(frame_xfms=frame_xfms, filename=output_image_path)
        ants.image_write(image=moco_img, filename=output_image_path)
        if copy_metadata:
            safe_copy_meta(input_image_path=input_image_path, out_image_path=output_image_path)

        return moco_img

def gen_timeseries_from_image_list(image_list: list[ants.core.ANTsImage]) -> ants.core.ANTsImage:
    r"""
    Takes a list of ANTs ndimages, and generates a 4D ndimage. Undoes :func:`ants.ndimage_to_list`
    so that we take a list of 3D images and generates a 4D image.

    Args:
        image_list (list[ants.core.ANTsImage]): A list of ndimages.

    Returns:
        ants.core.ANTsImage: 4D ndimage.
    """
    tmp_image = gen_nd_image_based_on_image_list(image_list)
    return ants.list_to_ndimage(tmp_image, image_list)


def _get_list_of_frames_above_total_mean(image_4d_path: str,
                                         scale_factor: float = 1.0):
    """
    Get the frame indices where the frame mean is higher than the total mean of a 4D image.

    This function calculates the mean voxel value of each frame in a 4D image and returns the
        indices of the frames whose mean voxel value is greater than or equal to the mean voxel
        value of the entire image, optionally scaled by a provided factor.

    Args:
        image_4d_path (str): Path to the input 4D PET image file.
        scale_factor (float, optional): Scale factor applied to the mean voxel value of the entire
            image for comparison. Must be greater than 0. Default is 1.0.

    Returns:
        list: A list of frame indices where the frame mean voxel value is greater than or equal to
            the scaled total mean voxel value.

    Example:

        .. code-block:: python

            from petpal.preproc.motion_corr import _get_list_of_frames_above_total_mean

            frame_ids = _get_list_of_frames_above_total_mean(image_4d_path='/path/to/image.nii.gz',
                                                                                  scale_factor=1.2)

            print(frame_ids)  # Output: [0, 3, 5, ...]

    Notes:
        - The :func:`ants.image_read` from ANTsPy is used to read the 4D image into memory.
        - The mean voxel value of the entire image is scaled by `scale_factor` for comparison with
            individual frame means.
        - The function uses the :func:`ants.ndimage_to_list` method from ANTsPy to convert the 4D
            image into a list of 3D frames.

    """
    assert scale_factor > 0
    image = ants.image_read(image_4d_path)
    total_mean = scale_factor * image.mean()

    frames_list = []
    for frame_id, a_frame in enumerate(image.ndimage_to_list()):
        if a_frame.mean() >= total_mean:
            frames_list.append(frame_id)

    return frames_list


def windowed_motion_corr_to_target(input_image_path: str,
                                   out_image_path: str | None,
                                   motion_target_option: str | tuple,
                                   window_duration: float,
                                   type_of_transform: str = 'QuickRigid',
                                   interpolator: str = 'linear',
                                   copy_metadata: bool = True,
                                   **kwargs):
    """
    Performs windowed motion correction (MoCo) to align frames of a 4D PET image to a given target image.
    We compute a combined image over the frames in a window, which is registered to the target image.
    Then, for each frame within the window, the same transformation is applied. This can be useful for
    initial frames with low counts. By setting a small-enough window size, later frames can still be
    individually registered to the target image.

    .. important::
        The motion-target will determine the space of the output image. If we provide a T1 image
        as the `motion_target_option`, the output image will be in T1-space.

    Note:
        This function is deprecated. Use :py:func:`~petpal.preproc.motion_corr.MotionCorrect`
        instead.

    Args:
        input_image_path (str): Path to the input 4D PET image file.
        out_image_path (str | None): Path to save the resulting motion-corrected image. If
            None, don't save image to disk.
        motion_target_option (str | tuple): Option to determine the motion target. This can
            be a path to a specific image file, a tuple of frame indices to generate a target, or
            specific options recognized by :func:`determine_motion_target`.
        window_duration (float): Window size in seconds for dividing the image into time sections.
        type_of_transform (str): Type of transformation to use in registration (default: 'QuickRigid').
        interpolator (str): Interpolation method for the transformation (default: 'linear').
        **kwargs: Additional arguments passed to :func:`ants.registration`.

    Returns:
        ants.core.ANTsImage: Motion-corrected 4D image.

    Workflow:
        1. Reads the input 4D image and splits it into individual frames.
        2. Computes index windows based on the specified window size (`window_duration`).
        3. Extracts necessary frame timing information and the tracer's half-life.
        4. For each window:
            - Calculates a weighted sum image for the window.
              See :func:`petpal.utils.useful_functions.weighted_series_sum_over_window_indecies`.
            - Performs registration of the weighted sum image to the target image.
            - Applies the obtained transformations to each frame within the window.
        5. Combines the transformed frames into a corrected 4D image.
        6. Saves the output image to the specified path, if provided.

    Note:
        If `out_image_path` is provided, the corrected 4D image will be saved to the specified path.
    """
    input_image = ants.image_read(filename=input_image_path)
    input_image_list = ants.ndimage_to_list(input_image)
    window_idx_pairs = get_window_index_pairs_for_image(image_path=input_image_path, window_duration=window_duration)
    half_life = get_half_life_from_nifti(image_path=input_image_path)
    frame_timing_info = ScanTimingInfo.from_nifti(image_path=input_image_path)

    target_image = determine_motion_target(motion_target_option=motion_target_option,
                                           input_image_path=input_image_path)
    target_image = ants.image_read(target_image)

    reg_kwargs_default = {'aff_metric'               : 'mattes',
                          'write_composite_transform': True}
    reg_kwargs = {**reg_kwargs_default, **kwargs}

    out_image = []
    for _, (st_id, end_id) in enumerate(zip(*window_idx_pairs)):
        window_tgt_image = weighted_series_sum_over_window_indices(input_image_4d=input_image,
                                                                   output_image_path=None,
                                                                   window_start_id=st_id,
                                                                   window_end_id=end_id,
                                                                   half_life=half_life,
                                                                   image_frame_info=frame_timing_info)
        window_registration = ants.registration(fixed=target_image,
                                                moving=window_tgt_image,
                                                type_of_transform=type_of_transform,
                                                interpolator=interpolator,
                                                **reg_kwargs)
        for frm_id in range(st_id, end_id):
            out_image.append(ants.apply_transforms(fixed=target_image,
                                                   moving=input_image_list[frm_id],
                                                   transformlist=window_registration['fwdtransforms']))

    out_image = gen_timeseries_from_image_list(out_image)

    if out_image_path is not None:
        ants.image_write(image=out_image, filename=out_image_path)

    if copy_metadata:
        image_io.safe_copy_meta(input_image_path=input_image_path,
                                out_image_path=out_image_path)
    return out_image
