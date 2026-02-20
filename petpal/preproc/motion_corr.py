"""
Provides methods to motion correct 4D PET data. Includes method
:meth:`determine_motion_target`, which produces a flexible target based on the
4D input data to optimize contrast when computing motion correction or
registration.
"""
from typing import Optional
from warnings import warn
import ants
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .motion_target import determine_motion_target
from ..utils.scan_timing import (ScanTimingInfo,
                                 get_window_index_pairs_from_durations)
from ..utils.useful_functions import (weighted_series_sum_over_window_indices,
                                      coerce_outpath_extension)
from ..utils.dimension import timeseries_from_img_list
from ..utils.image_io import get_half_life_from_nifti, safe_copy_meta
from ..io.table import TableSaver
from ..io.image import ImageLoader


class MotionCorrect:
    """Run windowed motion correction on an image and save the result.
    
    :ivar image_loader: :func:`~petpal.io.image.ImageLoader` instance or injectable replacement
    :ivar table_saver: :func:`~petpal.io.table.TableSaver` instance or injectable replacement
    :ivar input_img: (ants.ANTsImage) Dynamic PET image
    :ivar target_img: (ants.ANTsImage) Static target image
    :ivar scan_timing: :func:`~petpal.utils.scan_timing.ScanTimingInfo` Dynamic PET scan timing.
    :ivar half_life: (float) Half life of the PET tracer in seconds.
    :ivar: reg_kwargs: (dict) Keyword arguments passed on to :py:func:`~ants.registration`"""
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

    def window_target_img(self, start_index: int, end_index: int) -> ants.ANTsImage:
        """Calculates the sum over frames in the target image within the provided time window.
        
        Args:
            start_index (int): Index for the frame that the window begins on.
            end_index (int): Index for the frame that the window ends on.
        
        Returns:
            window_img (ants.ANTsImage): Sum of frames in the input image between `start_index`
                and `end_index`."""
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

    def register_windows(self, window_duration: float=300) -> list[ants.ANTsTransform]:
        """Run motion correction on the input image to the target image.

        Creates "windows" by summing over frames with total length equal to `window_duration` and
        registering the window to the target image. Returns the calculated transforms for each
        frame.

        Args:
            window_duration (float): Duration of each window to sum over.

        Returns:
            window_xfm_stack (list[ants.ANTsTransform]): The transform to apply to each frame
                calculated based on the window the frame is in."""
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

        moco_img = timeseries_from_img_list(moco_img_stack)
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

    def __call__(self,
                 input_image_path: str,
                 output_image_path: str,
                 motion_target_option: str | tuple,
                 window_duration: float = 300,
                 transform_type: str = 'DenseRigid',
                 **reg_kwargs) -> ants.ANTsImage:
        """Motion correct a dynamic PET image.

        Divides image into segments of duration in seconds `window_duration` and register each frame
        to a target image, using the same transformation on for every frame in each window.

        Args:
            input_image_path (str): Path to dynamic PET image.
            output_image_path (str): Path to which motion corrected image is saved.
            motion_target_option (str | tuple): Path to motion target image, or specify time window
                such as (0,600) or preset option such as 'weighted_series_sum'. See
                :py:func:`~petpal.preproc.motion_target.determine_motion_target`.
            transform_type (str):  Type of transform used in ants.registration. See
                https://antspyx.readthedocs.io/en/latest/registration.html. Default DenseRigid.
            window_duration (float): Duration of each window in seconds. Default 300.
            reg_kwargs (keyword arguments): Keyword arguments to pass on to the registration
                function. See :py:func:`~ants.registration`.

        Returns:
            moco_img (ants.ANTsImage): Motion corrected dynamic PET image.
        """
        self.set_input_scan_properties(input_image_path=input_image_path)
        self.set_target_img(input_image_path=input_image_path,
                            motion_target_option=motion_target_option)

        self.set_reg_kwargs(type_of_transform=transform_type, **reg_kwargs)

        frame_xfms = self.register_windows(window_duration=window_duration)
        moco_img = self.apply_motion_correction(frame_xfms=frame_xfms)


        if 'Rigid' in self.reg_kwargs['type_of_transform']:
            self.save_xfm_parameters(frame_xfms=frame_xfms, filename=output_image_path)
        else:
            warn("Saving transform parameters is only available for rigid registrations. Current "
                 f" transform type: {self.reg_kwargs['type_of_transform']}.")

        ants.image_write(image=moco_img, filename=output_image_path)
        safe_copy_meta(input_image_path=input_image_path, out_image_path=output_image_path)

        return moco_img

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
    reg_kwargs_default = {'aff_metric'               : 'mattes',
                          'write_composite_transform': True,
                          'type_of_transform': type_of_transform,
                          'interpolator': interpolator}
    reg_kwargs = {**reg_kwargs_default, **kwargs}

    motion_corrector = MotionCorrect()
    moco_img = motion_corrector(input_image_path=input_image_path,
                                output_image_path=out_image_path,
                                motion_target_option=motion_target_option,
                                window_duration=window_duration,
                                copy_metadata=copy_metadata,
                                save_xfm=False
                                **reg_kwargs)
    return moco_img
