"""
Module to handle timing information of PET scans.
"""
from typing import Self
from dataclasses import dataclass
import numpy as np

from .image_io import load_metadata_for_nifti_with_same_filename


@dataclass
class ScanTimingInfo:
    """
    A data structure to represent and streamline access to timing information for image scans.

    This class encapsulates details about a scan's timing, including:
    - Start and end times of each scan frame.
    - Duration and center times of the frames.
    - Decay values (if applicable).

    Additionally, the class provides properties for easy conversion of timing values to minutes
    if the times are given in seconds and exceed a threshold (assumed to be 200.0 seconds).

    Attributes:
        duration (np.ndarray[float]): Array of frame durations.
        end (np.ndarray[float]): Array of frame end times.
        start (np.ndarray[float]): Array of frame start times.
        center (np.ndarray[float]): Array of frame center times (midpoints).
        decay (np.ndarray[float]): Array of decay coefficients for the scan frames.

    Properties:
        duration_in_mins (np.ndarray[float]):
            Returns the frame durations converted to minutes if `end` is >= 200.0 seconds.
            Otherwise, returns the original durations.

        end_in_mins (np.ndarray[float]):
            Returns the frame end times converted to minutes if `end` is >= 200.0 seconds.
            Otherwise, returns the original end times.

        start_in_mins (np.ndarray[float]):
            Returns the frame start times converted to minutes if `end` is >= 200.0 seconds.
            Otherwise, returns the original start times.

        center_in_mins (np.ndarray[float]):
            Returns the frame center times converted to minutes if `end` is >= 200.0 seconds.
            Otherwise, returns the original center times.

    Examples:

        .. code-block:: python

            import numpy as np
            from petpal.utils.scan_timing import ScanTimingInfo

            # Explicitly setting the attributes
            ## Define scan timing information
            duration = np.array([60.0, 120.0, 180.0])  # seconds
            start = np.array([0.0, 60.0, 180.0])
            end = np.array([60.0, 180.0, 360.0])
            center = (start + end) / 2.0  # Calculate the midpoints
            decay = np.array([1.0, 0.9, 0.8])  # Example decay values

            ## Create an instance of ScanTimingInfo
            scan_timing_info = ScanTimingInfo(duration=duration, end=end, start=start, center=center, decay=decay)

            ## Access original timing information
            print(scan_timing_info.duration)  # [ 60. 120. 180.]
            print(scan_timing_info.center)    # [30.  120. 270.]

            ## Access timing as minutes (when times exceed 200.0 seconds)
            print(scan_timing_info.duration_in_mins)  # [ 60. 120. 180.] (Unchanged)
            print(scan_timing_info.center_in_mins)    # [30. 120. 270.] (Unchanged)

            ## Example when `end` is greater than 200.0:
            scan_timing_info.end = np.array([300.0, 400.0, 500.0])  # Update end times
            print(scan_timing_info.end_in_mins)  # [5. 6.66666667 8.33333333] (Converted to minutes)
            print(scan_timing_info.start_in_mins)  # [0. 1. 3.] (Converted to minutes)

            # Getting the object directly from a nifti image file
            # assuming the metadata shares the name
            scan_timing_info = ScanTimingInfo.from_nifti("/path/to/image.nii.gz")

    """
    duration: np.ndarray[float]
    start: np.ndarray[float]
    end: np.ndarray[float]
    center: np.ndarray[float]
    decay: np.ndarray[float]

    @property
    def duration_in_mins(self) -> np.ndarray[float]:
        """
        Returns the frame durations in minutes. Validates values by checking if the final frame
        value is greater than 200: if so, then assumes values are in seconds and divides by 60.
        """
        if self.end[-1] >= 200.0:
            return self.duration / 60.0
        return self.duration

    @property
    def end_in_mins(self) -> np.ndarray[float]:
        """
        Returns the frame time ends in minutes. Validates values by checking if the final frame
        value is greater than 200: if so, then assumes values are in seconds and divides by 60.
        """
        if self.end[-1] >= 200.0:
            return self.end / 60.0
        return self.end

    @property
    def start_in_mins(self) -> np.ndarray[float]:
        """
        Returns the frame time starts in minutes. Validates values by checking if the final frame
        value is greater than 200: if so, then assumes values are in seconds and divides by 60.
        """
        if self.end[-1] >= 200.0:
            return self.start / 60.0
        return self.start

    @property
    def center_in_mins(self) -> np.ndarray[float]:
        """
        Returns the frame reference times in minutes. Validates values by checking if the final
        frame value is greater than 200: if so, then assumes values are in seconds and divides by
        60.
        """
        if self.end[-1] >= 200.0:
            return self.center / 60.0
        return self.center


    @classmethod
    def from_metadata(cls, metadata_dict: dict) -> Self:
        r"""
        Extracts frame timing information and decay factors from a json metadata.
        Expects that the JSON metadata has ``FrameDuration`` and ``DecayFactor`` or
        ``DecayCorrectionFactor`` keys.

        .. important::
            This function tries to infer `FrameTimesEnd` and `FrameTimesStart` from the frame
            durations if those keys are not present in the metadata file. If the scan is broken,
            this might generate incorrect results.


        Args:
            metadata_dict (dict): The metadata dictionary, loaded into memory.

        Returns:
            :class:`ScanTimingInfo`: Frame timing information with the following elements:
                - duration (np.ndarray): Frame durations in seconds.
                - start (np.ndarray): Frame start times in seconds.
                - end (np.ndarray): Frame end times in seconds.
                - center (np.ndarray): Frame center times in seconds.
                - decay (np.ndarray): Decay factors for each frame.
        """
        frm_dur = np.asarray(metadata_dict['FrameDuration'], float)
        try:
            frm_starts = np.asarray(metadata_dict['FrameTimesStart'], float)
        except KeyError:
            frm_starts = np.cumsum(frm_dur)-frm_dur
        try:
            frm_ends = np.asarray(metadata_dict['FrameTimesEnd'], float)
        except KeyError:
            frm_ends = frm_starts+frm_dur
        try:
            decay = np.asarray(metadata_dict['DecayCorrectionFactor'], float)
        except KeyError:
            decay = np.asarray(metadata_dict['DecayFactor'], float)
        try:
            frm_centers = np.asarray(metadata_dict['FrameReferenceTime'], float)
        except KeyError:
            frm_centers = np.asarray(frm_starts + frm_dur / 2.0, float)

        return cls(duration=frm_dur,
                   start=frm_starts,
                   end=frm_ends,
                   center=frm_centers,
                   decay=decay)

    @classmethod
    def from_nifti(cls, image_path: str) -> Self:
        r"""
        Extracts frame timing information and decay factors from a NIfTI image metadata.
        Expects that the JSON metadata file has ``FrameDuration`` and ``DecayFactor`` or
        ``DecayCorrectionFactor`` keys.

        .. important::
            This function tries to infer `FrameTimesEnd` and `FrameTimesStart` from the frame 
            durations if those keys are not present in the metadata file. If the scan is broken,
            this might generate incorrect results.


        Args:
            image_path (str): Path to the NIfTI image file.

        Returns:
           :class:`ScanTimingInfo`: Frame timing information with the following elements:
                - duration (np.ndarray): Frame durations in seconds.
                - start (np.ndarray): Frame start times in seconds.
                - end (np.ndarray): Frame end times in seconds.
                - center (np.ndarray): Frame center times in seconds.
                - decay (np.ndarray): Decay factors for each frame.
        """
        _meta_data = load_metadata_for_nifti_with_same_filename(image_path=image_path)
        return cls.from_metadata(metadata_dict=_meta_data)

    @classmethod
    def from_start_end(cls,
                       frame_starts: np.ndarray,
                       frame_ends: np.ndarray,
                       decay_correction_factor: np.ndarray | None=None) -> Self:
        """Infer timing properties based on start and end time.
        
        Args:
            frame_starts (np.ndarray): Start time of each frame.
            frame_ends (np.ndarray): End time of each frame.
            decay_correction_factor (np.ndarray | None): Decay correction factor, which can be
                optionally provided based on the type of analysis being done. If None, frame decay
                will be set to ones. Default None.

        Returns:
            scan_timing_info (ScanTimingInfo): ScanTimingInfo object with the correct start, end,
                duration, midpoint, and (optionally) decay correction for each frame.

        Raises:
            ValueError: If frame_starts, frame_ends, and decay_correction_factor (if provided) are
                not of identical shape.

        """
        if frame_starts.shape != frame_ends.shape:
            raise ValueError("frame_ends must have the same shape as frame_starts")

        frame_duration = frame_ends - frame_starts
        frame_midpoint = frame_starts + frame_duration / 2
        frame_decay = np.ones_like(frame_starts)

        if decay_correction_factor is None:
            frame_decay = np.ones_like(frame_starts, dtype=float)
        else:
            frame_decay = np.asarray(decay_correction_factor, dtype=float)
            if frame_decay.shape != frame_starts.shape:
                raise ValueError("decay_correction_factor must have the same shape as frame_starts")

        return cls(duration=frame_duration,
                   start=frame_starts,
                   end=frame_ends,
                   center=frame_midpoint,
                   decay=frame_decay)


def get_window_index_pairs_from_durations(frame_durations: np.ndarray,
                                          window_duration: float) -> np.ndarray:
    r"""
    Computes start and end index pairs for windows of a given size based on frame durations.

    Args:
        frame_durations (np.ndarray): Array of frame durations in seconds.
        window_duration (float): Window size in seconds.

    Returns:
        np.ndarray: Array of shape (2, N), where the first row contains start indices,
            and the second row contains end indices for each window.

    Raises:
        ValueError: If `window_duration` is less than or equal to 0.
        ValueError: If `window_duration` is greater than the total duration of all frames.
    """
    if window_duration <= 0:
        raise ValueError("Window duration has to be > 0")
    if window_duration > np.sum(frame_durations):
        raise ValueError("Window duration is longer than the whole scan.")
    window_edge_indices = [0]
    window_duration_rolling_sum = 0
    for frame_index, frame_dur in enumerate(frame_durations):
        window_duration_rolling_sum += frame_dur
        if window_duration_rolling_sum >= window_duration:
            window_edge_indices.append(frame_index + 1)
            window_duration_rolling_sum = 0
    if window_edge_indices[-1]!=len(frame_durations):
        window_edge_indices.append(len(frame_durations))
    window_start_indices = np.asarray(window_edge_indices[:-1])
    window_end_indices = np.asarray(window_edge_indices[1:])
    index_pairs = np.vstack((window_start_indices, window_end_indices))
    return index_pairs


def get_window_index_pairs_for_image(image_path: str, window_duration: float):
    """
    Computes start and end index pairs for windows of a given size
    based on the frame durations of a NIfTI image.

    Args:
        image_path (str): Path to the NIfTI image file.
        window_duration (float): Window size in seconds.

    Returns:
        np.ndarray: Array of shape (2, N), where the first row contains start indices,
            and the second row contains end indices for each window.

    Raises:
        ValueError: If `window_duration` is less than or equal to 0.
        ValueError: If `window_duration` is greater than the total duration of all frames.

    See Also:
        :func:`get_window_index_pairs_from_durations`
    """
    image_frame_info = ScanTimingInfo.from_nifti(image_path=image_path)
    return get_window_index_pairs_from_durations(frame_durations=image_frame_info.duration,
                                                 window_duration=window_duration)


def calculate_frame_reference_time(frame_duration: np.ndarray,
                                   frame_start: np.ndarray,
                                   half_life: float) -> np.ndarray:
    r"""Compute frame reference time as the time at which the average activity occurs.
    
    Equation comes from the `DICOM standard documentation
    <https://dicom.innolitics.com/ciods/positron-emission-tomography-image/pet-image/00541300>`_

    :math:`T_{ave}=\frac{1}{\lambda}ln\frac{\lambda T}{1-e^{-\lambda T}}`

    where lambda is the decay constant, :math:`\frac{ln2}{T_{1/2}}`, :math:`T_{1/2}` is the half
    life, and :math:`T` is the frame duration.

    Args:
        frame_duration (np.ndarray): Duration of each frame in seconds.
        frame_start (np.ndarray): Start time of each frame relative to scan start, in seconds.
        half_life (float): Radionuclide half life in seconds.

    Returns: 
        np.ndarray: Frame reference time for each frame in the scan in seconds.
    """
    decay_constant = np.log(2)/half_life
    decay_over_frame = decay_constant*frame_duration
    reference_time_delay = np.log((decay_over_frame)/(1-np.exp(-decay_over_frame)))/decay_constant
    frame_reference_time = frame_start + reference_time_delay
    return frame_reference_time
