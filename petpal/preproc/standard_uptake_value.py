"""
Module for functions calculating standard uptake value (SUV) and related measures, such as standard
uptake value ratio (SUVR).
"""
import ants

from ..utils.dimension import gen_3d_img_from_timeseries

from ..utils.stats import mean_value_in_region
from ..utils.math_lib import weighted_sum_computation
from ..utils.useful_functions import nearest_frame_to_timepoint
from ..utils.image_io import (get_half_life_from_nifti,
                              load_metadata_for_nifti_with_same_filename,
                              safe_copy_meta)


def weighted_sum_for_suv(input_image_path: str,
                         output_image_path: str | None,
                         start_time: float=0,
                         end_time: float=-1) -> ants.ANTsImage:
    """Function that calculates the weighted series sum for a PET image specifically for
    calculating the standard uptake value (SUV) of the image.
    
    Args:
        input_image_path (str): Path to a 4D PET image which we calculate the sum on.
        output_image_path (str): Path to which output image is saved. If None, returns
            calculated image without saving.
        start_time: Time in seconds from the start of the scan from which to begin sum calculation.
            Only frames after selected time will be included in the sum. Default 0.
        end_time: Time in seconds from the start of the scan from which to end sum calculation.
            Only frames before selected time will be included in the sum. If -1, use all frames
            after `start_time` in the calculation. Default -1.
            
    Returns:
        weighted_sum_img (ants.ANTsImage): 3D image resulting from the sum calculation.
    """
    half_life = get_half_life_from_nifti(image_path=input_image_path)
    if half_life <= 0:
        raise ValueError('(ImageOps4d): Radioisotope half life is zero or negative.')
    pet_meta = load_metadata_for_nifti_with_same_filename(input_image_path)
    pet_img = ants.image_read(input_image_path)
    frame_start = pet_meta['FrameTimesStart']
    frame_duration = pet_meta['FrameDuration']

    if 'DecayCorrectionFactor' in pet_meta.keys():
        decay_correction = pet_meta['DecayCorrectionFactor']
    elif 'DecayFactor' in pet_meta.keys():
        decay_correction = pet_meta['DecayFactor']
    else:
        raise ValueError("Neither 'DecayCorrectionFactor' nor 'DecayFactor' exist in meta-data "
                         "file")

    last_frame_time = frame_start[-1]
    if end_time!=-1:
        last_frame_time = end_time
    scan_start = frame_start[0]
    nearest_frame = nearest_frame_to_timepoint(frame_times=frame_start)
    calc_first_frame = int(nearest_frame(start_time+scan_start))
    calc_last_frame = int(nearest_frame(last_frame_time+scan_start))
    if calc_first_frame==calc_last_frame:
        calc_last_frame += 1
    pet_series_adjusted = pet_img[:,:,:,calc_first_frame:calc_last_frame]
    frame_start_adjusted = frame_start[calc_first_frame:calc_last_frame]
    frame_duration_adjusted = frame_duration[calc_first_frame:calc_last_frame]
    decay_correction_adjusted = decay_correction[calc_first_frame:calc_last_frame]

    weighted_sum_arr = weighted_sum_computation(frame_duration=frame_duration_adjusted,
                                                half_life=half_life,
                                                pet_series=pet_series_adjusted,
                                                frame_start=frame_start_adjusted,
                                                decay_correction=decay_correction_adjusted)
    weighted_sum_img = ants.from_numpy_like(weighted_sum_arr,gen_3d_img_from_timeseries(pet_img))

    if output_image_path is not None:
        ants.image_write(weighted_sum_img, output_image_path)
        safe_copy_meta(input_image_path=input_image_path,
                       out_image_path=output_image_path)

    return weighted_sum_img


def suv(input_image_path: str,
        output_image_path: str | None,
        weight: float,
        dose: float,
        start_time: float,
        end_time: float) -> ants.ANTsImage:
    """Compute standard uptake value (SUV) over a pet image. Calculate the weighted image sum
    then divide by the dose and multiplying by the weight of the participant.

    Args:
        input_image_path: Path to input PET image.
        output_image_path: Path to which SUV image is saved.
        weight: Weight of the participant in kg.
        dose: Dose injected during the scan in MBq.
        start_time: Start time for the SUV calculation in seconds.
        end_time: End time for the SUV calculation in seconds.
        
    Returns:
        suv_img (ants.ANTsImage): The standard uptake value calculated on the input image."""
    wss_img = weighted_sum_for_suv(input_image_path=input_image_path,
                                   output_image_path=None,
                                   start_time=start_time,
                                   end_time=end_time)
    suv_img = wss_img / (dose*1000) * weight

    if output_image_path is not None:
        ants.image_write(suv_img, output_image_path)
        safe_copy_meta(input_image_path=input_image_path,
                       out_image_path=output_image_path)

    return suv_img


def suvr(input_image_path: str,
         output_image_path: str | None,
         segmentation_image_path: str,
         ref_region: int | list[int],
         start_time: float,
         end_time: float) -> ants.ANTsImage:
    """
    Computes an ``SUVR`` (Standard Uptake Value Ratio) by taking the average of
    an input image within a reference region, and dividing the input image by
    said average value.

    Args:
        input_image_path (str): Path to 4D PET image.
        output_image_path (str): Path to output image file which is written to. If None, no output
            is written.
        segmentation_image_path (str): Path to segmentation image, which we use
            to compute average uptake value in the reference region.
        ref_region (int): Region or list of region mappings over which to compute average SUV. If a
            list is provided, combines all regions in the list as one reference region.
        start_time: Time in seconds from the start of the scan from which to begin sum calculation.
            Only frames after selected time will be included in the sum. Default 0.
        end_time: Time in seconds from the start of the scan from which to end sum calculation.
            Only frames before selected time will be included in the sum. If -1, use all frames
            after `start_time` in the calculation. Default -1.

    Returns:
        ants.ANTsImage: SUVR parametric image
    """
    sum_img = weighted_sum_for_suv(input_image_path=input_image_path,
                                   output_image_path=None,
                                   start_time=start_time,
                                   end_time=end_time)
    segmentation_img = ants.image_read(filename=segmentation_image_path)

    ref_region_avg = mean_value_in_region(input_img=sum_img,
                                          seg_img=segmentation_img,
                                          mappings=ref_region)

    suvr_img = sum_img / ref_region_avg

    if output_image_path is not None:
        ants.image_write(image=suvr_img,
                         filename=output_image_path)
        safe_copy_meta(input_image_path=input_image_path,
                       out_image_path=output_image_path)

    return suvr_img
