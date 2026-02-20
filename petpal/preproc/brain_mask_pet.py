"""Tools to apply brain mask to PET images.
"""
import ants
from .motion_target import determine_motion_target
from ..utils.dimension import timeseries_from_img_list
from ..utils.image_io import safe_copy_meta

def brain_mask_pet(input_image_path: str,
                   out_image_path: str | None,
                   atlas_image_path: str,
                   atlas_mask_path: str,
                   motion_target_option='weighted_series_sum') -> ants.ANTsImage:
    """
    Apply brain mask to dynamic PET image.
    
    Create target PET image, which is then warped to a
    provided anatomical atlas. The transformation to atlas space is then applied to transform a
    provided mask in atlas space into PET space. Mask is applied to input dynamic PET, optionally
    saved to out_image_path, and returned.

    Args:
        input_image_path (str): Path to input 4D PET image.
        out_image_path (str): Path to which brain mask in PET space is written.
        atlas_image_path (str): Path to anatomical atlas image.
        atlas_mask_path (str): Path to brain mask in atlas space.
        motion_target_option: Used to determine 3D target in PET space. Default
            'weighted_series_sum'.

    Returns:
        pet_masked_image (ants.ANTsImage): Dynamic PET image masked to brain only.

    Note:
        Requires access to an anatomical atlas or scan with a corresponding brain mask on said
        anatomical data. FSL users can use the MNI152 atlas and mask available at 
        $FSLDIR/data/standard/.
    """
    atlas = ants.image_read(atlas_image_path)
    atlas_mask = ants.image_read(atlas_mask_path)
    motion_target = determine_motion_target(motion_target_option=motion_target_option,
                                            input_image_path=input_image_path)
    pet_ref = ants.image_read(motion_target)
    xfm = ants.registration(
        fixed=atlas,
        moving=pet_ref,
        type_of_transform='SyN'
    )
    mask_on_pet = ants.apply_transforms(
        fixed=pet_ref,
        moving=atlas_mask,
        transformlist=xfm['invtransforms'],
        interpolator='nearestNeighbor'
    )
    mask_img = mask_on_pet.get_mask()

    pet_img = ants.image_read(filename=input_image_path)
    pet_img_list = ants.ndimage_to_list(pet_img)
    pet_masked_img_list = []
    for frame in pet_img_list:
        pet_masked_img_list.append(ants.mask_image(image=frame, mask=mask_img))
    pet_masked_img = timeseries_from_img_list(image_list=pet_masked_img_list)

    if out_image_path is not None:
        safe_copy_meta(input_image_path=input_image_path, out_image_path=out_image_path)
        ants.image_write(image=pet_masked_img,filename=out_image_path)

    return pet_masked_img
