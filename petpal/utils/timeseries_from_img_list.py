"""Convert image list to timeseries image."""
import ants

from .useful_functions import gen_nd_image_based_on_image_list

def timeseries_from_img_list(image_list: list[ants.core.ANTsImage]) -> ants.core.ANTsImage:
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
