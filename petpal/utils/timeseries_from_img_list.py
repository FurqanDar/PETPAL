"""Convert image list to timeseries image."""
import ants
import numpy as np



def gen_nd_image_based_on_image_list(image_list: list[ants.ANTsImage]) -> ants.ANTsImage:
    r"""
    Generate a 4D ANTsImage based on a list of 3D ANTsImages.

    This function takes a list of 3D ANTsImages and constructs a new 4D ANTsImage,
    where the additional dimension represents the number of frames (3D images) in the list.
    The 4D image retains the spacing, origin, direction, and shape properties of the 3D images,
    with appropriate modifications for the additional dimension.

    Args:
        image_list (list[ants.core.ants_image.ANTsImage]):
            List of 3D ANTsImage objects to be combined into a 4D image.
            The list must contain at least one image, and all images must have the same
            dimensions and properties.

    Returns:
        ants.ANTsImage:
            A 4D ANTsImage constructed from the input list of 3D images. The additional
            dimension corresponds to the number of frames (length of the image list).

    Raises:
        AssertionError: If the `image_list` is empty or if the images in the list are not 3D.

    See Also
        * :func:`petpal.preproc.motion_corr.motion_corr_frame_list_to_t1`

    Example:

        .. code-block:: python


            import ants
            image1 = ants.image_read('frame1.nii.gz')
            image2 = ants.image_read('frame2.nii.gz')
            image_list = [image1, image2]
            result = _gen_nd_image_based_on_image_list(image_list)
            print(result.dimension)  # 4
            image4d = ants.list_to_ndimage(result, image_list)

    """
    assert len(image_list) > 0
    assert image_list[0].dimension == 3

    num_frames = len(image_list)
    spacing_3d = image_list[0].spacing
    origin_3d = image_list[0].origin
    shape_3d = image_list[0].shape
    direction_3d = image_list[0].direction

    direction_4d = np.eye(4)
    direction_4d[:3, :3] = direction_3d
    spacing_4d = (*spacing_3d, 1.0)
    origin_4d = (*origin_3d, 0.0)
    shape_4d = (*shape_3d, num_frames)

    tmp_image = ants.make_image(imagesize=shape_4d,
                                spacing=spacing_4d,
                                origin=origin_4d,
                                direction=direction_4d)
    return tmp_image

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
