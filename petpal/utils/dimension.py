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


def gen_3d_img_from_timeseries(input_img: ants.ANTsImage) -> ants.ANTsImage:
    """
    Get the first frame of a 4D image as a template 3D image with voxel value zero.

    A simplified version of :py:func:`ants.ndimage_to_list.ndimage_to_list`.

    Args:
        input_img (ants.ANTsImage): The 4D image from which to get the template image.

    Returns:
        img_3d (ants.ANTsImage): The 3D template of the input image as an ants image.
    """
    dimension = input_img.dimension
    subdimension = dimension - 1
    suborigin = ants.get_origin( input_img )[0:subdimension]
    subspacing = ants.get_spacing( input_img )[0:subdimension]
    subdirection = np.eye( subdimension )
    for i in range( subdimension ):
        subdirection[i,:] = ants.get_direction( input_img )[i,0:subdimension]
    img_shape = input_img.shape[:-1]
    img_3d = ants.make_image(img_shape)
    ants.set_spacing( img_3d, subspacing )
    ants.set_origin( img_3d, suborigin )
    ants.set_direction( img_3d, subdirection )

    return img_3d


def get_frame_from_timeseries(input_img: ants.ANTsImage, frame: int) -> ants.ANTsImage:
    """
    Get a single frame of a 4D image as a 3D image.

    A simplified version of :py:func:`ants.ndimage_to_list.ndimage_to_list`.

    Args:
        input_img (ants.ANTsImage): The 4D image from which to get the frame.
        frame (int): The index of the frame to extract from the time series image.

    Returns:
        img_3d (ants.ANTsImage): The 3D first frame of the input image as an ants image.
    """
    dimension = input_img.dimension
    subdimension = dimension - 1
    suborigin = ants.get_origin( input_img )[0:subdimension]
    subspacing = ants.get_spacing( input_img )[0:subdimension]
    subdirection = np.eye( subdimension )
    for i in range( subdimension ):
        subdirection[i,:] = ants.get_direction( input_img )[i,0:subdimension]
    img_3d = ants.slice_image( input_img, axis = subdimension, idx = frame )
    ants.set_spacing( img_3d, subspacing )
    ants.set_origin( img_3d, suborigin )
    ants.set_direction( img_3d, subdirection )

    return img_3d


def check_physical_space_for_ants_image_pair(image_1: ants.core.ANTsImage,
                                             image_2: ants.core.ANTsImage,
                                             tolerance: float=1e-2) -> bool:
    """
    Determines whether two ANTs images share the same physical space. This function works
    when comparing 4D-images with 3D-images, as opposed to
    :func:`ants.image_physical_space_consistency`.

    This function validates whether the direction matrices, spacing values, and origins
    of the two provided ANTs images are consistent, ensuring they reside in the same
    physical space.

    Args:
        image_1 (ants.core.ANTsImage): The first ANTs image for comparison.
        image_2 (ants.core.ANTsImage): The second ANTs image for comparison.
        tolerance (float): Absolute tolerance for differences between components of the affine
            transform matrix for the two images. Default 0.01.

    Returns:
        bool: `True` if both images share the same physical space, `False` otherwise.

    """


    dir_cons = np.allclose(image_1.direction[:3,:3], image_2.direction[:3,:3],atol=tolerance)
    spc_cons = np.allclose(image_1.spacing[:3], image_2.spacing[:3],atol=tolerance)
    org_cons = np.allclose(image_1.origin[:3], image_2.origin[:3],atol=tolerance)

    return dir_cons and spc_cons and org_cons
