"""
A collection of decorators to handle boilerplate code. Most decorators will
extend the functionality of functions that only work with objects or arrays.
The extensions allow for the flexibility of passing in image
paths or the image objects themselves, and allow the users to pass an
optional path for saving the output of the decorated function.
"""

import functools
import ants


def ANTsImageToANTsImage(func):
    """
    A decorator for functions that process an ANTs image and output another ANTs image.
    Assumes that the argument of the passed in function is an ANTs image.

    This decorator is designed to extend functions that take an ANTs image as input
    and output another ANTs image. It supports seamless handling of input images
    provided as either file paths (str) or `ants.core.ANTsImage` objects. The resulting
    processed image can optionally be saved to a specified file path.

    Args:
        func (Callable): The function to be decorated. It should accept an ANTs image as
            the first argument and return a processed ANTs image.

    Returns:
        Callable: A wrapper function that:
            - Reads the input image if a file path (str) is provided.
            - Passes an `ants.core.ANTsImage` object to the decorated function.
            - Saves the output image to the specified file path if `out_path` is provided.

    Example:

        .. code-block:: python

            import ants
            from petpal.utils.decorators import ANTsImageToANTsImage
            from petpal.preproc.segmentaion_tools import calc_vesselness_measure_image

            # Defining the decorated function
            @ANTsImageToANTsImage
            def step_calc_vesselness_measure_image(input_image: ants.core.ANTsImage,
                                                   sigma_min: float = 2.0,
                                                   sigma_max: float = 8.0,
                                                   alpha: float = 0.5,
                                                   beta: float = 0.5,
                                                   gamma: float = 5.0,
                                                   morph_open_radius: int = 1,
                                                   **hessian_func_kwargs):
                return calc_vesselness_measure_image(input_image=input_image,
                                                     sigma_min=sigma_min,
                                                     sigma_max=sigma_max,
                                                     alpha=alpha,
                                                     beta=beta,
                                                     gamma=gamma,
                                                     morph_open_radius=morph_open_radius,
                                                     **hessian_func_kwargs)

            # Conventional use of calc_vesselness_measure_image

            input_img = ants.image_read('/path/to/3d/img/.nii.gz')
            vess_img = calc_vesselness_measure_image(input_img) # Using all default values
            ants.image_write(vess_img, '/path/to/out/img/.nii.gz')


            # Using the decorated version
            ## Using paths as inputs
            vess_img = step_calc_vesselness_measure_image('/path/to/3d/img/.nii.gz',
                                                          '/path/to/out/img/.nii.gz')

            ### Not saving output image
            vess_img = step_calc_vesselness_measure_image('/path/to/3d/img/.nii.gz',
                                                          None)

            ## Using images as inputs
            input_img = ants.image_read('/path/to/3d/img/.nii.gz')
            vess_img = step_calc_vesselness_measure_image(input_img,
                                                          '/path/to/out/img/.nii.gz')

            ## Ignoring the return value to just save image
            step_calc_vesselness_measure_image('/path/to/3d/img/.nii.gz',
                                               '/path/to/out/img/.nii.gz')


    Raises:
        TypeError: If `in_img` is not a string or `ants.core.ANTsImage`.

    Notes:
        - If `in_img` is provided as a file path, the image is read using `ants.image_read`.
        - The output image is written to the desired path using `ants.image_write` if
          `out_path` is specified.
    """

    @functools.wraps(func)
    def wrapper(in_img:ants.core.ANTsImage | str,
                out_path: str,
                *args, **kwargs):
        if isinstance(in_img, str):
            in_image = ants.image_read(in_img)
        elif isinstance(in_img, ants.core.ANTsImage):
            in_image = in_img
        else:
            raise TypeError('in_img must be str or ants.core.ANTsImage')
        out_img = func(in_image, *args, **kwargs)
        if out_path is not None:
            ants.image_write(out_img, out_path)
        return out_img
    return wrapper
