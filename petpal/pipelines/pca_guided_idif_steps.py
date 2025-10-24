r"""
This module provides classes and utilities for computing Image-Derived Input Functions
(IDIF) using a PCA-guided approach. It is designed to integrate into preprocessing pipelines
and offers functionality for optimizing voxel selection, applying fitting techniques,
and selecting top-ranked voxels from PCA components.

Classes:
--------

1. :class:`PCAGuidedIDIFMixin`:
    - Acts as a mixin to provide shared functionality across PCA-guided steps.
    - Manages input/output paths, initializes configurations, and helps connect pipeline steps.
    - Provides methods to infer output paths and set input paths dynamically.

2. :class:`PCAGuidedFitIDIFStep`:
    - Represents a pipeline step for optimizing voxel selection to fit the IDIF using PCA-guided methods.
    - Extends both :class:`~.ObjectBasedStep` and :class:`~.PCAGuidedIDIFMixin` to integrate optimization fitting into a pipeline.
    - Includes parameters to control optimization behavior, such as:
        - **alpha**: Weight for the peak term to encourage a high TAC peak.
        - **beta**: Weight for the smoothness term to ensure low total variation.
        - **method**: The optimization method for fitting.

3. :class:`PCAGuidedTopVoxelsIDIFStep`:
    - Represents a pipeline step to calculate the IDIF by selecting the top `N` voxels from a specific PCA component.
    - Extends both :class:`~.ObjectBasedStep` and :class:`~.PCAGuidedIDIFMixin` for pipeline functionality.
    - Allows customizing the selected PCA component and the number of voxels to include.

Key Features:
-------------
- **Pipeline Integration**:
    All classes are designed to be modular and compatible with preprocessing pipelines.
    They inherit functionalities from :class:`~.ObjectBasedStep` and :class:`~.StepsAPI`.

- **BIDS-Compatible Output**:
    Output paths for TAC files are generated in a BIDS-like format using subject and session information
    extracted from the input image paths.

- **Optimization and Voxel Selection**:
    Classes include functionality to:
        - Select voxels based on PCA component ranking.
        - Optimize voxel selection iteratively for better IDIF calculation.
        - Use terms like smoothness, noise reduction, and peak emphasis for guiding optimization.

"""

import warnings

from ..input_function import pca_guided_idif
from ..pipelines.preproc_steps import ImageToImageStep
from ..pipelines.steps_base import StepsAPI, ObjectBasedStep
from ..utils.bids_utils import gen_bids_like_filepath, parse_path_to_get_subject_and_session_id, snake_to_camel_case


class PCAGuidedIDIFMixin(StepsAPI):
    r"""Mixin class to provide shared functionality for PCA-guided IDIF pipeline steps.

    This class is designed to standardize and extend functionality across various PCA-guided
    IDIF calculation steps. It handles input and output path management, configuration initialization,
    and helps define the relationships between preprocessing pipeline steps.

    Inheritance:
        Inherits from the :class:`~.StepsAPI` base class.

    Attributes:
        init_kwargs (dict): A dictionary containing initial configuration parameters.
        call_kwargs (dict): A dictionary containing runtime call-specific parameters.
        input_image_path (str): Path to the input 4D-PET image.
        mask_image_path (str): Path to the mask image in the same space as 4D-PET image.
        output_tac_path (str): Path where output TAC data will be saved.
        num_pca_components (int): Number of PCA components to compute.
        verbose (bool): Whether to show detailed diagnostic output or not.
        name (str): Name of the pipeline step (default is "pca_guided_idif_base").

    """
    def __init__(self,
                 input_image_path: str,
                 mask_image_path: str,
                 output_tac_path: str,
                 num_pca_components: int,
                 verbose: bool
                 ):
        StepsAPI.__init__(self)
        self.init_kwargs = {'input_image_path': input_image_path,
                            'mask_image_path': mask_image_path,
                            'output_tac_path': output_tac_path,
                            'num_pca_components': num_pca_components,
                            'verbose': verbose
                            }
        self.call_kwargs = {}
        self.input_image_path = input_image_path
        self.mask_image_path = mask_image_path
        self.output_tac_path = output_tac_path
        self.num_pca_components = num_pca_components
        self.verbose = verbose
        self.name = 'pca_guided_idif_base'


    @property
    def input_image_path(self):
        return self.init_kwargs['input_image_path']

    @input_image_path.setter
    def input_image_path(self, value):
        self.init_kwargs['input_image_path'] = value

    @property
    def mask_image_path(self):
        return self.init_kwargs['mask_image_path']

    @mask_image_path.setter
    def mask_image_path(self, value):
        self.init_kwargs['mask_image_path'] = value

    @property
    def output_tac_path(self) -> str:
        return self.init_kwargs['output_tac_path']

    @output_tac_path.setter
    def output_tac_path(self, value):
        self.init_kwargs['output_tac_path'] = value

    @property
    def num_pca_components(self):
        return self.init_kwargs['num_pca_components']

    @num_pca_components.setter
    def num_pca_components(self, value):
        self.init_kwargs['num_pca_components'] = value

    @property
    def verbose(self):
        return self.init_kwargs['verbose']

    @verbose.setter
    def verbose(self, value):
        self.init_kwargs['verbose'] = value

    def set_input_as_output_from(self, *sending_steps: ImageToImageStep) -> None:
        """Sets the input image paths based on the output paths from other steps in the pipeline.

        The first sending step will set the input image path, and the second sending step will
        set the second image path.

        Args:
            sending_steps (tuple[ImageToImageStep]): Two pipeline steps whose outputs will be used
                as the input image path and second image input path.

        Raises:
            AssertionError: If the number of provided sending steps is not exactly two.
        """
        assert len(sending_steps) == 2, "ImagePairToArrayStep must have 2 sending ImageToImageStep steps."
        if isinstance(sending_steps[0], ImageToImageStep):
            self.input_image_path = sending_steps[0].output_image_path
        else:
            super().set_input_as_output_from(sending_steps[0])
        if isinstance(sending_steps[1], ImageToImageStep):
            self.mask_image_path = sending_steps[1].output_image_path
        else:
            super().set_input_as_output_from(sending_steps[1])

    def infer_outputs_from_inputs(self,
                                  out_dir: str,
                                  der_type: str = 'tacs',
                                  suffix: str = 'tac',
                                  ext: str = '.tsv',
                                  **extra_desc):
        r"""Infers the output array path based on the inputs and specified parameters.

        This method generates a BIDS-like derivatives filepath for the output based on the subject and
        session IDs extracted from the input image path.

        Args:
            out_dir (str): Directory where the output array will be saved.
            der_type (str, optional): Type of derivative. Will set the sub-directory in `out_dir`. Defaults to 'tacs'.
            suffix (str, optional): Suffix for the output filename. Defaults to 'tac'.
            ext (str, optional): File extension for the output file. Defaults to '.tsv'.
            **extra_desc: Additional descriptive parameters for the output filename.

        """
        sub_id, ses_id = parse_path_to_get_subject_and_session_id(self.input_image_path)
        step_name_in_camel_case = snake_to_camel_case(self.name)
        filepath = gen_bids_like_filepath(sub_id=sub_id, ses_id=ses_id, suffix=suffix, bids_dir=out_dir,
                                          modality=der_type, ext=ext, desc=step_name_in_camel_case, **extra_desc)
        self.output_tac_path = filepath


class PCAGuidedFitIDIFStep(ObjectBasedStep, PCAGuidedIDIFMixin):
    r"""Pipeline step for fitting the PCA-guided IDIF using an optimization-based approach.

    This step extends the :class:`~.PCAGuidedIDIFMixin` for shared functionality and the :class:`~.ObjectBasedStep`
    for pipeline execution. It uses the :class:`~.PCAGuidedIdifFitter` class to iteratively optimize voxel
    selection for IDIF generation based on specified parameters.

    Inheritance:
        Inherits from :class:`~.ObjectBasedStep` and :class:`~.PCAGuidedIDIFMixin`.

    Attributes:
        alpha (float): Parameter weight for the peak term (emphasizes high TAC peaks).
        beta (float): Parameter weight for the smoothness term (encourages smoother signals).
        method (str): Optimization method to be used during fitting.

    Example:

        .. code-block:: python

            from petpal.pipelines.pca_guided_idif_steps import PCAGuidedFitIDIFStep

            step = PCAGuidedFitIDIFStep(input_image_path='/path/to/image.nii',
                                        mask_image_path='/path/to/mask.nii',
                                        output_array_path='/path/to/output.tsv',
                                        num_pca_components=5,
                                        verbose=True,
                                        alpha=0.5,
                                        beta=0.3,
                                        method='ampgo')
            ## Running and saving the results
            step()  # Executes the step

    """
    def __init__(self,
                 input_image_path: str,
                 mask_image_path: str,
                 output_array_path: str,
                 num_pca_components: int,
                 verbose: bool,
                 alpha: float,
                 beta: float,
                 method: str,
                 **meth_kwargs):
        r"""Initializes the PCA-guided fitting IDIF step.

        Args:
            input_image_path (str): Path to the input 4D-PET image.
            mask_image_path (str): Path to the mask image in the same space as 4D-PET image.
            output_array_path (str): Path where the resulting TAC data will be saved.
            num_pca_components (int): Number of PCA components to consider.
            verbose (bool): If `True`, enables detailed diagnostic output.
            alpha (float): Parameter weight for the peak term.
            beta (float): Parameter weight for the smoothness term.
            method (str): Optimization method to use for fitting.
            **meth_kwargs: Additional keyword arguments for the optimization method.

        Side Effects:
            - Initializes `python-guided-idif` fitter with the provided parameters.
            - Populates `call_kwargs` with runtime-specific options for the fitting process.

        """
        PCAGuidedIDIFMixin.__init__(self,
                                    input_image_path=input_image_path,
                                    mask_image_path=mask_image_path,
                                    output_tac_path=output_array_path,
                                    num_pca_components=num_pca_components,
                                    verbose=verbose)
        ObjectBasedStep.__init__(self,
                                 name='pca_guided_fit_idif',
                                 class_type=pca_guided_idif.PCAGuidedIdifFitter,
                                 init_kwargs={**self.init_kwargs},
                                 call_kwargs=(self.call_kwargs | {'alpha' : alpha,
                                                                  'beta'  : beta,
                                                                  'method': method,
                                                                  **meth_kwargs}), )

        self.alpha = alpha
        self.beta = beta
        self.method = method

    @property
    def alpha(self):
        return self.call_kwargs['alpha']

    @alpha.setter
    def alpha(self, value):
        self.call_kwargs['alpha'] = value

    @property
    def beta(self):
        return self.call_kwargs['beta']

    @beta.setter
    def beta(self, value):
        self.call_kwargs['beta'] = value

    @property
    def method(self):
        return self.call_kwargs['method']

    @method.setter
    def method(self, value):
        self.call_kwargs['method'] = value

    @classmethod
    def default_pca_guided_idif_fit(cls, name: str = 'pca_guided_fit_idif', verbose=False, **overrides):
        r"""Creates a default instance for PCA-guided IDIF calculation using an optimization-based fitting approach.


        Uses :class:`~.pca_guided_idif.PCAGuidedIdifFitter`

        All file paths are set to empty strings by default.

        Args:
            name (str): Name of the step. Defaults to 'pca_guided_fit_idif'.
            verbose (bool): Whether to enable verbose mode (default: False).
            **overrides: Parameter values to override the default settings.

        Returns:
            PCAGuidedFitIDIFStep: A new instance of the PCA-guided IDIF fitting step.

        Default Parameters:
            - input_image_path: '' (empty string)
            - mask_image_path: '' (empty string)
            - output_array_path: '' (empty string)
            - num_pca_components: 3
            - verbose: False
            - alpha: 2.5
            - beta: 0.0
            - method: 'dual_annealing'

        Side Effects:
            - If an invalid override is provided, a warning is issued, and the default instance is returned instead.

        """
        defaults = dict(input_image_path='',
                        mask_image_path='',
                        output_array_path='',
                        num_pca_components=3,
                        verbose=verbose,
                        alpha=2.5,
                        beta=0.0,
                        method='dual_annealing')

        override_dict = defaults | overrides

        try:
            out_class = cls(**override_dict)
        except RuntimeError as err:
            warnings.warn(f"Invalid override: {err}. Using default instance instead.", stacklevel=2)
            out_class = cls(**defaults)
        out_class.name = name
        return out_class

class PCAGuidedTopVoxelsIDIFStep(ObjectBasedStep, PCAGuidedIDIFMixin):
    r"""Pipeline step for PCA-guided IDIF calculation using top-ranked voxels.

    This step extends the :class:`~.PCAGuidedIDIFMixin` and provides functionality to select the top `N`
    voxels from a specified PCA component for IDIF generation. It uses the
    :class:`~.PCAGuidedTopVoxelsIDIF` class for its calculations.

    Inheritance:
        Inherits from :class:`~.ObjectBasedStep` and :class:`~.PCAGuidedIDIFMixin`.

    Attributes:
        selected_component (int): PCA component selected for voxel ranking.
        num_of_voxels (int): Number of top-ranking voxels to select.

    Example:

        .. code-block:: python

            from petpal.pipelines.pca_guided_idif_steps import PCAGuidedTopVoxelsIDIFStep

            ## Initializing the step
            step = PCAGuidedTopVoxelsIDIFStep(input_image_path='/path/to/image.nii',
                                              mask_image_path='/path/to/mask.nii',
                                              output_array_path='/path/to/output.tsv',
                                              num_pca_components=3,
                                              verbose=False,
                                              selected_component=2,
                                              num_of_voxels=50)
            ## Running the step and saving the results
            step()

    """
    def __init__(self,
                 input_image_path: str,
                 mask_image_path: str,
                 output_array_path: str,
                 num_pca_components: int,
                 verbose: bool,
                 selected_component: int,
                 num_of_voxels: int):
        r"""Initializes the PCA-guided top voxels IDIF pipeline step.

        Args:
            input_image_path (str): Path to the input 4D-PET image.
            mask_image_path (str): Path to the mask image in the same space as 4D-PET image.
            output_array_path (str): Path where the resulting TAC data will be saved.
            num_pca_components (int): Number of PCA components to consider.
            verbose (bool): If `True`, enables detailed diagnostic output.
            selected_component (int): PCA component to use for voxel ranking.
            num_of_voxels (int): Number of top voxels to select.

        """
        PCAGuidedIDIFMixin.__init__(self,
                                    input_image_path=input_image_path,
                                    mask_image_path=mask_image_path,
                                    output_tac_path=output_array_path,
                                    num_pca_components=num_pca_components,
                                    verbose=verbose)
        ObjectBasedStep.__init__(self,
                                 name='pca_guided_top_voxels_idif',
                                 class_type=pca_guided_idif.PCAGuidedTopVoxelsIDIF,
                                 init_kwargs={'input_image_path'  : input_image_path,
                                              'mask_image_path'   : mask_image_path,
                                              'output_tac_path'   : output_array_path,
                                              'num_pca_components': num_pca_components,
                                              'verbose'           : verbose
                                              },
                                 call_kwargs=(self.call_kwargs | {'selected_component': selected_component,
                                                                  'num_of_voxels'     : num_of_voxels})
                                 )

        self.selected_component = selected_component
        self.num_of_voxels = num_of_voxels

    @property
    def selected_component(self):
        return self.call_kwargs['selected_component']

    @selected_component.setter
    def selected_component(self, value):
        self.call_kwargs['selected_component'] = value

    @property
    def num_of_voxels(self):
        return self.call_kwargs['num_of_voxels']

    @num_of_voxels.setter
    def num_of_voxels(self, value):
        self.call_kwargs['num_of_voxels'] = value

    @classmethod
    def default_pca_guided_idif_top_voxels(cls, name: str = 'pca_guided_top_voxels_idif', verbose=False, **overrides):
        r"""Creates a default instance for PCA-guided IDIF calculation based on selecting the top-ranked voxels from a single PCA component.


         All paths are set to empty strings by default.

        Args:
            name (str): Name of the step. Defaults to 'pca_guided_top_voxels_idif'.
            verbose (bool): Whether to enable verbose mode (default: False).
            **overrides: Parameter values to override the default settings.

        Returns:
            PCAGuidedTopVoxelsIDIFStep: A new instance of the PCA-guided top voxels IDIF step.

        Default Parameters:
            - input_image_path: '' (empty string)
            - mask_image_path: '' (empty string)
            - output_array_path: '' (empty string)
            - num_pca_components: 3
            - verbose: False
            - selected_component: 0
            - num_of_voxels: 50

        Side Effects:
            - If an invalid override is provided, a warning is issued, and the default instance is returned instead.

        """
        defaults = dict(input_image_path = '',
                        mask_image_path = '',
                        output_array_path = '',
                        num_pca_components = 3,
                        verbose = verbose,
                        selected_component = 0,
                        num_of_voxels = 50)

        overrides_dict = defaults | overrides

        try:
            out_class = cls(**overrides_dict)
        except TypeError as err:
            warnings.warn(f"Invalid override: {err}. Using default instance instead.", stacklevel=2)
            out_class = cls(**defaults)
        out_class.name = name
        return out_class
