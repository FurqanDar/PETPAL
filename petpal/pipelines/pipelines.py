import copy
from typing import Union

from .steps_base import *
from .pipeline_context import BIDSyPathsForPipelines, PipelineContext
from .steps_containers import StepsContainer, StepsPipeline
from ..utils.image_io import get_half_life_from_nifti


class BIDS_Pipeline(StepsPipeline):
    """
    A class that combines BIDS data path management with a steps-based pipeline for processing.

    Inherits from:
        BIDSyPathsForPipelines: Manages paths for BIDS data and analysis directories.
        StepsPipeline: Manages a series of processing steps in a pipeline.

    Attributes:
        sub_id (str): Subject ID.
        ses_id (str): Session ID.
        pipeline_name (str): Name of the pipeline. Defaults to 'generic_pipeline'.
        list_of_analysis_dir_names (Union[None, list[str]]): List of names for analysis directories.
        bids_root_dir (Optional[str]): Optional path to the BIDS root directory.
        derivatives_dir (Optional[str]): Optional path to the derivatives directory.
        raw_pet_img_path (Optional[str]): Optional path to the raw PET image.
        raw_anat_img_path (Optional[str]): Optional path to the raw anatomical image.
        segmentation_img_path (Optional[str]): Optional path to the segmentation image.
        segmentation_label_table_path (Optional[str]): Optional path to the segmentation label table.
        raw_blood_tac_path (Optional[str]): Optional path to the raw blood TAC file.
        step_containers (list[StepsContainer]): List of step containers for the pipeline.
        
    Example:
        
        The following is a basic example demonstrating how to instantiate the basic default
        pipeline which performs the following steps:

        #. Crop the raw PET image using a threshold.
        #. Motion correct the cropped image to a mean-PET target where the frames have mean intensities
           greater than or equal to the total mean intensity.
        #. Computes a `weighted-series-sum` image from the cropped PET image. **We add this step since
           it is not part of the default pipeline.**
        #. Registers the motion corrected PET image to a T1w reference image.
        #. For each of the ROI segments defined, we extract TACs and save them.
        #. For the blood TAC, which is assumed to be decay corrected (and WB corrected if appliclable),
           we resample the TAC on the PET scan frame times.
        #. Generate parametric patlak slope and intercept images from the register PET image.
        #. Generate parametric logan slope and intercept images from the register PET image.
        #. For each ROI TAC, calculate a 1TCM fit.
        #. For each ROI TAC, calculate a 1TCM fit.
        #. For each ROI TAC, calculate an irreversible 2TCM (:math:`k_{4}=0`) fit.
        #. For each ROI TAC, calculate a serial 2TCM fit.
        #. For each ROI TAC, calculate a patlak fit.
        #. For each ROI TAC, calculate a logan fit.
        
        We assume that we are running the following code in the ``/code`` folder of a BIDS project.
        
        .. code-block:: python
        
            from petpal.pipelines.pipelines import *
            
            
            # Assuming that the current directory is the `BIDS_ROOT/code` of a BIDS directory.
            
            this_pipeline = BIDS_Pipeline.default_bids_pipeline(sub_id='XXXX',
                                                                ses_id='XX',
                                                                pipeline_name='study_pipeline')
            
            # Plot the dependency graph to quickly glance at all the steps in the pipeline.
            this_pipeline.plot_dependency_graph()
            
            # Check if all the steps can potentially run
            print(this_pipeline.can_steps_potentially_run())
            
            # Check which steps can potentially run
            print(this_pipeline.get_steps_potential_run_state())
            
            ## Editing a pipeline by adding new steps and removing pre-defined steps
            # Instantiating a weighted-series-sum step, and using the pipeline-inferred raw-PET image
            # path to infer the half-life of the radioisotope for the calculation.
            from petpal.pipelines.preproc_steps import ImageToImageStep
            from petpal.utils.useful_functions import weighted_series_sum as wss_func
            wss_step = ImageToImageStep(name='wss',
                                        function=wss_func,
                                        input_image_path='',
                                        output_image_path='',
                                        half_life=get_half_life_from_nifti(this_pipeline.pet_path),
                                        verbose=False
                                       )
            
            # Adding the step to the pipeline with the dependency that wss receives the output from
            # the 'thresh_crop' step in the 'preproc' container.
            this_pipeline.add_step(container_name='preproc', step=wss_step)
            this_pipeline.add_dependency(sending='thresh_crop', receiving='wss')
            
            # Removing the step for calculating the Alt-Logan fits for each of the ROI TACs
            this_pipeline.remove_step('roi_alt_logan_fit')
            
            # Removing the step for calculating the parametric Alt-Logan fits.
            this_pipeline.remove_step('parametric_alt_logan_fit')
            
            # Since we added, and removed steps, we have to update the dependencies.
            this_pipeline.update_dependencies()
            
            # Looking at the updated dependency graph in text-format
            this_pipeline.print_dependency_graph()
            
            # Run all the steps in the pipeline in topological order.
            this_pipeline()
        
    """
    def __init__(self,
                 sub_id: str,
                 ses_id: str,
                 pipeline_name: str = 'generic_pipeline',
                 list_of_analysis_dir_names: Union[None, list[str]] = None,
                 bids_root_dir: str = None,
                 derivatives_dir: str = None,
                 raw_pet_img_path: str = None,
                 raw_anat_img_path: str = None,
                 segmentation_img_path: str = None,
                 segmentation_label_table_path: str = None,
                 raw_blood_tac_path: str = None,
                 step_containers: list[StepsContainer] = None):
        """
        Initializes the BIDS_Pipeline object with the given subject and session IDs,
        pipeline name, paths for various types of data, and step containers for processing.

        Args:
            sub_id (str): Subject ID.
            ses_id (str): Session ID.
            pipeline_name (str, optional): Name of the pipeline. Defaults to 'generic_pipeline'.
            list_of_analysis_dir_names (Union[None, list[str]], optional): List of names for analysis
                directories. Defaults to None.
            bids_root_dir (Optional[str]): Optional path to the BIDS root directory. Defaults to None.
            derivatives_dir (Optional[str]): Optional path to the derivatives directory. Defaults to None.
            raw_pet_img_path (Optional[str]): Optional path to the raw PET image. Defaults to None.
            raw_anat_img_path (Optional[str]): Optional path to the raw anatomical image. Defaults to None.
            segmentation_img_path (Optional[str]): Optional path to the segmentation image. Defaults to None.
            segmentation_label_table_path (Optional[str]): Optional path to the segmentation label table.
                Defaults to None.
            raw_blood_tac_path (Optional[str]): Optional path to the raw blood TAC file. Defaults to None.
            step_containers (list[StepsContainer], optional): List of step containers for the pipeline.
                Defaults to an empty list.
        """
        self.pipeline_context = PipelineContext(sub_id=sub_id,
                                                ses_id=ses_id,
                                                pipeline_name=pipeline_name,
                                                list_of_analysis_dir_names=list_of_analysis_dir_names,
                                                bids_root_directory=bids_root_dir,
                                                derivatives_dir=derivatives_dir,
                                                raw_pet_img_path=raw_pet_img_path,
                                                raw_anat_img_path=raw_anat_img_path,
                                                segmentation_img_path=segmentation_img_path,
                                                segmentation_label_table_path=segmentation_label_table_path,
                                                raw_blood_tac_path=raw_blood_tac_path )
        StepsPipeline.__init__(self, name=pipeline_name, step_containers=step_containers)
        

    def __repr__(self):
        """
        Provides an unambiguous string representation of the BIDSyPathsForPipelines instance.

        Returns:
            str: A string representation showing how the instance can be recreated.

        See Also:
            - :meth:`BIDSyPathsForPipelines.__repr__`
            - :meth:`StepsPipeline.__repr__`
            
        """
        cls_name = type(self).__name__
        info_str = [f'{cls_name}(', ]
        
        in_kwargs = ArgsDict(dict(sub_id=self.sub_id,
                                  ses_id=self.ses_id,
                                  pipeline_name = self.name,
                                  list_of_analysis_dir_names = self.list_of_analysis_dir_names,
                                  bids_root_dir = self.bids_dir,
                                  derivatives_dir = self.derivatives_dir,
                                  raw_pet_img_path = self.pet_path,
                                  raw_anat_img_path = self.anat_path,
                                  segmentation_img_path = self.seg_img,
                                  segmentation_label_table_path = self.seg_table,
                                  raw_blood_tac_path = self.blood_path)
                )
        
        for arg_name, arg_val in in_kwargs.items():
            info_str.append(f'{arg_name}={repr(arg_val)},')
        
        info_str.append('step_containers=[')
        
        for _, container in self.step_containers.items():
            info_str.append(f'{repr(container)},')
        
        info_str.append(']')
        info_str.append(')')
        
        return f'\n    '.join(info_str)
    
    def __str__(self):
        """
        Returns a human-readable string representation of the object. Lists all the directories
        and paths, and the steps-containers with their dependencies.

        Returns:
            str: A string representation of the object.
            
        See Also:
            - :meth:`BIDSyPathsForPipelines.__str__`
            - :meth:`StepsPipeline.__str__<petpal.pipelines.steps_contaiers.StepsPipeline.__str__>`
        """
        pipeline_info_str = StepsPipeline.__str__(self).split("\n")
        paths_info_str = BIDSyPathsForPipelines.__str__(self).split("\n")
        info_str = ["*"*50]+paths_info_str + pipeline_info_str
        return "\n".join(info_str)
        
        
    def update_dependencies_for(self, step_name, verbose=False):
        """
        Updates the dependencies for a specified step in the pipeline. Extends
        :meth:`StepsPipeline.update_dependencies_for<petpal.pipelines.steps_containers.StepsPipeline.update_dependencies_for>`
        to infer the outputs from input paths after updating dependencies.

        Args:
            step_name (str): The name of the step for which to update dependencies.
            verbose (bool, optional): If True, print verbose updates. Defaults to False.
            
        Raises:
            NotImplementedError: The ``infer_outputs_from_inputs`` for the sending step is not
                implemented, OR, ``set_input_as_output_from`` for the receiving step is not
                implemented.
        
        """
        super().update_dependencies_for(step_name, verbose=verbose)
        this_step = self.get_step_from_node_label(step_name)
        this_step_grp_name = self.dependency_graph.nodes(data=True)[step_name]['grp']
        try:
            this_step.infer_outputs_from_inputs(out_dir=self.pipeline_dir,
                                                der_type=this_step_grp_name,)
        except NotImplementedError:
            raise NotImplementedError(f"Step {step_name} does not have the `infer_outputs_from_inputs` "
                                      f"method implemented.")

    @classmethod
    def default_bids_pipeline(cls,
                              sub_id: str,
                              ses_id: str,
                              pipeline_name: str = 'generic_pipeline',
                              list_of_analysis_dir_names: Union[None, list[str]] = None,
                              bids_root_dir: str = None,
                              derivatives_dir: str = None,
                              raw_pet_img_path: str = None,
                              raw_anat_img_path: str = None,
                              segmentation_img_path: str = None,
                              segmentation_label_table_path: str = None,
                              raw_blood_tac_path: str = None):
        """
        Creates a default BIDS pipeline with predefined steps and dependencies.

        Args:
            sub_id (str): Subject ID.
            ses_id (str): Session ID.
            pipeline_name (str, optional): Name of the pipeline. Defaults to 'generic_pipeline'.
            list_of_analysis_dir_names (Union[None, list[str]], optional): List of names for
                analysis directories. Defaults to None.
            bids_root_dir (Optional[str]): Optional path to the BIDS root directory. Defaults
                to None.
            derivatives_dir (Optional[str]): Optional path to the derivatives directory.
                Defaults to None.
            raw_pet_img_path (Optional[str]): Optional path to the raw PET image. Defaults
                to None.
            raw_anat_img_path (Optional[str]): Optional path to the raw anatomical image.
                Defaults to None.
            segmentation_img_path (Optional[str]): Optional path to the segmentation image.
                Defaults to None.
            segmentation_label_table_path (Optional[str]): Optional path to the segmentation
                label table. Defaults to None.
            raw_blood_tac_path (Optional[str]): Optional path to the raw blood TAC file.
                Defaults to None.

        Returns:
            BIDS_Pipeline: A BIDS_Pipeline object with the default steps and dependencies set.
            
        Notes:
            The following steps are defined:
                - Crop the raw PET image using a threshold.
                - Motion correct the cropped image to a mean-PET target where the frames have mean intensities
                  greater than or equal to the total mean intensity.
                - Registers the motion corrected PET image to a T1w reference image.
                - For each of the ROI segments defined, we extract TACs and save them.
                - For the blood TAC, which is assumed to be decay corrected (and WB corrected if appliclable),
                  we resample the TAC on the PET scan frame times.
                - Generate parametric patlak slope and intercept images from the register PET image.
                - Generate parametric logan slope and intercept images from the register PET image.
                - Generate parametric alt-logan slope and intercept images from the register PET image.
                - For each ROI TAC, calculate a 1TCM fit.
                - For each ROI TAC, calculate a 1TCM fit.
                - For each ROI TAC, calculate an irreversible 2TCM (:math:`k_{4}=0`) fit.
                - For each ROI TAC, calculate a serial 2TCM fit.
                - For each ROI TAC, calculate a patlak fit.
                - For each ROI TAC, calculate a logan fit.
                - For each ROI TAC, calculate an alt-logan fit.
                
        See Also:
            - :meth:`default_preprocess_steps<petpal.pipelines.steps_containers.StepsContainer.default_preprocess_steps>`
            - :meth:`default_kinetic_analysis_steps<petpal.pipelines.steps_containers.StepsContainer.default_kinetic_analysis_steps>`
            
        """
        temp_pipeline = StepsPipeline.default_steps_pipeline()
        
        obj = cls(sub_id=sub_id,
                  ses_id=ses_id,
                  pipeline_name=pipeline_name,
                  list_of_analysis_dir_names=list_of_analysis_dir_names,
                  bids_root_dir=bids_root_dir,
                  derivatives_dir=derivatives_dir,
                  raw_pet_img_path=raw_pet_img_path,
                  raw_anat_img_path=raw_anat_img_path,
                  segmentation_img_path=segmentation_img_path,
                  segmentation_label_table_path=segmentation_label_table_path,
                  raw_blood_tac_path=raw_blood_tac_path,
                  step_containers=list(temp_pipeline.step_containers.values())
                  )
        
        obj.dependency_graph = copy.deepcopy(temp_pipeline.dependency_graph)
        
        del temp_pipeline
        
        containers = obj.step_containers
        
        containers["preproc"][0].input_image_path = obj.pipeline_context.paths.pet_path
        containers["preproc"][1].kwargs['half_life'] = get_half_life_from_nifti(obj.pipeline_context.paths.pet_path)
        containers["preproc"][2].kwargs['reference_image_path'] = obj.pipeline_context.paths.anat_path
        containers["preproc"][2].kwargs['half_life'] = get_half_life_from_nifti(obj.pipeline_context.paths.pet_path)
        containers["preproc"][3].segmentation_label_map_path = obj.pipeline_context.paths.seg_table
        containers["preproc"][3].segmentation_image_path = obj.pipeline_context.paths.seg_img
        containers["preproc"][4].raw_blood_tac_path = obj.pipeline_context.paths.blood_path
        
        obj.update_dependencies(verbose=False)
        return obj
