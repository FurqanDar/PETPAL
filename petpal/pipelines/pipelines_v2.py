import copy
import warnings
import networkx as nx
from ..utils.time_activity_curve import MultiTACAnalysisMixin
from ..utils.image_io import safe_copy_meta
from typing import Callable, Union
import inspect
from ..preproc import preproc
from ..input_function import blood_input
from ..kinetic_modeling import parametric_images
from ..kinetic_modeling import tac_fitting
from ..kinetic_modeling import rtm_analysis as pet_rtms
from ..kinetic_modeling import graphical_analysis as pet_grph
from .pipelines import ArgsDict


class FunctionBasedStep:
    def __init__(self, name: str, function: Callable, *args, **kwargs) -> None:
        self.name = name
        self.function = function
        self._func_name = function.__name__
        self.args = args
        self.kwargs = ArgsDict(kwargs)
        self.func_sig = inspect.signature(self.function)
        self.validate_kwargs_for_non_default_have_been_set()
        
    def get_function_args_not_set_in_kwargs(self) -> ArgsDict:
        unset_args_dict = ArgsDict()
        func_params = self.func_sig.parameters
        arg_names = list(func_params)
        for arg_name in arg_names[len(self.args):]:
            if arg_name not in self.kwargs and arg_name != 'kwargs':
                unset_args_dict[arg_name] = func_params[arg_name].default
        return unset_args_dict
    
    def get_empty_default_kwargs(self) -> list:
        unset_args_dict = self.get_function_args_not_set_in_kwargs()
        empty_kwargs = []
        for arg_name, arg_val in unset_args_dict.items():
            if arg_val is inspect.Parameter.empty:
                if arg_name not in self.kwargs:
                    empty_kwargs.append(arg_name)
        return empty_kwargs
    
    def validate_kwargs_for_non_default_have_been_set(self) -> None:
        empty_kwargs = self.get_empty_default_kwargs()
        if empty_kwargs:
            unset_args = '\n'.join(empty_kwargs)
            raise RuntimeError(f"For {self._func_name}, the following arguments must be set:\n{unset_args}")
    
    
    def execute(self):
        print(f"(Info): Executing {self.name}")
        self.function(*self.args, **self.kwargs)
        print(f"(Info): Finished {self.name}")
        
    def generate_kwargs_from_args(self) -> ArgsDict:
        args_to_kwargs_dict = ArgsDict()
        for arg_name, arg_val in zip(list(self.func_sig.parameters), self.args):
            args_to_kwargs_dict[arg_name] = arg_val
        return args_to_kwargs_dict
    
    def __str__(self):
        args_to_kwargs_dict = self.generate_kwargs_from_args()
        info_str = [f'({type(self).__name__} Info):',
                    f'Step Name: {self.name}',
                    f'Function Name: {self._func_name}',
                    f'Arguments Passed:',
                    f'{args_to_kwargs_dict if args_to_kwargs_dict else "N/A"}',
                    'Keyword-Arguments Set:',
                    f'{self.kwargs if self.kwargs else "N/A"}',
                    'Default Arguments:',
                    f'{self.get_function_args_not_set_in_kwargs()}']
        return '\n'.join(info_str)
    
    def set_input_as_output_from(self, sending_step):
        raise NotImplementedError
    
    def all_args_non_empty_strings(self):
        for arg in self.args:
            if arg == '':
                return False
        return True
    
    def all_kwargs_non_empty_strings(self):
        for arg_name, arg in self.kwargs.items():
            if arg == '':
                return False
        return True
    
    def can_potentially_run(self):
        return self.all_args_non_empty_strings() and self.all_kwargs_non_empty_strings()
   
   
class TACsFromSegmentationStep(FunctionBasedStep):
    def __init__(self,
                 input_image_path: str,
                 segmentation_image_path:str,
                 segmentation_label_map_path: str,
                 out_tacs_dir:str,
                 out_tacs_prefix:str,
                 time_keyword='FrameReferenceTime',
                 verbose=False) -> None:
        super().__init__(name='write_roi_tacs',
                         function=preproc.image_operations_4d.write_tacs,
                         input_image_path=input_image_path,
                         segmentation_image_path=segmentation_image_path,
                         label_map_path=segmentation_label_map_path,
                         out_tac_dir=out_tacs_dir,
                         out_tac_prefix=out_tacs_prefix,
                         time_frame_keyword=time_keyword,
                         verbose=verbose,
                         )
        self._input_image = input_image_path
        self._segmentation_image = segmentation_image_path
        self._segmentation_label_map = segmentation_label_map_path
        self._out_tacs_path = out_tacs_dir
        self._out_tacs_prefix = out_tacs_prefix
        
    @property
    def segmentation_image_path(self):
        return self._segmentation_image
    
    @segmentation_image_path.setter
    def segmentation_image_path(self, segmentation_image_path):
        self._segmentation_image = segmentation_image_path
        self.kwargs['segmentation_image_path'] = segmentation_image_path
        
    @property
    def segmentation_label_map_path(self):
        return self._segmentation_label_map
    
    @segmentation_label_map_path.setter
    def segmentation_label_map_path(self, segmentation_label_map_path):
        self._segmentation_label_map = segmentation_label_map_path
        self.kwargs['label_map_path'] = segmentation_label_map_path
        
    @property
    def out_tacs_path(self):
        return self._out_tacs_path
    
    @out_tacs_path.setter
    def out_tacs_path(self, out_tacs_path: str):
        self.kwargs['out_tac_dir'] = out_tacs_path
        self._out_tacs_path = out_tacs_path
    
    @property
    def out_tacs_prefix(self):
        return self._out_tacs_prefix
    
    @out_tacs_prefix.setter
    def out_tacs_prefix(self, out_tacs_prefix: str):
        self.kwargs['out_tac_prefix'] = out_tacs_prefix
        self._out_tacs_prefix = out_tacs_prefix
    
    
    @property
    def out_path_and_prefix(self):
        return self._out_tacs_path, self._out_tacs_prefix
    
    @out_path_and_prefix.setter
    def out_path_and_prefix(self, out_dir_and_prefix):
        try:
            out_dir, out_prefix = out_dir_and_prefix
        except ValueError:
            raise ValueError("Pass a tuple with two items: `(out_dir, out_prefix)`")
        else:
            self.out_tacs_path = out_dir
            self.out_tacs_prefix = out_prefix
        
    @property
    def input_image_path(self):
        return self._input_image
    
    @input_image_path.setter
    def input_image_path(self, input_image_path: str):
        self.kwargs['input_image_path'] = input_image_path
        self._input_image = input_image_path
        
    def set_input_as_output_from(self, sending_step):
        if isinstance(sending_step, ImageToImageStep):
            self.input_image_path = sending_step.output_image_path
        else:
            super().set_input_as_output_from(sending_step)
            
    @classmethod
    def _empty_write_tacs_from_segmentation_rois(cls):
        return cls(input_image_path='',
                   segmentation_image_path='',
                   segmentation_label_map_path='',
                   out_tacs_dir='',
                   out_tacs_prefix='',
                   time_keyword='FrameReferenceTime',
                   verbose=False)
        
    
class ResampleBloodTACStep(FunctionBasedStep):
    def __init__(self,
                 input_raw_blood_tac_path: str,
                 input_image_path: str,
                 out_tac_path:str,
                 lin_fit_thresh_in_mins=30.0
                 ):
        super().__init__(name='resample_bTAC',
                         function=blood_input.resample_blood_data_on_scanner_times,
                         raw_blood_tac=input_raw_blood_tac_path,
                         pet4d_path=input_image_path,
                         out_tac_path=out_tac_path,
                         lin_fit_thresh_in_mins=lin_fit_thresh_in_mins)
        self._raw_blood_tac_path = input_raw_blood_tac_path
        self._input_image_path = input_image_path
        self._resampled_tac_path = out_tac_path
        
    @property
    def raw_blood_tac_path(self):
        return self._raw_blood_tac_path
    
    @raw_blood_tac_path.setter
    def raw_blood_tac_path(self, raw_blood_tac_path):
        self.kwargs['raw_blood_tac'] = raw_blood_tac_path
        self._raw_blood_tac_path = raw_blood_tac_path
        
    @property
    def input_image_path(self):
        return self._input_image_path
    
    @input_image_path.setter
    def input_image_path(self, input_image_path: str):
        self.kwargs['pet4d_path'] = input_image_path
        self._input_image_path = input_image_path
        
    @property
    def resampled_tac_path(self):
        return self._resampled_tac_path
    
    @resampled_tac_path.setter
    def resampled_tac_path(self, resampled_tac_path):
        self.kwargs['out_tac_path'] = resampled_tac_path
        self._resampled_tac_path = resampled_tac_path
        
    def set_input_as_output_from(self, sending_step):
        if isinstance(sending_step, ImageToImageStep):
            self.input_image_path = sending_step.output_image_path
        else:
            super().set_input_as_output_from(sending_step)
            
    @classmethod
    def _empty_resample_blood_data_on_scanner_times(cls):
        return cls(input_raw_blood_tac_path='',
                   input_image_path='',
                   out_tac_path='',
                   lin_fit_thresh_in_mins=30.0)
   
        
class ObjectBasedStep:
    def __init__(self,
                 name: str,
                 class_type: type,
                 init_kwargs: dict,
                 call_kwargs: dict) -> None:
        self.name: str = name
        self.class_type: type = class_type
        self.init_kwargs: ArgsDict = ArgsDict(init_kwargs)
        self.call_kwargs: ArgsDict = ArgsDict(call_kwargs)
        self.init_sig: inspect.Signature = inspect.signature(self.class_type.__init__)
        self.call_sig: inspect.Signature = inspect.signature(self.class_type.__call__)
        self.validate_kwargs()
        self.instance: type = self.class_type(**self.init_kwargs)
        
    def remake_instance(self):
        self.instance = self.class_type(**self.init_kwargs)
    
    def validate_kwargs(self):
        empty_init_kwargs = self.get_empty_default_kwargs(self.init_sig, self.init_kwargs)
        empty_call_kwargs = self.get_empty_default_kwargs(self.call_sig, self.call_kwargs)
        
        if empty_init_kwargs or empty_call_kwargs:
            err_msg = [f"For {self.class_type.__name__}, the following arguments must be set:"]
            if empty_init_kwargs:
                err_msg.append("Initialization:")
                err_msg.append(f"{empty_init_kwargs}")
            if empty_call_kwargs:
                err_msg.append("Calling:")
                err_msg.append(f"{empty_call_kwargs}")
            raise RuntimeError("\n".join(err_msg))
    
    @staticmethod
    def get_args_not_set_in_kwargs(sig: inspect.Signature, kwargs: dict) -> dict:
        unset_args_dict = ArgsDict()
        for arg_name, arg_val in sig.parameters.items():
            if arg_name not in kwargs and arg_name != 'self':
                unset_args_dict[arg_name] = arg_val.default
        return unset_args_dict
    
    def get_empty_default_kwargs(self, sig: inspect.Signature, set_kwargs: dict) -> list:
        unset_kwargs = self.get_args_not_set_in_kwargs(sig=sig, kwargs=set_kwargs)
        empty_kwargs = []
        for arg_name, arg_val in unset_kwargs.items():
            if arg_val is inspect.Parameter.empty:
                if arg_name not in set_kwargs:
                    empty_kwargs.append(arg_name)
        return empty_kwargs
        
    def execute(self, remake_obj: bool = True) -> None:
        if remake_obj:
            self.remake_instance()
        print(f"(Info): Executing {self.name}")
        self.instance(**self.call_kwargs)
        print(f"(Info): Finished {self.name}")
    
    def __str__(self):
        unset_init_args = self.get_args_not_set_in_kwargs(self.init_sig, self.init_kwargs)
        unset_call_args = self.get_args_not_set_in_kwargs(self.call_sig, self.call_kwargs)
        
        info_str = [f'({type(self).__name__} Info):',
                    f'Step Name: {self.name}',
                    f'Class Name: {self.class_type.__name__}',
                    'Initialization Arguments:',
                    f'{self.init_kwargs}',
                    'Default Initialization Arguments:',
                    f'{unset_init_args if unset_init_args else "N/A"}',
                    'Call Arguments:',
                    f'{self.call_kwargs if self.call_kwargs else "N/A"}',
                    'Default Call Arguments:',
                    f'{unset_call_args if unset_call_args else "N/A"}']
        return '\n'.join(info_str)
    
    def set_input_as_output_from(self, sending_step):
        raise NotImplementedError
    
    def all_init_kwargs_non_empty_strings(self):
        for arg_name, arg_val in self.init_kwargs.items():
            if arg_val == '':
                return False
        return True
    
    def all_call_kwargs_non_empty_strings(self):
        for arg_name, arg_val in self.call_kwargs.items():
            if arg_val == '':
                return False
        return True
    
    def can_potentially_run(self):
        return self.all_init_kwargs_non_empty_strings() and self.all_call_kwargs_non_empty_strings()

    
class ImageToImageStep(FunctionBasedStep):
    def __init__(self,
                 name: str,
                 function: Callable,
                 input_image_path: str,
                 output_image_path: str,
                 *args,
                 **kwargs) -> None:
        super().__init__(name, function, *(input_image_path, output_image_path, *args), **kwargs)
        self.input_image_path = copy.copy(self.args[0])
        self.output_image_path = copy.copy(self.args[1])
        self.args = self.args[2:]
    
    def execute(self, copy_meta_file: bool = True) -> None:
        print(f"(Info): Executing {self.name}")
        self.function(self.input_image_path, self.output_image_path, *self.args, **self.kwargs)
        if copy_meta_file:
            safe_copy_meta(input_image_path=self.input_image_path,
                           out_image_path=self.output_image_path)
        print(f"(Info): Finished {self.name}")
    
    def __str__(self):
        io_dict = ArgsDict({'input_image_path': self.input_image_path,
                            'output_image_path': self.output_image_path
                           })
        sup_str_list = super().__str__().split('\n')
        args_ind = sup_str_list.index("Arguments Passed:")
        sup_str_list.insert(args_ind, f"Input & Output Paths:\n{io_dict}")
        def_args_ind = sup_str_list.index("Default Arguments:")
        sup_str_list.pop(def_args_ind + 1)
        sup_str_list.pop(def_args_ind + 1)
        
        return "\n".join(sup_str_list)
    
    def set_input_as_output_from(self, sending_step: FunctionBasedStep) -> None:
        if isinstance(sending_step, ImageToImageStep):
            self.input_image_path = sending_step.output_image_path
        else:
            super().set_input_as_output_from(sending_step)
        
    def can_potentially_run(self):
        input_img_non_empty_str = False if self.input_image_path == '' else True
        output_img_non_empty_str = False if self.output_image_path == '' else True
        return super().can_potentially_run() and input_img_non_empty_str and output_img_non_empty_str
    
    @classmethod
    def default_threshold_cropping(cls, **overrides):
        defaults = dict(name='thresh_crop',
                        function=preproc.image_operations_4d.SimpleAutoImageCropper,
                        input_image_path='',
                        output_image_path='',)
        override_dict = {**defaults, **overrides}
        try:
            return cls(**override_dict)
        except RuntimeError as err:
            warnings.warn(f"Invalid override: {err}. Using default instance instead.", stacklevel=2)
            return cls(**defaults)
        
    @classmethod
    def default_moco_frames_above_mean(cls, verbose=False, **overrides):
        defaults = dict(name='moco_frames_above_mean',
                        function=preproc.motion_corr.motion_corr_frames_above_mean_value,
                        input_image_path='',
                        output_image_path='',
                        motion_target_option='mean_image',
                        verbose=verbose,
                        half_life=None,)
        override_dict = {**defaults, **overrides}
        try:
            return cls(**override_dict)
        except RuntimeError as err:
            warnings.warn(f"Invalid override: {err}. Using default instance instead.", stacklevel=2)
            return cls(**defaults)
    
    @classmethod
    def default_register_pet_to_t1(cls, reference_image_path='', half_life='', verbose=False, **overrides):
        defaults = dict(name='register_pet_to_t1',
                        function=preproc.register.register_pet,
                        input_image_path='',
                        output_image_path='',
                        reference_image_path=reference_image_path,
                        motion_target_option='weighted_series_sum',
                        verbose=verbose,
                        half_life=half_life, )
        override_dict = {**defaults, **overrides}
        try:
            return cls(**override_dict)
        except RuntimeError as err:
            warnings.warn(f"Invalid override: {err}. Using default instance instead.", stacklevel=2)
            return cls(**defaults)
        

PreprocSteps = Union[TACsFromSegmentationStep, ResampleBloodTACStep, ImageToImageStep]


class TACAnalysisStepMixin(MultiTACAnalysisMixin):
    def __init__(self,
                 input_tac_path: str,
                 roi_tacs_dir: str,
                 output_directory: str,
                 output_prefix: str,
                 is_ref_tac_based_model:bool,
                 **kwargs):
        if is_ref_tac_based_model:
            self.init_kwargs = dict(ref_tac_path=input_tac_path,
                                    roi_tacs_dir=roi_tacs_dir,
                                    output_directory=output_directory,
                                    output_filename_prefix=output_prefix,
                                    **kwargs)
        else:
            self.init_kwargs = dict(input_tac_path=input_tac_path,
                                    roi_tacs_dir=roi_tacs_dir,
                                    output_directory=output_directory,
                                    output_filename_prefix=output_prefix,
                                    **kwargs)
        MultiTACAnalysisMixin.__init__(self,
                                       input_tac_path=input_tac_path,
                                       tacs_dir=roi_tacs_dir)
        
        self._output_directory = output_directory
        self._output_prefix = output_prefix
        
    @property
    def input_tac_path(self) -> str:
        return self._input_tac_path
    
    @input_tac_path.setter
    def input_tac_path(self, input_tac_path: str):
        self._input_tac_path = input_tac_path
        self.init_kwargs['input_tac_path'] = input_tac_path
        
    @property
    def reference_tac_path(self) -> str:
        return self.input_tac_path
    
    @reference_tac_path.setter
    def reference_tac_path(self, ref_tac_path: str):
        self.input_tac_path = ref_tac_path
        self.init_kwargs['ref_tac_path'] = ref_tac_path
    
    @property
    def roi_tacs_dir(self) -> str:
        return self.tacs_dir
    
    @roi_tacs_dir.setter
    def roi_tacs_dir(self, roi_tacs_dir: str):
        self.tacs_dir = roi_tacs_dir
        self.init_kwargs['roi_tacs_dir'] = roi_tacs_dir
    
    @property
    def output_directory(self) -> str:
        return self._output_directory
    
    @output_directory.setter
    def output_directory(self, output_directory: str):
        self._output_directory = output_directory
        self.init_kwargs['output_directory'] = output_directory
    
    @property
    def output_prefix(self) -> str:
        return self._output_prefix
    
    @output_prefix.setter
    def output_prefix(self, output_prefix: str):
        self._output_prefix = output_prefix
        self.init_kwargs['output_filename_prefix'] = output_prefix
    
    @property
    def out_path_and_prefix(self):
        return self._output_directory, self._output_prefix
    
    @out_path_and_prefix.setter
    def out_path_and_prefix(self, out_dir_and_prefix):
        try:
            out_dir, out_prefix = out_dir_and_prefix
        except ValueError:
            raise ValueError("Pass a tuple with two items: `(out_dir, out_prefix)`")
        else:
            self.output_directory = out_dir
            self.output_prefix = out_prefix


class GraphicalAnalysisStep(ObjectBasedStep, TACAnalysisStepMixin):
    def __init__(self,
                 input_tac_path: str,
                 roi_tacs_dir: str,
                 output_directory: str,
                 output_prefix: str,
                 method: str,
                 fit_threshold_in_mins: float = 30.0,
                 image_rescale: float = 1.0/37000.0):
        TACAnalysisStepMixin.__init__(self,
                                      input_tac_path=input_tac_path,
                                      roi_tacs_dir=roi_tacs_dir,
                                      output_directory=output_directory,
                                      output_prefix=output_prefix,
                                      is_ref_tac_based_model=False,
                                      method=method,
                                      fit_thresh_in_mins = fit_threshold_in_mins,
                                      )
        ObjectBasedStep.__init__(self, name=f'roi-{method}-fit',
                                 class_type=pet_grph.MultiTACGraphicalAnalysis,
                                 init_kwargs=self.init_kwargs,
                                 call_kwargs=dict())
    
    def set_input_as_output_from(self, sending_step: PreprocSteps) -> None:
        if isinstance(sending_step, TACsFromSegmentationStep):
            self.roi_tacs_dir = sending_step.out_tacs_path
        elif isinstance(sending_step, ResampleBloodTACStep):
            self.input_tac_path = sending_step.resampled_tac_path
        else:
            super().set_input_as_output_from(sending_step)
        
        
class TCMFittingAnalysisStep(ObjectBasedStep, TACAnalysisStepMixin):
    def __init__(self,
                 input_tac_path: str,
                 roi_tacs_dir: str,
                 output_directory: str,
                 output_prefix: str,
                 compartment_model='2tcm-k4zer0',
                 **kwargs):
        TACAnalysisStepMixin.__init__(self,
                                      input_tac_path=input_tac_path,
                                      roi_tacs_dir=roi_tacs_dir,
                                      output_directory=output_directory,
                                      output_prefix=output_prefix,
                                      is_ref_tac_based_model=False,
                                      compartment_model=compartment_model,
                                      **kwargs)
        
        ObjectBasedStep.__init__(self, name=f'roi-{compartment_model}-fit',
                                 class_type=tac_fitting.FitTCMToManyTACs,
                                 init_kwargs=self.init_kwargs,
                                 call_kwargs=dict())
        
        
    def set_input_as_output_from(self, sending_step: PreprocSteps) -> None:
        if isinstance(sending_step, TACsFromSegmentationStep):
            self.roi_tac_path = sending_step.out_tacs_path
        elif isinstance(sending_step, ResampleBloodTACStep):
            self.input_tac_path = sending_step.resampled_tac_path
        else:
            super().set_input_as_output_from(sending_step)
        

class RTMFittingAnalysisStep(ObjectBasedStep, TACAnalysisStepMixin):
    def __init__(self,
                 ref_tac_path: str,
                 roi_tac_path: str,
                 output_directory: str,
                 output_prefix: str,
                 rtm_model: str,
                 bounds = None,
                 k2_prime = None,
                 fit_threshold_in_mins: float = 30.0,):
        
        TACAnalysisStepMixin.__init__(self, input_tac_path=ref_tac_path,
                                      roi_tacs_dir=roi_tac_path,
                                      output_directory=output_directory,
                                      output_prefix=output_prefix,
                                      is_ref_tac_based_model=True,
                                      method=rtm_model, )
        ObjectBasedStep.__init__(self, name=f'roi-{rtm_model}-fit',
                                 class_type=pet_rtms.RTMAnalysis,
                                 init_kwargs=self.init_kwargs,
                                 call_kwargs=dict(bounds=bounds,
                                                  t_thresh_in_mins=fit_threshold_in_mins,
                                                  k2_prime=k2_prime))

    def set_input_as_output_from(self, sending_step: PreprocSteps) -> None:
        if isinstance(sending_step, TACsFromSegmentationStep):
            self.roi_tac_path = sending_step.out_tacs_path
        else:
            super().set_input_as_output_from(sending_step)


class ParametricGraphicalAnalysisStep(ObjectBasedStep, TACAnalysisStepMixin):
    def __init__(self,
                 input_tac_path: str,
                 input_image_path: str,
                 output_directory: str,
                 output_prefix: str,
                 method: str,
                 fit_threshold_in_mins: float = 30.0, ):
        TACAnalysisStepMixin.__init__(self, input_tac_path=input_tac_path,
                                      pet4D_img_path=input_image_path,
                                      roi_tacs_dir=None,
                                      output_directory=output_directory,
                                      output_prefix=output_prefix,
                                      is_ref_tac_based_model=False,
                                      )
        del self.init_kwargs['roi_tacs_dir']
        
        ObjectBasedStep.__init__(self, name=f'parametric-{method}-fit',
                                 class_type=parametric_images.GraphicalAnalysisParametricImage,
                                 init_kwargs=self.init_kwargs,
                                 call_kwargs=dict(method_name=method,
                                                  t_thresh_in_mins=fit_threshold_in_mins,))
        self._input_image_path = input_image_path
    
    
    @property
    def input_image_path(self) -> str:
        return self._input_image_path
    
    @input_image_path.setter
    def input_image_path(self, input_image_path: str):
        self._input_image_path = input_image_path
        self.init_kwargs['pet4D_img_path'] = input_image_path
    
    
    def set_input_as_output_from(self, sending_step: PreprocSteps) -> None:
        if isinstance(sending_step, ResampleBloodTACStep):
            self.input_tac_path = sending_step.resampled_tac_path
        elif isinstance(sending_step, ImageToImageStep):
            self.input_image_path = sending_step.output_image_path
        else:
            super().set_input_as_output_from(sending_step)


def get_template_steps():
    
    out_dict = dict(
            thresh_crop_step = ImageToImageStep.default_threshold_cropping(),
            moco_frames_above_mean = ImageToImageStep.default_moco_frames_above_mean(),
            register_pet_to_t1 = ImageToImageStep.default_register_pet_to_t1(),
            write_roi_tacs = TACsFromSegmentationStep._empty_write_tacs_from_segmentation_rois(),
            resample_blood = ResampleBloodTACStep._empty_resample_blood_data_on_scanner_times(),
            )
    
    return out_dict


KMStepType = Union[GraphicalAnalysisStep,
                    TCMFittingAnalysisStep,
                    ParametricGraphicalAnalysisStep,
                    RTMFittingAnalysisStep]

StepType = Union[FunctionBasedStep, ObjectBasedStep,
                 PreprocSteps, KMStepType]


class StepsContainer:
    def __init__(self, name: str):
        self.name = name
        self.step_objs: list[StepType] = []
        self.step_names: list[str] = []
        
    def add_step(self, step: StepType):
        if step.name not in self.step_names:
            self.step_objs.append(copy.deepcopy(step))
            self.step_names.append(self.step_objs[-1].name)
        else:
            raise KeyError("A step with this name already exists.")
        
    def print_step_details(self):
        if not self.step_objs:
            print("No steps in container.")
        else:
            print(f"({self.name} Pipeline Info):")
            for step_id, a_step in enumerate(self.step_objs):
                print('-' * 80)
                print(f"Step Number {step_id + 1}")
                print(a_step)
                print('-' * 80)
            print("*" * 90)
            
    def print_step_names(self) -> None:
        if not self.step_objs:
            print("No steps in container.")
        else:
            print(f"({self.name} pipeline info):")
            print('-' * 80)
            for step_id, step_name in enumerate(self.step_names):
                print(f"Step Number {step_id+1}: {step_name}")
            print('-' * 80)
            
    def __call__(self):
        for step_id, (step_name, a_step) in enumerate(zip(self.step_names, self.step_objs)):
            a_step.execute()
            
    def __getitem__(self, step: Union[int, str]):
        if isinstance(step, int):
            try:
                return self.step_objs[step]
            except IndexError:
                raise IndexError(f"Step number {step} does not exist.")
        elif isinstance(step, str):
            if step not in self.step_names:
                raise KeyError(f"Step name {step} does not exist.")
            try:
                step_index = self.step_names.index(step)
                return self.step_objs[step_index]
            except KeyError:
                raise KeyError(f"Step name {step} does not exist.")
        else:
            raise TypeError(f"Key must be an integer or a string. Got {type(step)}")
        
    
    
class StepsPipeline:
    def __init__(self, name: str):
        self.name = name
        self.preproc = StepsContainer(name='preproc')
        self.km = StepsContainer(name='km')
        self.dependency_graph = nx.DiGraph()
        
    def __call__(self):
        self.preproc()
        self.km()
        
    def run_km(self):
        self.km()
        
    def run_preproc(self):
        self.preproc()
    
    def add_step(self, container_name: str, step: StepType):
        if step.name in self.dependency_graph: #Ensures unique step names for now
            raise KeyError(f"Step name {step.name} already exists.")
        
        if container_name == 'preproc':
            self.preproc.add_step(step)
        elif container_name == 'km':
            self.km.add_step(step)
        else:
            raise KeyError(f"Container name {container_name} does not exist. "
                           f"Must be 'preproc' or 'km'.")
        self.dependency_graph.add_node(f"{step.name}")
        self.dependency_graph.nodes[f"{step.name}"]['grp'] = container_name
        
    def print_steps_names(self, container_name: str = None):
        if container_name is None:
            self.preproc.print_step_names()
            self.km.print_step_names()
        elif container_name == 'preproc':
            self.preproc.print_step_names()
        elif container_name == 'km':
            self.km.print_step_names()
        else:
            raise KeyError(f"Container name {container_name} does not exist. "
                           f"Must be 'preproc' or 'km'.")
        
    def print_steps_details(self, container_name: str = None):
        if container_name is None:
            self.preproc.print_step_details()
            self.km.print_step_details()
        elif container_name == 'preproc':
            self.preproc.print_step_details()
        elif container_name == 'km':
            self.km.print_step_details()
        else:
            raise KeyError(f"Container name {container_name} does not exist. "
                           f"Must be 'preproc' or 'km'.")
        
    def add_dependency(self, sending: str, receiving: str):
        node_names = list(self.dependency_graph.nodes)
        if sending not in node_names:
            raise KeyError(f"Step name {sending} does not exist.")
        if receiving not in node_names:
            raise KeyError(f"Step name {receiving} does not exist.")
        
        self.dependency_graph.add_edge(sending, receiving)
        
        if not nx.is_directed_acyclic_graph(self.dependency_graph):
            raise RuntimeError(f"Adding dependency {sending} -> {receiving} creates a cycle!")
        
    def get_step_from_node_label(self, node_label: str):
        graph_nodes = self.dependency_graph.nodes(data=True)
        container_name = graph_nodes[node_label]['grp']
        if container_name == 'preproc':
            return self.preproc[node_label]
        elif container_name == 'km':
            return self.km[node_label]
        else:
            raise KeyError(f"Container name {container_name} does not exist.")
    
    def update_dependencies(self, verbose=False):
        for node_name in nx.topological_sort(self.dependency_graph):
            sending_step = self.get_step_from_node_label(node_name)
            for an_edge in self.dependency_graph[node_name]:
                receiving_step = self.get_step_from_node_label(an_edge)
                try:
                    receiving_step.set_input_as_output_from(sending_step)
                except NotImplementedError:
                    # warnings.warn(
                    #     f"Step `{receiving_step.name}` of type `{type(receiving_step).__name__}` does not have a "
                    #     f"set_input_as_output_from method implemented.\nSkipping.", RuntimeWarning, stacklevel=1)
                    if verbose:
                        print(f"Step `{receiving_step.name}` of type `{type(receiving_step).__name__}` does not have a "
                              f"set_input_as_output_from method implemented.\nSkipping.")
                else:
                    if verbose:
                        print(f"Updated input-output dependency between {sending_step.name} and {receiving_step.name}")

    def get_steps_potential_run_state(self) -> dict:
        step_maybe_runnable = {}
        for node_name in nx.topological_sort(self.dependency_graph):
            step = self.get_step_from_node_label(node_name)
            step_maybe_runnable[node_name] = step.can_potentially_run()
        return step_maybe_runnable

    def can_steps_potentially_run(self) -> bool:
        steps_run_state = self.get_steps_potential_run_state()
        for step_name, run_state in steps_run_state.items():
            if not run_state:
                return False
        return True
        
        