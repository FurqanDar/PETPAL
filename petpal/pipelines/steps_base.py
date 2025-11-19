import inspect
from typing import Callable
import types

class ArgsDict(dict):
    """
    A specialized subclass of Python's built-in `dict` that provides a customized string representation.

    Attributes:
        None, since ArgsDict inherits directly from dict.
    """
    def __str__(self):
        """
        Returns a formatted string representation of the dictionary.
        
        The string output will list each key-value pair on a new line with indentation to improve readability.
        
        Returns:
            str: A string containing the formatted key-value pairs, indented for clarity, and only to be used internally
                for formatting inputs of steps.
        """
        rep_str = [f'\t{arg}={repr(val)}' for arg, val in self.items()]
        return ',\n'.join(rep_str)

    def __repr__(self):
        rep_str = [f'{type(self).__name__}'+'({']
        rep_str.extend([f'  {repr(arg)}:{repr(val)},' for arg, val in self.items()])
        rep_str.append('})')
        return '\n'.join(rep_str)


class StepsAPI:
    """
    StepsAPI provides an interface for defining steps in a processing pipeline.

    This class outlines methods that allow input and output management between different steps,
    and perform inference of output files based on input data and given parameters.

    """

    def __init__(self,
                 name: str,
                 skip_step: bool = False):
        self.name = name
        self.skip_step = skip_step

    def execute(self, *args, **kwargs):
        raise NotImplementedError

    def set_input_as_output_from(self, *sending_steps):
        """
        Sets the input of the current step as the output from a list of steps.

        Args:
            *sending_steps: The previous steps from which the output will be used as input for the current step.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
            
        Notes:
            For a concrete example, take a look at:
            :meth:`TACsFromSegmentationStep<petpal.pipelines.preproc_steps.TACsFromSegmentationStep.set_input_as_output_from>`

        .. important::
           If a step takes multiple input steps. the implementation will have a defined order for steps.

        """
        raise NotImplementedError
    
    def infer_outputs_from_inputs(self, out_dir: str, der_type: str, suffix: str = None, ext: str = None, **extra_desc):
        """
        Infers output files from input data based on the specified output directory,
        derivative type, optional suffix and extension, plus any extra descriptions.

        Args:
            out_dir (str): The directory where the output files will be saved.
            der_type (str): The type of derivative being produced.
            suffix (str, optional): An optional suffix for the output files.
            ext (str, optional): An optional extension for the output files.
            **extra_desc: Additional keyword arguments for extra descriptions to be included.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
            
        Notes:
            For a concrete example, take a look at:
            :meth:`TACsFromSegmentationStep<petpal.pipelines.preproc_steps.TACsFromSegmentationStep.infer_outputs_from_inputs>`
        """
        raise NotImplementedError

    def configure(self,
                  input_setter: Callable = None,
                  executor: Callable = None,
                  output_inferrer: Callable = None):
        if input_setter:
            self.set_input_as_output_from = types.MethodType(input_setter, self)
        if executor:
            self.execute = types.MethodType(executor, self)
        if output_inferrer:
            self.infer_outputs_from_inputs = types.MethodType(output_inferrer, self)

    def __call__(self, *args, **kwargs):
        if not self.skip_step:
            print(f"(Info): Executing {self.name}")
            self.execute(*args, **kwargs)
            print(f"(Info): Finished {self.name}")
        else:
            print(f"(Info): Skipping {self.name}")


class BaseProcessingStep(StepsAPI):
    def __init__(self,
                 name: str,
                 callable_target: Callable | type,
                 *args,
                 init_kwargs: dict = None,
                 call_kwargs: dict = None,
                 **kwargs):
        StepsAPI.__init__(self, skip_step=False, name=name)
        self.callable_target: Callable | type = callable_target
        self.is_class: bool = inspect.isclass(callable_target)
        self.is_function: bool = not self.is_class

        # Validate parameter usage based on Callable type
        self._validate_parameter_usage(kwargs=kwargs, init_kwargs=init_kwargs, call_kwargs=call_kwargs)

        # Initialize storage
        self.init_kwargs = ArgsDict(init_kwargs or {})
        self.call_kwargs = ArgsDict(call_kwargs or {})
        self.args = args or ()
        self.kwargs = ArgsDict(kwargs or {})
        if self.is_class:
            self.init_sig = self._get_valid_signature_for_method(callable_target.__init__)
            self.call_sig = self._get_valid_signature_for_method(callable_target.__call__)
            self._validate_object()
        else:
            self.func_sig = inspect.signature(callable_target)
            self._validate_function()

    def __str__(self):
        if self.is_class:
            unset_init = self._get_unset_object_args(self.init_sig, self.init_kwargs, len(self.args))
            unset_call = self._get_unset_object_args(self.call_sig, self.call_kwargs)

            info_str = [
                f'({type(self).__name__} Info):',
                f'Step Name: {self.name}',
                f'Class Name: {self.callable_target.__name__}',
                'Initialization Arguments:',
                f'{self.init_kwargs}',
                'Default Initialization Arguments:',
                f'{unset_init if unset_init else "N/A"}',
                'Call Arguments:',
                f'{self.call_kwargs if self.call_kwargs else "N/A"}',
                'Default Call Arguments:',
                f'{unset_call if unset_call else "N/A"}'
                ]
        else:
            # For functions, show the reconstructed function interface
            func_params = list(inspect.signature(self.callable_target).parameters)
            reconstructed_args = ArgsDict()
            for arg_name, arg_val in zip(func_params, self.args):
                reconstructed_args[arg_name] = arg_val
            info_str = [
                f'({type(self).__name__} Info):',
                f'Step Name: {self.name}',
                f'Function Name: {self.callable_target.__name__}',
                'Positional Arguments:',
                f'    {", ".join(reconstructed_args)}' if reconstructed_args else "N/A",
                'Keyword Arguments:',
                f'{self.kwargs if self.kwargs else "N/A"}',
                'Default Arguments:',
                f'{self._get_unset_function_args()}'
                ]
        return '\n'.join(info_str)

    def __repr__(self):
        cls_name = type(self).__name__
        target_name = f'{self.callable_target.__module__}.{self.callable_target.__name__}'

        info_str = [f'{cls_name}(']
        info_str.append(f'\tname={repr(self.name)},')
        info_str.append(f'\tcallable_target={target_name},')

        if self.is_class:
            if self.init_kwargs:
                info_str.append('init_kwargs={')
                for k, v in self.init_kwargs.items():
                    info_str.append(f'    {repr(k)}: {repr(v)},')
                info_str.append('},')
            if self.call_kwargs:
                info_str.append('call_kwargs={')
                for k, v in self.call_kwargs.items():
                    info_str.append(f'    {repr(k)}: {repr(v)},')
                info_str.append('},')
        else:
            if self.args:
                info_str.append(f'*{str(self.args)},')
            if self.kwargs:
                info_str.append(f'{str(self.kwargs)},')

        info_str.append(')')
        return '\n'.join(info_str)

    def _validate_parameter_usage(self, kwargs, init_kwargs, call_kwargs):
        if self.is_class:
            if kwargs:
                raise ValueError("Keyword arguments (**kwargs) are not allowed when passing a class. "
                                 "Use init_kwargs and call_kwargs instead.")
        else:
            if init_kwargs is not None:
                raise ValueError("init_kwargs is not allowed when passing a function. "
                                 "Use positional arguments (*args) instead.")
            if call_kwargs is not None:
                raise ValueError("call_kwargs is not allowed when passing a function. "
                                 "Use keyword arguments (**kwargs) instead.")

    def execute(self):
        if self.is_class:
            obj_instance = self.callable_target(*self.args, **self.init_kwargs)
            obj_instance(**self.call_kwargs)
        else:
            self.callable_target(*self.args, **self.kwargs)

    def _get_valid_signature_for_method(self, method: Callable) -> inspect.Signature:
        valid_params = []
        for i, (name, param) in enumerate(inspect.signature(method).parameters.items()):
            if i == 0:
                continue
            valid_params.append(inspect.Parameter(name=name,
                                                  kind=param.kind,
                                                  default=param.default,
                                                  annotation=param.annotation))
        return inspect.Signature(valid_params)

    def _get_arguments_not_set_in_kwargs_for_signature(self,
                                                       sig: inspect.Signature,
                                                       args: tuple,
                                                       kwargs: dict) -> ArgsDict:
        unset_args_dict = ArgsDict()
        func_params = sig.parameters
        arg_names = list(func_params)
        print(args)
        for arg_name in arg_names[len(args):]:
            if arg_name not in kwargs:
                if (func_params[arg_name].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD):
                    unset_args_dict[arg_name] = func_params[arg_name].default
        return unset_args_dict

    def _get_func_arguments_not_set_in_kwargs(self) -> ArgsDict:
        assert not self.is_class, ("args and kwargs do not exist when objects are passed in. "
                                   "Did you mean to call _get_empty_object_init_kwargs or "
                                   "_get_empty_object_call_kwargs?")
        return self._get_arguments_not_set_in_kwargs_for_signature(self.func_sig,
                                                                   self.args,
                                                                   self.kwargs)

    def _get_kwarg_names_without_default_values_for_signature(self,
                                                              sig: inspect.Signature,
                                                              args: tuple,
                                                              kwargs: dict) -> list:
        unset_args_dict = self._get_arguments_not_set_in_kwargs_for_signature(sig=sig,
                                                                              args=args,
                                                                              kwargs=kwargs)
        print("Calling _get_kwarg_names_without_default_values_for_signature")
        print(args)
        print(unset_args_dict)
        empty_kwargs = []
        for arg_name, arg_val in unset_args_dict.items():
            if arg_val is inspect.Parameter.empty:
                if (arg_name not in kwargs):
                    empty_kwargs.append(arg_name)
        return empty_kwargs

    def _get_funcs_empty_default_kwargs(self) -> list:
        assert not self.is_class, ("args and kwargs do not exist when objects are passed in. MORE TO WRITE")
        return self._get_kwarg_names_without_default_values_for_signature(self.func_sig,
                                                                          self.args,
                                                                          self.kwargs)

    def _validate_function(self) -> None:
        empty_kwargs = self._get_funcs_empty_default_kwargs()
        if empty_kwargs:
            err_msg = [f'For {self.callable_target.__name__}, the following arguments must be set:']
            err_msg.extend([f'    {arg_name}' for arg_name in empty_kwargs])
            raise RuntimeError("\n".join(err_msg))

    def _get_names_of_empty_object_kwargs_for_signature(self,
                                                        sig: inspect.Signature,
                                                        kwargs_to_skip: dict,
                                                        skip_first_n: int = 0) -> list:
        empty_args = []
        for argID, (arg_name, param) in enumerate(sig.parameters.items()):
            if argID < skip_first_n:
                continue
            if (arg_name not in kwargs_to_skip and arg_name != 'self' and
                    param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and
                    param.default is inspect.Parameter.empty):
                empty_args.append(arg_name)
        return empty_args

    def _validate_object(self):
        empty_init_kwargs = self._get_names_of_empty_object_kwargs_for_signature(sig=self.init_sig,
                                                                                 kwargs_to_skip=self.init_kwargs,
                                                                                 skip_first_n=len(self.args))
        empty_call_kwargs = self._get_names_of_empty_object_kwargs_for_signature(sig=self.call_sig,
                                                                                 kwargs_to_skip=self.call_kwargs,
                                                                                 skip_first_n=0)
        if empty_init_kwargs or empty_call_kwargs:
            err_msg = [f"For {self.callable_target.__name__}, the following arguments must be set:"]
            if empty_init_kwargs:
                err_msg.append("Initialization:")
                err_msg.extend(f"  {arg}" for arg in empty_init_kwargs)
            if empty_call_kwargs:
                err_msg.append("Calling:")
                err_msg.extend(f"  {arg}" for arg in empty_call_kwargs)
            raise RuntimeError("\n".join(err_msg))

    def can_potentially_run(self):
        if self.is_class:
            return (self._all_non_empty_strings(self.init_kwargs.values()) and
                    self._all_non_empty_strings(self.call_kwargs.values()))
        else:
            return (self._all_non_empty_strings(self.init_kwargs.values()) and
                    self._all_non_empty_strings(self.call_kwargs.values()))

    def _all_non_empty_strings(self, values):
        return all(val != '' for val in values)

    def _get_unset_function_args(self):
        assert not self.is_class, "Cannot get unset function arguments when a class is passed in."
        unset_args = ArgsDict()
        func_params = self.func_sig.parameters
        provided_args = set(self.kwargs.keys())

        for param_name, param in func_params.items():
            if (param_name not in provided_args and
                    param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD):
                unset_args[param_name] = param.default
        return unset_args

    def _get_unset_object_args(self,
                               sig: inspect.Signature,
                               kwargs_to_ignore: dict,
                               skip_first_n: int = 0):
        assert self.is_class, "Cannot get unset object arguments when a function is passed in."
        unset_args = ArgsDict()
        for argID, (arg_name, param) in enumerate(sig.parameters.items()):
            if argID < skip_first_n:
                continue
            if (arg_name not in kwargs_to_ignore and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD):
                unset_args[arg_name] = param.default
        return unset_args

    @classmethod
    def _get_args_not_set_in_kwargs(cls,
                                    sig: inspect.Signature,
                                    set_kwargs: dict | None = None,
                                    args_satisfied_by_positionals: int = 0,
                                    skip_self: bool = True):
        missing_kwargs = []
        full_params = list(sig.parameters.values())
        if not full_params:
            return missing_kwargs

        start_idx = 1 if (skip_self and full_params[0].name == 'self') else 0
        check_start_idx = start_idx + args_satisfied_by_positionals

        for i in range(check_start_idx, len(full_params)):
            param = full_params[i]
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.name not in set_kwargs:
                missing_kwargs.append(param)

        return missing_kwargs

    @classmethod
    def _get_missing_args(cls,
                          sig: inspect.Signature,
                          set_kwargs: dict | None = None,
                          args_satisfied_by_positionals: int = 0,
                          skip_self: bool = True):
        missing_args = cls._get_args_not_set_in_kwargs(sig=sig,
                                                       set_kwargs=set_kwargs,
                                                       args_satisfied_by_positionals=args_satisfied_by_positionals,
                                                       skip_self=skip_self)
        return [arg for arg in missing_args if arg.default is inspect.Parameter.empty]

    @classmethod
    def _get_default_args(cls,
                          sig: inspect.Signature,
                          set_kwargs: dict | None = None,
                          args_satisfied_by_positionals: int = 0,
                          skip_self: bool = True):
        missing_args = cls._get_args_not_set_in_kwargs(sig=sig,
                                                       set_kwargs=set_kwargs,
                                                       args_satisfied_by_positionals=args_satisfied_by_positionals,
                                                       skip_self=skip_self)
        return [arg for arg in missing_args if arg.default is not inspect.Parameter.empty]


class FunctionBasedStep(StepsAPI):
    """
    A step in a processing pipeline based on a callable function.

    This class allows for the execution of a given function with specified arguments and keyword arguments,
    validating that all mandatory parameters are provided.

    Attributes:
        name (str): The name of the step.
        function (Callable): The function to be executed in this step.
        args (tuple): Positional arguments to be passed to the function.
        kwargs (ArgsDict): Keyword arguments to be passed to the function.
        func_sig (inspect.Signature): The signature of the function for validating arguments.

    """
    def __init__(self, name: str, function: Callable, *args, **kwargs) -> None:
        """
        Initializes a function-based step in the processing pipeline.

        Args:
            name (str): The name of the step.
            function (Callable): The function to be executed in this step.
            *args: Positional arguments to be passed to the function.
            **kwargs: Keyword arguments to be passed to the function.
            
        """
        StepsAPI.__init__(self, name=name, skip_step=False)
        self.function = function
        self._func_name = function.__name__
        self.args = args
        self.kwargs = ArgsDict(kwargs)
        self.func_sig = inspect.signature(self.function)
        self.validate_kwargs_for_non_default_have_been_set()
    
    def get_function_args_not_set_in_kwargs(self) -> ArgsDict:
        """
        Retrieves arguments of the function that are not set in the keyword arguments.

        Returns:
            ArgsDict: A dictionary of function arguments that have not been set in the keyword arguments.
        """
        unset_args_dict = ArgsDict()
        func_params = self.func_sig.parameters
        arg_names = list(func_params)
        for arg_name in arg_names[len(self.args):]:
            if arg_name not in self.kwargs:
                if (func_params[arg_name].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD):
                    unset_args_dict[arg_name] = func_params[arg_name].default
        return unset_args_dict
    
    def get_empty_default_kwargs(self) -> list:
        """
        Identifies arguments that have not been provided and lack default values.

        Returns:
            list: A list of argument names that have no default values and are not provided.
        """
        unset_args_dict = self.get_function_args_not_set_in_kwargs()
        empty_kwargs = []
        for arg_name, arg_val in unset_args_dict.items():
            if arg_val is inspect.Parameter.empty:
                if arg_name not in self.kwargs:
                    empty_kwargs.append(arg_name)
        return empty_kwargs
    
    def validate_kwargs_for_non_default_have_been_set(self) -> None:
        """
        Validates that all mandatory arguments have been provided.

        Raises:
            RuntimeError: If any mandatory arguments are missing.
        """
        empty_kwargs = self.get_empty_default_kwargs()
        if empty_kwargs:
            unset_args = '\n'.join(empty_kwargs)
            raise RuntimeError(f"For {self._func_name}, the following arguments must be set:\n{unset_args}")
    
    def execute(self):
        """
        Executes the function with the provided arguments and keyword arguments.

        Raises:
            The function may raise any exceptions that its implementation can throw.
        """
        self.function(*self.args, **self.kwargs)

    def generate_kwargs_from_args(self) -> ArgsDict:
        """
        Converts positional arguments into keyword arguments.

        Returns:
            ArgsDict: A dictionary where positional arguments are mapped to their corresponding parameter names.
        """
        args_to_kwargs_dict = ArgsDict()
        for arg_name, arg_val in zip(list(self.func_sig.parameters), self.args):
            args_to_kwargs_dict[arg_name] = arg_val
        return args_to_kwargs_dict
    
    def __str__(self):
        """
        Returns a detailed string representation of the FunctionBasedStep instance.

        Returns:
            str: A string describing the step, including its name, function, arguments, and keyword arguments.
        """
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
    
    def __repr__(self):
        """
        Returns an unambiguous string representation of the FunctionBasedStep instance.

        Returns:
            str: A string representation showing how the FunctionBasedStep can be recreated.
        """
        cls_name = type(self).__name__
        full_func_name = f'{self.function.__module__}.{self._func_name}'
        info_str = [f'{cls_name}(', f'name={repr(self.name)},', f'function={full_func_name},']
        
        init_params = inspect.signature(self.__init__).parameters
        for arg_name in list(init_params)[2:-2]:
            info_str.append(f'{arg_name}={repr(getattr(self, arg_name))},')
        
        for arg_name, arg_val in zip(list(self.func_sig.parameters), self.args):
            info_str.append(f'{arg_name}={repr(arg_val)}', )
        
        for arg_name, arg_val in self.kwargs.items():
            info_str.append(f'{arg_name}={repr(arg_val)},')
        
        info_str.append(')')
        
        return f'\n    '.join(info_str)
    
    def all_args_non_empty_strings(self):
        """
        Checks if all positional arguments are non-empty strings.

        Returns:
            bool: True if all positional arguments are non-empty strings, False otherwise.
        """
        for arg in self.args:
            if arg == '':
                return False
        return True
    
    def all_kwargs_non_empty_strings(self):
        """
        Checks if all keyword arguments are non-empty strings.

        Returns:
            bool: True if all keyword arguments are non-empty strings, False otherwise.
        """
        for arg_name, arg in self.kwargs.items():
            if arg == '':
                return False
        return True
    
    def can_potentially_run(self):
        """
        Determines if the step can potentially be executed based on argument validation.
        Very simply checks if all arguments and keyword arguments are non-empty strings.

        Returns:
            bool: True if the step can potentially run, False otherwise.
        """
        return self.all_args_non_empty_strings() and self.all_kwargs_non_empty_strings()


class ObjectBasedStep(StepsAPI):
    """
    A step in a processing pipeline that is based on instantiating and invoking methods on an object.

    This class allows for initialization and execution of a specified object with given arguments and keyword arguments,
    validating that all mandatory parameters are provided.

    Attributes:
        name (str): The name of the step.
        class_type (type): The class type to be instantiated in this step.
        init_kwargs (ArgsDict): Keyword arguments for initializing the class.
        call_kwargs (ArgsDict): Keyword arguments for invoking the class.
        init_sig (inspect.Signature): The initialization signature of the class for validating arguments.
        call_sig (inspect.Signature): The call signature of the class for validating arguments.

    """
    def __init__(self, name: str, class_type: type, init_kwargs: dict, call_kwargs: dict) -> None:
        """
        Initializes an object-based step in the processing pipeline.

        Args:
            name (str): The name of the step.
            class_type (type): The class type to be instantiated in this step.
            init_kwargs (dict): Keyword arguments for initializing the class.
            call_kwargs (dict): Keyword arguments for invoking the class.
        """
        StepsAPI.__init__(self, name=name, skip_step=False)
        self.name: str = name
        self.class_type: type = class_type
        self.init_kwargs: ArgsDict = ArgsDict(init_kwargs)
        self.call_kwargs: ArgsDict = ArgsDict(call_kwargs)
        self.init_sig: inspect.Signature = inspect.signature(self.class_type.__init__)
        self.call_sig: inspect.Signature = inspect.signature(self.class_type.__call__)
        self.validate_kwargs()
    
    def validate_kwargs(self):
        """
        Validates that all mandatory initialization and call arguments have been provided.

        Raises:
            RuntimeError: If any mandatory arguments are missing.
        """
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
        """
        Retrieves arguments of the signature that are not set in the keyword arguments.

        Args:
            sig (inspect.Signature): The signature of the function or method.
            kwargs (dict): The keyword arguments provided.

        Returns:
            dict: A dictionary of arguments that are not set in the keyword arguments.
        """
        unset_args_dict = ArgsDict()
        for arg_name, arg_val in sig.parameters.items():
            if arg_name not in kwargs and arg_name != 'self':
                if arg_val.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    unset_args_dict[arg_name] = arg_val.default
        return unset_args_dict
    
    def get_empty_default_kwargs(self, sig: inspect.Signature, set_kwargs: dict) -> list:
        """
        Identifies arguments that have not been provided and lack default values.

        Args:
            sig (inspect.Signature): The signature of the function or method.
            set_kwargs (dict): The keyword arguments provided.

        Returns:
            list: A list of argument names that have no default values and are not provided.
        """
        unset_kwargs = self.get_args_not_set_in_kwargs(sig=sig, kwargs=set_kwargs)
        empty_kwargs = []
        for arg_name, arg_val in unset_kwargs.items():
            if arg_val is inspect.Parameter.empty:
                if arg_name not in set_kwargs:
                    empty_kwargs.append(arg_name)
        return empty_kwargs
    
    def execute(self) -> None:
        """
        Instantiates the class and invokes it with the provided arguments.

        Raises:
            The function may raise any exceptions that its implementation can throw.
        """
        obj_instance = self.class_type(**self.init_kwargs)
        obj_instance(**self.call_kwargs)

    def __str__(self):
        """
        Returns a detailed string representation of the ObjectBasedStep instance.

        Returns:
            str: A string describing the step, including its name, class, initialization, and call arguments.
        """
        unset_init_args = self.get_args_not_set_in_kwargs(self.init_sig, self.init_kwargs)
        unset_call_args = self.get_args_not_set_in_kwargs(self.call_sig, self.call_kwargs)
        
        info_str = [f'({type(self).__name__} Info):', f'Step Name: {self.name}',
                    f'Class Name: {self.class_type.__name__}', 'Initialization Arguments:', f'{self.init_kwargs}',
                    'Default Initialization Arguments:', f'{unset_init_args if unset_init_args else "N/A"}',
                    'Call Arguments:', f'{self.call_kwargs if self.call_kwargs else "N/A"}', 'Default Call Arguments:',
                    f'{unset_call_args if unset_call_args else "N/A"}']
        return '\n'.join(info_str)
    
    def __repr__(self):
        """
        Returns an unambiguous string representation of the ObjectBasedStep instance.

        Returns:
            str: A string representation showing how the ObjectBasedStep can be recreated.
        """
        cls_name = type(self).__name__
        full_func_name = f'{self.class_type.__module__}.{self.class_type.__name__}'
        info_str = [f'{cls_name}(', f'name={repr(self.name)},', f'class_type={full_func_name},']
        
        if self.init_kwargs:
            info_str.append('init_kwargs={')
            for arg_name, arg_val in self.init_kwargs.items():
                info_str.append(f'    {arg_name}={repr(arg_val)},')
            info_str[-1] = f'{info_str[-1]}' + '}'
        
        if self.call_kwargs:
            info_str.append('call_kwargs={')
            for arg_name, arg_val in self.call_kwargs.items():
                info_str.append(f'    {arg_name}={repr(arg_val)},')
            info_str[-1] = f'{info_str[-1]}' + '}'
        
        info_str.append(')')
        
        return f'\n    '.join(info_str)
    
    def all_init_kwargs_non_empty_strings(self):
        """
        Checks if all initialization keyword arguments are non-empty strings.

        Returns:
            bool: True if all initialization keyword arguments are non-empty strings, False otherwise.
        """
        for arg_name, arg_val in self.init_kwargs.items():
            if arg_val == '':
                return False
        return True
    
    def all_call_kwargs_non_empty_strings(self):
        """
        Checks if all call keyword arguments are non-empty strings.

        Returns:
            bool: True if all call keyword arguments are non-empty strings, False otherwise.
        """
        for arg_name, arg_val in self.call_kwargs.items():
            if arg_val == '':
                return False
        return True
    
    def can_potentially_run(self):
        """
        Determines if the step can potentially be executed based on argument validation.
        Very simply checks if all __init__ and __call__  keyword arguments for the object are
        non-empty strings.
        
        Returns:
            bool: True if the step can potentially run, False otherwise.
        """
        return self.all_init_kwargs_non_empty_strings() and self.all_call_kwargs_non_empty_strings()

