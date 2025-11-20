import inspect
from typing import Callable, Any
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

    @classmethod
    def from_default_params(cls, params: list[inspect.Parameter]):
        return cls({param.name: param.default for param in params})


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

        self._validate_parameter_usage(kwargs=kwargs, init_kwargs=init_kwargs, call_kwargs=call_kwargs)

        self.init_kwargs = ArgsDict(init_kwargs or {})
        self.call_kwargs = ArgsDict(call_kwargs or {})
        self.args = list(args) or []
        self.kwargs = ArgsDict(kwargs or {})
        if self.is_class:
            self.init_sig = inspect.signature(callable_target.__init__)
            self.call_sig = inspect.signature(callable_target.__call__)
        else:
            self.func_sig = inspect.signature(callable_target)
        self.validate()

    def validate(self):
        errors = []

        target_sig = self.init_sig if self.is_class else self.func_sig
        target_kwargs = self.init_kwargs if self.is_class else self.kwargs

        params_list = list(target_sig.parameters.values())
        start_idx = 1 if (self.is_class) else 0

        covered_params = params_list[start_idx: start_idx + len(self.args)]
        covered_names = [p.name for p in covered_params]

        collision = set(covered_names).intersection(set(target_kwargs.keys()))
        if collision:
            errors.append(
                    f"Collision detected: Arguments {list(collision)} were passed as both positional (*args) and "
                    f"keyword ({'**kwargs' if self.is_function else 'init_kwargs'}).")
            args_idx = [str(covered_names.index(arg)) for arg in collision]
            errors.append(f"Passed kwargs collide with passed args at idx: {', '.join(args_idx)}")

        if self.is_class:
            missing_init = self._get_missing_args(
                            sig=target_sig,
                            set_kwargs=target_kwargs,
                            args_satisfied_by_positionals=len(self.args),
                            skip_self=True
                            )
            missing_init = ArgsDict.from_default_params(missing_init)

            missing_call = self._get_missing_args(
                            sig=self.call_sig,
                            set_kwargs=self.call_kwargs,
                            args_satisfied_by_positionals=len(self.args),
                            skip_self=True
                            )
            missing_call = ArgsDict.from_default_params(missing_call)

            if missing_init:
                errors.append(f"Initialization missing: {list(missing_init.keys())}")
            if missing_call:
                errors.append(f"Call missing: {list(missing_call.keys())}")

        else:
            missing_func = self._get_missing_args(
                    sig=target_sig,
                    set_kwargs=target_kwargs,
                    args_satisfied_by_positionals=len(self.args),
                    skip_self=False
                    )
            missing_func = ArgsDict.from_default_params(missing_func)

            if missing_func:
                errors.append(f"Function missing: {list(missing_func.keys())}")

        if errors:
            target_name = self.callable_target.__name__
            raise RuntimeError(f"Step '{self.name}' (Target: {target_name}) cannot run:\n" + "\n".join(errors))

    def __str__(self):
        lines = []
        lines.extend(self._str_header())
        lines.extend(self._str_extra_info())  # <--- The main hook for subclasses
        lines.extend(self._str_args_info())
        lines.extend(self._str_defaults_info())
        lines.extend(self._str_footer())

        return "\n".join(filter(None, lines))

    def _str_header(self) -> list[str]:
        return [
            f"({type(self).__name__} Info):",
            f"Step Name: {self.name}"
            ]

    def _str_extra_info(self) -> list[str]:
        return []

    def _str_args_info(self) -> list[str]:
        info = []
        target_name = self.callable_target.__name__

        if self.is_class:
            info.append(f"Target Class: {target_name}")
            info.append(f"Initialization Arguments:\n{self.init_kwargs if self.init_kwargs else 'N/A'}")
            info.append(f"Call Arguments:\n{self.call_kwargs if self.call_kwargs else 'N/A'}")
        else:
            info.append(f"Target Function: {target_name}")

            func_params = list(self.func_sig.parameters)
            reconstructed = ArgsDict()
            for i, val in enumerate(self.args):
                name = func_params[i] if i < len(func_params) else f"arg_{i}"
                reconstructed[name] = val
            if reconstructed:
                info.append(f"Positional Arguments:\n{reconstructed}")

            info.append(f"Keyword Arguments:\n{self.kwargs}")

        return info

    def _str_defaults_info(self) -> list[str]:
        info = []

        if self.is_class:
            def_init = self._get_default_args(sig=self.init_sig,
                                              set_kwargs=self.init_kwargs,
                                              args_satisfied_by_positionals=len(self.args),
                                              skip_self=True)
            def_call = self._get_default_args(sig=self.call_sig,
                                              set_kwargs=self.call_kwargs,
                                              args_satisfied_by_positionals=0,
                                              skip_self=True)
            if def_init:
                info.append(f"Default Init Args:\n{ArgsDict.from_default_params(def_init)}")
            if def_call:
                info.append(f"Default Call Args:\n{ArgsDict.from_default_params(def_call)}")
        else:
            def_func = self._get_default_args(sig=self.func_sig,
                                              set_kwargs=self.kwargs,
                                              args_satisfied_by_positionals=len(self.args),
                                              skip_self=False)
            if def_func:
                info.append(f"Default Arguments:\n{ArgsDict.from_default_params(def_func)}")
        return info

    def _str_footer(self) -> list[str]:
        return []

    def __repr__(self):
        cls_name = type(self).__name__
        target_name = f'{self.callable_target.__module__}.{self.callable_target.__name__}'

        info_str = [f'{cls_name}(']
        info_str.append(f'\t{repr(self.name)},')
        info_str.append(f'\t{target_name},')

        if self.is_class:
            if self.args:
                info_str.append(f'*{repr(tuple(self.args))},')
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
                info_str.append(f'*{repr(tuple(self.args))},')
            if self.kwargs:
                info_str.append(f'{repr(self.kwargs)},')

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

    def can_potentially_run(self):
        if self.is_class:
            return (self._all_non_empty_strings(self.init_kwargs.values()) and
                    self._all_non_empty_strings(self.call_kwargs.values()))
        else:
            return (self._all_non_empty_strings(self.init_kwargs.values()) and
                    self._all_non_empty_strings(self.call_kwargs.values()))

    def _all_non_empty_strings(self, values):
        return all(val != '' for val in values)

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


class PositionalBinder:
    def __init__(self, index: int):
        self.index = index

    def __get__(self, instance, owner):
        if instance is None: return self
        try:
            return instance.args[self.index]
        except IndexError:
            return None

    def __set__(self, instance, value):
        while len(instance.args) <= self.index:
            instance.args.append(None)
        instance.args[self.index] = value


class ImageToImageStepV2(BaseProcessingStep):
    input_image_path = PositionalBinder(0)
    output_image_path = PositionalBinder(1)

    def __init__(self,
                 name: str,
                 callable_target: Callable,
                 input_image_path: str,
                 output_image_path: str,
                 *args,
                 init_kwargs: dict = None,
                 call_kwargs: dict = None,
                 **kwargs):
        BaseProcessingStep.__init__(self,
                                    name,
                                    callable_target,
                                    *(input_image_path, output_image_path, *args),
                                    init_kwargs=init_kwargs,
                                    call_kwargs=call_kwargs,
                                    **kwargs)

    def _str_extra_info(self) -> list[str]:
        return [
            "Input & Output Paths:",
            f"\tInput:  {repr(self.input_image_path)}",
            f"\tOutput: {repr(self.output_image_path)}"
            ]


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

