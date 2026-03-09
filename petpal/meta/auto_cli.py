"""Wrappers for generating CLI functions from PETPAL callable objects"""
import argparse
import inspect
from pydoc import locate

class ParseKwargs(argparse.Action):
    """Action to parse keyword arguments."""
    def __call__(self, parser, namespace, values, option_string=None):
        for value in values:
            key, val = value.split('=')
            setattr(namespace, key.replace('-','_'), val)


def auto_cli(petpal_class: object):
    """Generate a command line interface for a PETPAL function
    
    Args:
        petpal_class (object): Class defined in PETPAL that can be instantiated without specifying
            __init__ arguments. Must contain function __call__ with a docstring.
    

        Example:
            
            .. code-block:: python

                import numpy as np
                from petpal.meta.auto_cli import auto_cli
                import external_func

                class my_class:
                    mri_img: ants.ANTsImage
                    pet_img: ants.ANTsImage

                    def my_func(self, **kwargs):
                        return external_func(self.mri_img, self.pet_img, **kwargs)

                    def __call__(self, mri_img_path, pet_img_path, out_img_path, **kwargs):
                        self.mri_img = ants.image_read(mri_img_path)
                        self.pet_img = ants.image_read(pet_img_path)
                        output_img = my_func(**kwargs)
                        ants.image_write(output_img, output_img)
                
                    def main():
                        # Creates CLI for class my_class
                        # __call__ args are interpreted as required arguments
                        # **kwargs is interpreted as optional keyword arguments
                        # usage: --mri-img-path [MRI_IMG_PATH] --pet-img-path [PET_IMG_PATH] --kwargs [kwarg1=val1 kwarg2=val2 ...]

                        auto_cli(petpal_class=my_class)

                    if __name__=='__main__':
                        main()
                    """
    parser = argparse.ArgumentParser(prog=petpal_class.__name__,
                                     description=petpal_class.__call__.__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)

    for _, v in inspect.signature(petpal_class.__call__).parameters.items():
        name = str(v)
        if name=='self':
            continue
        arg_and_type = name.split(': ')
        if len(arg_and_type)==2:
            arg_name = f'--{arg_and_type[0]}'.replace('_','-')
            arg_type = locate(arg_and_type[1])
            parser.add_argument(arg_name,type=arg_type,required=True)
        elif arg_and_type[0].startswith('**'):
            kwarg_name = arg_and_type[0].replace('**','--').replace('_','-')
            parser.add_argument(kwarg_name, nargs='*', action=ParseKwargs, required=False)
    args = parser.parse_args()
    arg_vals = vars(args)

    for arg in arg_vals:
        val = arg_vals[arg]
        if isinstance(val, dict):
            for kwarg in val:
                arg_vals[kwarg] = val[kwarg]
            arg_vals.pop(arg)


    init_class = petpal_class()
    init_class(**arg_vals)
