"""Tools for running statistics on PET imaging data."""
from collections.abc import Callable
import numpy as np
import ants

class RegionalStats:
    """Run statistics on each region in a parametric 3D PET kinetic model or other image."""
    def __init__(self,
                 input_image_path: str,
                 segmentation_image_path: str,
                 stats_func: Callable=np.mean):
        self.pet_img = ants.image_read(input_image_path)
        self.seg_img = ants.image_read(segmentation_image_path)
        self.stats_func = stats_func
