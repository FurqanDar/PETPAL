"""Tools for running statistics on PET imaging data."""
from collections.abc import Callable
import numpy as np
import ants

from petpal.meta.label_maps import LabelMapLoader

class RegionalStats:
    """Run statistics on each region in a parametric 3D PET kinetic model or other image.
    
    :ivar pet_img: 3D PET image on which to get statistics for each region.
    :ivar seg_img: Segmentation image in same space as `pet_img` defining regions.
    :ivar stats_func: Callable function applied to voxels in PET image in each region.
    :ivar label_map: Dictionary that assigns labels to regions in `seg_img`."""
    def __init__(self,
                 input_image_path: str,
                 segmentation_image_path: str,
                 label_map_option: str | dict,
                 stats_func: Callable=np.mean):
        self.pet_img = ants.image_read(input_image_path)
        self.seg_img = ants.image_read(segmentation_image_path)
        self.stats_func = stats_func
        self.label_map = LabelMapLoader(label_map_option=label_map_option).label_map

    def get_voxels(self, label: str) -> np.ndarray:
        """Get the voxel array for the selected label.
        
        Args:
            label (str): Name of the region from which to extract voxels.
            
        Returns:
            voxel_arr (np.ndarray): Voxels in region as a flattened array."""
        mappings = self.label_map[label]
        region_mask = ants.mask_image(self.pet_img, self.seg_img, level=mappings)
        region_arr = region_mask.numpy().flatten()
        region_arr_nonzero = region_arr.nonzero()
        voxel_arr = region_arr[region_arr_nonzero]
        return voxel_arr
