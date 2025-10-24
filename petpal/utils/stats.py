"""Tools for running statistics on PET imaging data."""
from collections.abc import Callable
import numpy as np
import ants

from ..meta.label_maps import LabelMapLoader
from .useful_functions import check_physical_space_for_ants_image_pair

class RegionalStats:
    """Run statistics on each region in a parametric 3D PET kinetic model or other image.
    
    Options:
        * RegionalStats.mean: Mean value within each region
        * RegionalStats.std: Standard devation of values within each region.
        * RegionalStats.nvox: Number of voxels in each region.
        * RegionalStats.max: Maximum value in each region.
        * RegionalStats.min: Minimum value in each region.
        * RegionalStats.median: Median value in each region.
        * RegionalStats.get_stats(stats_func): Get a generic statistic run on each region. Runs
          function `stats_func` on each region, which must take a 1D array as the only argument.

    Example:

        .. code-block:: python

            import numpy as np
            from petpal.utils.stats import RegionalStats
            from petpal.utils.image_io import write_dict_to_json

            # Set up class
            input_image_path = 'sub-001_ses-01_space-mpr_desc-SUVR_pet.nii.gz'
            segmentation_image_path = 'sub-001_ses-01_seg.nii.gz'
            region_stats_obj = RegionalStats(input_image_path = input_image_path,
                                             segmentation_image_path = segmentation_image_path,
                                             label_map_option = 'freesurfer')
            
            # Preset statistic: get mean in each region
            region_means = region_stats_obj.mean
            write_dict_to_json(region_means,'sub-001_ses-01_RegionMeanSUVR.json')

            # Create function to get 95th percentile value for each region
            def calc_95th_percentile(arr: np.ndarray):
                return np.percentile(arr,95)
            region_95th = region_stats_obj.get_stats(calc_95th_percentile)
            write_dict_to_json(region_95th,'sub-001_ses-01_Region95thPercentileSUVR.json')

    :ivar pet_img: 3D PET image on which to get statistics for each region.
    :ivar seg_img: Segmentation image in same space as `pet_img` defining regions.
    :ivar label_map: Dictionary that assigns labels to regions in `seg_img`."""
    def __init__(self,
                 input_image_path: str,
                 segmentation_image_path: str,
                 label_map_option: str | dict):
        self.pet_img = ants.image_read(input_image_path)
        self.seg_img = ants.image_read(segmentation_image_path)
        assert check_physical_space_for_ants_image_pair(self.pet_img, self.seg_img), (
            "input image and anatomical image must occupy the same physical space")
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

    def get_stats(self, stats_func: Callable, dtype: object=float) -> dict:
        """Get stats for all regions. Applies `stats_func` to the `voxel_arr` to return a
        value for each region. Set `dtype` depending on what type the output is (e.g. float or
        int).
        
        Args:
            stats_func (Callable): The function to run on each region's voxels. Must take a 1D
              array as the only required positional argument and return a value or np.ndarray.
            dtype (object): The dtype that `stats_func` returns, if not float. Typically float or
              int. Default float.
        
        Returns:
            region_stats (dict): The statistics for each region of interest.
        """
        region_stats = {}
        for label in self.label_map:
            voxel_arr = self.get_voxels(label=label)
            region_stat = stats_func(voxel_arr)
            if isinstance(region_stat,np.ndarray):
                region_stats[label] = region_stat.astype(dtype).tolist()
            else:
                region_stats[label] = dtype(region_stat)
        return region_stats

    @property
    def mean(self) -> dict:
        """Get mean value for each region."""
        return self.get_stats(stats_func=np.mean)

    @property
    def std(self) -> dict:
        """Get standard deviation of values for each region."""
        return self.get_stats(stats_func=np.std)

    @property
    def nvox(self) -> dict:
        """Get number of voxels in each region."""
        return self.get_stats(stats_func=len, dtype=int)

    @property
    def max(self) -> dict:
        """Get maximum value in each region."""
        return self.get_stats(stats_func=np.max)

    @property
    def min(self) -> dict:
        """Get minimum value in each region."""
        return self.get_stats(stats_func=np.min)

    @property
    def median(self) -> dict:
        """Get median value in each region."""
        return self.get_stats(stats_func=np.median)
