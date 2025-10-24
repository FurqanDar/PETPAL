"""Run regional statistics on PET Images"""
import argparse
from ..utils.stats import RegionalStats
from ..utils.image_io import write_dict_to_json

def main():
    """Run selected stats and write to file."""
    prog_desc = "Run statistics on a PET image for each region and write results to JSON file."
    epilog = "petpal-pet-stats --input-image-path suvr.nii.gz --segmentation-path " \
             "aparc+aseg.nii.gz --label-map freesurfer --statistic mean --out-path mean_suvr.json"
    parser = argparse.ArgumentParser(prog="STATS CLI", description=prog_desc, epilog=epilog)
    parser.add_argument("-i",
                        "--input-image-path",
                        required=True,
                        help="Path to 3D PET parametric image, such as SUVR or Vt.")
    parser.add_argument("-s",
                        "--segmentation-path",
                        required=True,
                        help="Path to 3D segmentation image, such as aparc+aseg.")
    parser.add_argument("-l",
                        "--label-map",
                        required=True,
                        help="Label map for ROIs. Presets include freesurfer, "
                             "freesurfer_merge_lr, perlcyno, perlcyno_merge_lr.")
    parser.add_argument("-f",
                        "--statistic",
                        required=True,
                        help="The statistic to calculate.",
                        choices=['mean','std','median','max','min','nvox'])
    parser.add_argument("-o",
                        "--out-path",
                        required=True,
                        help="Path to save output as JSON file.")
    args = parser.parse_args()

    stats_obj = RegionalStats(input_image_path=args.input_image_path,
                              segmentation_image_path=args.segmentation_path,
                              label_map_option=args.label_map)
    stats_result = getattr(stats_obj, args.statistic.lower())
    write_dict_to_json(meta_data_dict=stats_result, out_path=args.out_path)
