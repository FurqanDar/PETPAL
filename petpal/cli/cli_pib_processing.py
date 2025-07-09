import argparse
import petpal

_PIB_EXAMPLE_ = (r"""
Example:
    - Run a PIB scan through SUVR pipeline:
      petpal-pib-proc --sub 001 --ses 01 --anat-path ./sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz --seg-path ./derivatives/freesurfer/sub-001/ses-01/mri/wmparc.mgz --seg-table-path ./dseg.tsv --ref-region-label 8
                 
    - Only compute SUVR for the range of 30-60 min:
      petpal-pib-proc --sub 001 --ses 01 --anat-path ./sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz --seg-path ./derivatives/freesurfer/sub-001/ses-01/mri/wmparc.mgz --seg-table-path ./dseg.tsv --ref-region-label 8 --suvr-start 1800 --suvr-end 3600
          
""")

def main():
    parser = argparse.ArgumentParser(prog='petpal-pib-proc',
                                     description='Command line interface for running PIB processing.',
                                     epilog=_PIB_EXAMPLE_, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--sub',required=True,help='Subject or participant identifier. Note that this should not include "sub-", but only the value, i.e."--sub 001"')
    parser.add_argument('--ses',required=True,help='Session identifier. Note that this should not include "ses-", but only the value, i.e. "--ses 01"')
    parser.add_argument('--pet-path',required=True,help='Path to 4D pet image')
    parser.add_argument('--anat-path',required=True,help='Path to anatomical image')
    parser.add_argument('--seg-path',required=True,help='Path to segmentation image')
    parser.add_argument('--seg-table-path',required=True,help='Path to segmentation table file (see /shared/dseg.tsv in petpal for an example)')
    parser.add_argument('--ref-region-label',required=True,help='Integer label which corresponds to the reference region in the segmentation image')
    parser.add_argument('--suvr-start',required=False,default=0,help='Time (in seconds) to start SUVR calculation. Defaults to 0 (beginning of scan)')
    parser.add_argument('--suvr-end',required=False,default=-1,help='Time (in seconds) to end SUVR calculation. Default to -1 (the end of the scan)')
    parser.add_argument('--bids-root',required=False,default='.',help='Path to root of bids directory. Defaults to current path (".")')
    
    args = parser.parse_args()

    sub_id = args.sub
    ses_id = args.ses
    pet_path = args.pet_path
    seg_path = args.seg_path
    anat_path = args.anat_path
    bids_dir = args.bids_root
    seg_table_path = args.seg_table_path
    suvr_start = args.suvr_start
    suvr_end = args.suvr_end
    ref_region_label = args.ref_region_label

    PiB_Pipeline = petpal.pipelines.pipelines.BIDS_Pipeline(sub_id=sub_id,
                                                            ses_id=ses_id,
                                                            pipeline_name='petpal-pib-cli',
                                                            raw_anat_img_path=anat_path,
                                                            segmentation_img_path=seg_path,
                                                            bids_root_dir=bids_dir,
                                                            segmentation_label_table_path=seg_table_path)


    preproc_container = petpal.pipelines.steps_containers.StepsContainer(name='preproc')

    # Configure steps for preproc container
    thresh_crop_step = petpal.pipelines.preproc_steps.ImageToImageStep.default_threshold_cropping(input_image_path=PiB_Pipeline.pet_path)
    registration_step = petpal.pipelines.preproc_steps.ImageToImageStep.default_register_pet_to_t1(reference_image_path=PiB_Pipeline.anat_path,
                                                                                            half_life=petpal.utils.constants.HALF_LIVES['c11'])
    moco_step = petpal.pipelines.preproc_steps.ImageToImageStep.default_windowed_moco()
    write_tacs_step = petpal.pipelines.preproc_steps.TACsFromSegmentationStep.default_write_tacs_from_segmentation_rois(segmentation_image_path=PiB_Pipeline.seg_img,
                                                                                                    segmentation_label_map_path=PiB_Pipeline.seg_table)
    wss_step = petpal.pipelines.preproc_steps.ImageToImageStep(name='weighted_series_sum',
                                            function=petpal.utils.useful_functions.weighted_series_sum,
                                            input_image_path='',
                                            output_image_path='',
                                            half_life=petpal.utils.constants.HALF_LIVES['c11'],
                                            start_time=suvr_start,
                                            end_time=suvr_end)

    # Add steps to preproc container
    preproc_container.add_step(step=thresh_crop_step)
    preproc_container.add_step(step=registration_step)
    preproc_container.add_step(step=moco_step)
    preproc_container.add_step(step=write_tacs_step)
    preproc_container.add_step(step=wss_step)

    kinetic_modeling_container = petpal.pipelines.steps_containers.StepsContainer(name='km')

    # Configure steps for kinetic modeling container
    suvr_step = petpal.pipelines.preproc_steps.ImageToImageStep(name='suvr',
                                                                function=petpal.preproc.image_operations_4d.suvr,
                                                                input_image_path='',
                                                                output_image_path='',
                                                                ref_region=ref_region_label,
                                                                segmentation_image_path=seg_path,
                                                                verbose=False)

    # Add steps to kinetic modeling container
    kinetic_modeling_container.add_step(step=suvr_step)

    PiB_Pipeline.add_container(step_container=preproc_container)
    PiB_Pipeline.add_container(step_container=kinetic_modeling_container)

    PiB_Pipeline.add_dependency(sending='thresh_crop', receiving='windowed_moco')
    PiB_Pipeline.add_dependency(sending='windowed_moco', receiving='register_pet_to_t1')
    PiB_Pipeline.add_dependency(sending='register_pet_to_t1', receiving='write_roi_tacs')
    PiB_Pipeline.add_dependency(sending='register_pet_to_t1', receiving='weighted_series_sum')
    PiB_Pipeline.add_dependency(sending='weighted_series_sum', receiving='suvr')

    PiB_Pipeline.update_dependencies(verbose=True)

    PiB_Pipeline()
