import os
import subprocess
import shutil

from mribrew.utils import (colours, replace_special_chars, unzip_dcm, revert_replaced_chars_in_filedirs)

cwd = os.getcwd()
misc_dir = os.path.join(cwd, 'misc')
data_dir = os.path.join(cwd, 'data')
dcm_zip_dir = os.path.join(data_dir, 'dcm') # replace with actual dcm files location
raw_dir = os.path.join(data_dir, 'raw')

# Create temporary worklow directory
wf_dir = os.path.join(cwd, 'wf', 'dcm2nii')
os.makedirs(wf_dir, exist_ok=True)
dcm_temp_dir = os.path.join(wf_dir, 'dcm_unzip_temp')

# Names of the dcm files
dcm_fnames = ['Serie_03_t1_mprage_sag_p2_iso_1.0.zip',
              'Serie_10_ep2d_diff_hardi_s2.zip',
              'Serie_08_ep2d_diff_hardi_s2_pa.zip']

# Config file (info about dcm files) location
config_fname = os.path.join(misc_dir, 'dcm2nii_config.json')

# List all subjects in the dcm directory (replace with actual list if needed)
sub_list = next(os.walk(dcm_zip_dir))[1]

# Loop serially through all subjects in the subject list
for i, sub in enumerate(sub_list):
    print(colours.CBLUE + f'Converting DICOM to NIfTI for {sub} ({i+1}/{len(sub_list)})...\n' + colours.CEND)
    # Convert the subject name to alnum temporarily (required by BIDS command)
    sub_alnum = replace_special_chars(sub, to_replace='_', replace_with='xxx')
    sub_mapped = {'sub-'+sub_alnum : sub}

    # Loop through all dcm zips
    for dcm_fname in dcm_fnames:
        try:
            # Extract dcm zip to temporary directory
            sub_dcm_zip_dir = os.path.join(dcm_zip_dir, sub, dcm_fname)
            sub_dcm_dir = unzip_dcm(sub_dcm_zip_dir, os.path.join(dcm_temp_dir, sub))

            # Convert dcm to nifti and place according to BIDS (using config file)
            subprocess.run([
                'dcm2bids', '-d', sub_dcm_dir, '-p', sub_alnum, '-c', 
                config_fname, '-o', raw_dir
            ])
            print(colours.CGREEN + f'{dcm_fname} converted.\n' + colours.CEND)
        except:
            print(colours.CYELLOW + f'{dcm_fname} not found. Continuing...\n' + colours.CEND)
            continue
    
    # Revert alphanumeric subject names back to original
    revert_replaced_chars_in_filedirs(raw_dir, sub_mapped)
    print(colours.CGREEN + 'Directories and file names changed after temporary name changes.' + colours.CEND)

# Delete unzipper dcm folder (to save space) & move logs to wf folder
shutil.rmtree(dcm_temp_dir)
if os.path.isdir(wf_dir): shutil.rmtree(wf_dir)
_ = shutil.move(os.path.join(raw_dir, 'tmp_dcm2bids'), wf_dir, copy_function=shutil.copytree)
print(colours.UBOLD + 'ALL DONE! Unzipped DICOM folders deleted and logs moved to workflow directory.' + colours.CEND)