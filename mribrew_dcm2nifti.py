# %%

# TO-DO
# - removal of scan (and in some cases subject) folders which end up empty (ie dont match the config sidecars)

import os
import re
import subprocess
import shutil

from mribrew.utils import (colours, replace_special_chars, unzip_dcm, move_contents,
                           find_matching_filenames, revert_replaced_chars_in_filedirs)

cwd = os.getcwd()
misc_dir = os.path.join(cwd, 'misc')
data_dir = os.path.join(cwd, 'data')
dcm_zip_dir = '/mnt/raid1/shares/nicolatoomas/dwi_dicom'
raw_dir = os.path.join(data_dir, 'raw')

# Create temporary worklow directory
wf_dir = os.path.join(cwd, 'wf', 'dcm2nii')
os.makedirs(wf_dir, exist_ok=True)
dcm_temp_dir = os.path.join(wf_dir, 'dcm_unzip_temp')

# Regular expression patterns for the dcm files
dcm_fpatterns = [r'Serie_\d+_ep2d_diff_hardi_s2\.zip', 
                 r'Serie_\d+_ep2d_diff_hardi_s2_pa\.zip']

# Config file (info about dcm files) location
config_fname = os.path.join(misc_dir, 'dcm2nii_config.json')

# List all subjects in the dcm directory (replace with actual list if needed)
sub_list = next(os.walk(dcm_zip_dir))[1]

# Loop serially through all subjects in the subject list
for i, sub in enumerate(sub_list):
    print(colours.CBLUE + f'\nConverting DICOM to NIfTI for {sub} ({i+1}/{len(sub_list)})...' + colours.CEND)
    # Convert the subject name to alnum temporarily (required by BIDS command)
    sub_alnum = replace_special_chars(sub, to_replace='_', replace_with='xxx')
    sub_mapped = {'sub-'+sub_alnum : sub}

    # Get subject directory and list of scans
    sub_dir, sub_scan_list, _ = next(os.walk(os.path.join(dcm_zip_dir, sub)))

    # Loop through all scans
    for j, scan in enumerate(sub_scan_list):
        print(colours.CBLUE + f'Scan {scan} ({j+1}/{len(sub_scan_list)})...' + colours.CEND)
        # Find matching dcm filenames for the current scan
        scan_dcm_dir = os.path.join(sub_dir, scan, 'DCM')
        try:
            dcm_fnames = find_matching_filenames(scan_dcm_dir, dcm_fpatterns)
            
            # Loop through all dcm zips
            for dcm_fname in dcm_fnames:
                try:
                    # Extract dcm zip to temporary directory
                    sub_dcm_zip_dir = os.path.join(scan_dcm_dir, dcm_fname)
                    sub_dcm_dir = unzip_dcm(sub_dcm_zip_dir, os.path.join(dcm_temp_dir, sub, scan))
                        
                    # Define the scan-specific output directory for BIDS conversion
                    scan_raw_dir = os.path.join(raw_dir, sub, scan)
                    os.makedirs(scan_raw_dir, exist_ok=True)

                    # Convert dcm to nifti and place according to BIDS (using config file)
                    subprocess.run([
                        'dcm2bids', '-d', sub_dcm_dir, '-p', sub_alnum, '-c', 
                        config_fname, '-o', scan_raw_dir
                    ])
                    print(colours.CGREEN + f'{dcm_fname} converted.\n' + colours.CEND)
                except Exception as e:
                    print(colours.CYELLOW + f'Error with {dcm_fname}: {e}. Continuing...\n' + colours.CEND)
                    continue

            # Revert alphanumeric subject names back to original and move logs to wf
            revert_replaced_chars_in_filedirs(scan_raw_dir, sub_mapped)
            move_contents(os.path.join(scan_raw_dir, 'tmp_dcm2bids'), os.path.join(wf_dir, sub, scan))
            print(colours.CGREEN + 'Directories and file names changed after temporary name changes.' + colours.CEND)
                
        except Exception as e:
            print(colours.CYELLOW + f'Error with finding matching file names in  {scan_dcm_dir}: {e}. Continuing...\n' + colours.CEND)
            continue

        
    # After each subject, delete their unzipped DCM folder (to save space)
    try:
        shutil.rmtree(dcm_temp_dir)
    except:
        continue

print(colours.UBOLD + 'ALL DONE!' + colours.CEND)
# %%
