# %%
import os
import shutil
from mribrew.utils import colours

# Define paths
cwd = os.getcwd()
dwi_proc_path = os.path.join(cwd, 'data', 'proc', 'dwi_proc')
fs_path = '/Users/to8050an/Documents/Data/fs_dwi_2_subs' # '/home/toomas/fs/l'
output_path = os.path.join(cwd, 'data', 'proc', 'freesurfer')

ignore_dirs = ['.DS_Store']

# Iterate through the subjects in the processed DWI data
for i, sub_id in enumerate(os.listdir(dwi_proc_path)):
    if sub_id in ignore_dirs: continue
    print(colours.CBLUE + f"{sub_id} ({i+1}/{len(os.listdir(dwi_proc_path))})" + colours.CEND)

    # Check if subject is a directory
    sub_path = os.path.join(dwi_proc_path, sub_id)
    if os.path.isdir(sub_path):

        # Iterate through the subject's scans
        for j, scan_date_sfx in enumerate(os.listdir(sub_path)):
            if scan_date_sfx in ignore_dirs: continue
            print(colours.CBLUE + f"{scan_date_sfx} ({j+1}/{len(os.listdir(sub_path))})" + colours.CEND)
            
            # Check if subject/scan is a directory
            if os.path.isdir(os.path.join(sub_path, scan_date_sfx)):
                # Extract scan date without the suffix (e.g., _1)
                scan_date = scan_date_sfx.split('_')[0]

                # Construct the corresponding path to freesurfer data
                fs_subject_folder = f'_subject_id_{sub_id}'
                fs_scan_folder = f'{sub_id}__{scan_date}'
                fs_full_path = os.path.join(fs_path, fs_subject_folder, 
                                            'freeschlongsmurf', fs_scan_folder)
                
                # Check if this freesurfer path exists
                if os.path.isdir(fs_full_path):
                    # Construct the output path and copy the directory
                    output_dir_path = os.path.join(output_path, sub_id, scan_date_sfx)
                    os.makedirs(output_dir_path, exist_ok=True)
                    # Copy contents into existing directory
                    shutil.copytree(fs_full_path, output_dir_path, dirs_exist_ok=True)
                    print(colours.CGREEN + f"Data copied from {fs_full_path} to {output_dir_path}" + colours.CEND)
                else:
                    print(colours.CYELLOW + f"No such Freesurfer directory: {fs_full_path}" + colours.CEND)

print(colours.UBOLD + f"Copying complete." + colours.CEND)

# %%
