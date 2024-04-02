# %%
import os
import shutil
from mribrew.utils import colours

# Define paths
cwd = os.getcwd()
dwi_proc_path = os.path.join(cwd, 'data', 'proc', 'dwi_proc')
fs_l_path = '/home/toomas/fs/l'
fs_x_path = '/home/toomas/fs/x'
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

            # Check if there is already output, skip if so
            output_dir_path = os.path.join(output_path, sub_id, scan_date_sfx)
            if os.path.isdir(output_dir_path):
                print(colours.CYELLOW + f"Skipping copying as output already exists: {output_dir_path}" + colours.CEND)
                continue
            
            # Check if subject/scan is a directory
            if os.path.isdir(os.path.join(sub_path, scan_date_sfx)):
                # Extract scan date without the suffix (e.g., _1)
                scan_date = scan_date_sfx.split('_')[0]

                # Construct the corresponding path to freesurfer data
                fs_subject_folder = f'_subject_id_{sub_id}'
                fs_scan_folder = f'{sub_id}__{scan_date}'
                fs_subjectscan_folder = f'_subject_id_{sub_id}__{scan_date}'
                fs_l_full_path = os.path.join(fs_l_path, fs_subject_folder, 
                                            'freeschlongsmurf', fs_scan_folder)
                fs_x_full_path = os.path.join(fs_x_path, fs_subjectscan_folder, 
                                            'autorecon1', fs_scan_folder)
                
                # Check if freesurfer path exists
                if os.path.isdir(fs_l_full_path): 
                    fs_full_path = fs_l_full_path
                elif os.path.isdir(fs_x_full_path): 
                    fs_full_path = fs_x_full_path
                else: 
                    print(colours.CRED + f"No such Freesurfer directory:"
                          f"\n{fs_l_full_path}\n{fs_x_full_path}" + colours.CEND)
                    continue

                
                # Copy contents into existing directory
                os.makedirs(output_dir_path, exist_ok=True)
                shutil.copytree(fs_full_path, output_dir_path, dirs_exist_ok=True)
                print(colours.CGREEN + f"Data copied from {fs_full_path} to {output_dir_path}" + colours.CEND)
                    

print(colours.UBOLD + f"Copying complete." + colours.CEND)

# %%
