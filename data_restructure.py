import os
import shutil

cwd = os.getcwd()
data_dir = os.path.join(cwd, 'data')
proc_dir = os.path.join(data_dir, 'proc')

# Lists to store the different types of directories
fs_dir = os.path.join(proc_dir, "fs")
t1_dir = os.path.join(proc_dir, "t1")
rsfMRI_dir = os.path.join(proc_dir, "RSfMRI")

def create_and_move(source, dest_folder):
    if not os.path.exists(source):  # Check if the file exists
        print(f"Warning: File {source} not found.")
        return

    dest = os.path.join(dest_folder, os.path.basename(source))
    shutil.move(source, dest)

# Iterate over subjects in each directory
for subject in os.listdir(rsfMRI_dir):
    if not os.path.isdir(os.path.join(rsfMRI_dir, subject)):
        continue  # skip if not a directory

    subject_path = os.path.join(proc_dir, subject)

    # Create directories under each subject
    os.makedirs(os.path.join(subject_path, "fs"), exist_ok=True)
    os.makedirs(os.path.join(subject_path, "t1"), exist_ok=True)
    os.makedirs(os.path.join(subject_path, "RSfMRI"), exist_ok=True)

    create_and_move(os.path.join(fs_dir, subject, "aparc+aseg_reo.nii.gz"), os.path.join(subject_path, "fs"))
    create_and_move(os.path.join(t1_dir, subject, "t1_reo.nii.gz"), os.path.join(subject_path, "t1"))
    create_and_move(os.path.join(rsfMRI_dir, subject, "processed_and_censored_32bit.nii.gz"), os.path.join(subject_path, "RSfMRI"))

# Optionally, remove the old directories if they are empty
shutil.rmtree(fs_dir)
shutil.rmtree(t1_dir)
shutil.rmtree(rsfMRI_dir)
