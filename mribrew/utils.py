class colours:
    # Terminal color codes
    CEND = '\33[0m'
    CBLUE = '\33[94m'
    CGREEN = '\33[92m'
    CRED = '\33[91m'
    CYELLOW = '\33[93m'
    UBOLD = '\33[1m'

def unzip_dcm(dcm_fname, unzip_dir):
    """Unzip DICOM data and return the location."""
    import os, zipfile
    with zipfile.ZipFile(dcm_fname, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
    extracted_zip_name = dcm_fname.split('/')[-1].replace('.zip', '')
    extracted_zip_dir = os.path.join(unzip_dir, extracted_zip_name)
    return extracted_zip_dir

def replace_special_chars(subname, to_replace='_', replace_with='xxx'):
    """Replace specified characters in a string."""
    import re
    return re.sub(to_replace, replace_with, subname)

# def revert_replaced_chars_in_filedirs(base_dir, name_mapping):
#     """Reverse the replaced characters for subject names and change directory and file names."""
#     import os, shutil
#     for encoded_name, original_name in name_mapping.items():
#         encoded_path = os.path.join(base_dir, encoded_name)
#         original_path = os.path.join(base_dir, original_name)

#         # If the original path exists and is a directory, remove it to allow overwrite
#         if os.path.isdir(original_path): shutil.rmtree(original_path)

#         # Rename directories
#         if os.path.isdir(encoded_path):
#             os.rename(encoded_path, original_path)

#         # Recursively rename files within the directory
#         for root, dirs, files in os.walk(original_path):
#             for file in files:
#                 if encoded_name in file:
#                     original_file_name = file.replace(encoded_name+'_', '')
#                     os.rename(os.path.join(root, file), os.path.join(root, original_file_name))

def find_matching_filenames(directory, patterns):
    """Find pattern matching filenames in a directory"""
    import os, re
    return [f for f in os.listdir(directory) if any(re.search(pattern, f) for pattern in patterns)]

def revert_replaced_chars_in_filedirs(base_dir, name_mapping):
    """Reverse the replaced characters for subject names, change directory and file names, 
    and move contents up one level in the directory hierarchy."""
    import os
    import shutil

    for encoded_name, original_name in name_mapping.items():
        encoded_path = os.path.join(base_dir, encoded_name)

        # If the encoded path doesn't exist, it might have been moved up already
        if not os.path.exists(encoded_path):
            continue

        # Move contents up one level
        parent_dir = os.path.dirname(encoded_path)
        for item in os.listdir(encoded_path):
            src_path = os.path.join(encoded_path, item)
            dst_path = os.path.join(parent_dir, item)

            # Check if destination exists and remove if necessary
            if os.path.exists(dst_path):
                if os.path.isdir(dst_path):
                    shutil.rmtree(dst_path)
                else:
                    os.remove(dst_path)

            shutil.move(src_path, parent_dir)

        # Remove the now-empty encoded directory
        if os.path.isdir(encoded_path):
            os.rmdir(encoded_path)

        # Rename the directories and files in the parent directory
        for item in os.listdir(parent_dir):
            if encoded_name in item:
                original_item_name = item.replace(encoded_name + '_', '')
                os.rename(os.path.join(parent_dir, item), os.path.join(parent_dir, original_item_name))

        # Recursively rename files within the moved directories
        for root, dirs, files in os.walk(parent_dir):
            for file in files:
                if encoded_name in file:
                    original_file_name = file.replace(encoded_name + '_', '')
                    os.rename(os.path.join(root, file), os.path.join(root, original_file_name))

def move_contents(src, dst):
    """Move contents of src directory to dst directory."""
    import os
    import shutil

    if not os.path.exists(src):
        print(f"Source directory {src} does not exist. Skipping.")
        return

    # Create destination directory if it does not exist
    os.makedirs(dst, exist_ok=True)

    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dst_item = os.path.join(dst, item)

        # If the destination item exists and is a file, remove it before moving
        if os.path.isfile(dst_item):
            os.remove(dst_item)

        # Move the item
        shutil.move(src_item, dst_item)

    # Remove the now-empty source directory
    os.rmdir(src)

def split_subject_scan_list(subject_scan):
    subject_id, scan_id = subject_scan
    return subject_id, scan_id

def create_subject_scan_container(subject_scan):
    subject_id, scan_id = subject_scan
    return f"{subject_id}/{scan_id}"