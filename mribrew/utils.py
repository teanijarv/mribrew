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

def revert_replaced_chars_in_filedirs(base_dir, name_mapping):
    """Reverse the replaced characters for subject names and change directory and file names."""
    import os, shutil
    for encoded_name, original_name in name_mapping.items():
        encoded_path = os.path.join(base_dir, encoded_name)
        original_path = os.path.join(base_dir, original_name)

        # If the original path exists and is a directory, remove it to allow overwrite
        if os.path.isdir(original_path): shutil.rmtree(original_path)

        # Rename directories
        if os.path.isdir(encoded_path):
            os.rename(encoded_path, original_path)

        # Recursively rename files within the directory
        for root, dirs, files in os.walk(original_path):
            for file in files:
                if encoded_name in file:
                    original_file_name = file.replace(encoded_name+'_', '')
                    os.rename(os.path.join(root, file), os.path.join(root, original_file_name))