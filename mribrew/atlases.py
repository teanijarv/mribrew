import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting, datasets
import abagen

def fetch_dk_atlas(coords=False):
    """Load Desikan-Killiany atlas' image and labels (and optionally coordinates)."""

    # dk_dict = abagen.fetch_desikan_killiany()
    # dk_img = nib.load(dk_dict['image'])
    # dk_info = pd.read_csv(dk_dict['info'])
    # dk_info['labelname'] = (dk_info['label'] + "_" + dk_info['hemisphere']).tolist()
    # dk_labels = {label: idx+1 for idx, label in enumerate(dk_info['labelname'])}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'misc', 'atlases', 
                             'fs_aparcaseg_dk', 'aparc+aseg_mni_relabel.nii.gz')
    dk_img = nib.load(file_path)
    dk_labels = fetch_mrtrix_fs_labels()

    if coords: 
        dk_coords = plotting.find_parcellation_cut_coords(dk_img)
        return dk_img, dk_labels, dk_coords

    return dk_img, dk_labels

def fetch_schaefer_atlas(n_rois=400, coords=False):
    """Load Schaefer 2018 atlas' image and labels (and optionally coordinates)."""

    schaefer_dict = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois) 
    schaefer_img = nib.load(schaefer_dict['maps'])
    schaefer_labels = {label.decode('utf-8'): float(idx+1) for idx, label in enumerate(schaefer_dict['labels'])}
    if coords: 
        schaefer_coords = plotting.find_parcellation_cut_coords(schaefer_img)
        return schaefer_img, schaefer_labels, schaefer_coords

    return schaefer_img, schaefer_labels

def fetch_mrtrix_fs_labels(original_labels=False, drop_unknown=True, drop_duplicate_thalamus=True):
    """
    Reads the fs_default.txt file containing FreeSurfer labels and returns a dictionary mapping label names to indexes.
    Optionally modifies the labels to a specified format.

    Parameters:
    modify_labels (bool): Whether to modify the label names to a specified format.
    drop_duplicate_thalamus (bool): Whether to drop the Thalamus label (same index as thalamus-proper)

    Returns:
    dict: A dictionary where the keys are label names and the values are indexes.
    """
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    replacements = {
        'ctx-lh-': 'L_', 'ctx-rh-': 'R_', 'Left-': 'L_', 'Right-': 'R_',
        '-Cortex': '', '-Proper': 'proper', '-area': 'area',
    }

    # Construct the path to the fs_default.txt file
    file_path = os.path.join(current_dir, '..', 'misc', 'fs_labels', 'fs_default.txt')

    # Open and read the file
    label_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Skip comments and empty lines
            if line.startswith('#') or not line.strip():
                continue
            # Split the line into components
            parts = line.split()
            if len(parts) >= 3:
                # Extract the index and label name
                index = int(parts[0])
                label_name = parts[2]
                if drop_unknown:
                    if label_name in ('Unknown'):
                        continue
                if drop_duplicate_thalamus:
                    if label_name in ('Left-Thalamus', 'Right-Thalamus'):
                        continue
                # Modify the label name if requested
                if not original_labels:
                    for old, new in replacements.items():
                        label_name = label_name.replace(old, new)
                    if label_name.startswith('L_'):
                        label_name = label_name[2:].lower() + '_L'
                    elif label_name.startswith('R_'):
                        label_name = label_name[2:].lower() + '_R'
                # Add to the dictionary
                label_dict[label_name] = index

    return label_dict
