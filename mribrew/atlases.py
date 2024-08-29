import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting, datasets
import abagen

def fetch_dk_atlas(coords=False, lut_idx=False):
    """Load Desikan-Killiany atlas' image and labels (and optionally coordinates)."""

    # load the DK image
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'misc', 'atlases', 
                             'fs_aparcaseg_dk', 'aparc+aseg_mni_relabel.nii.gz')
    dk_img = nib.load(file_path)
    
    # get labels as dict with values as either LUT IDs or just iterations
    if lut_idx:
        dk_labels = get_dk_fs_lut()
    else:
        dk_labels = fetch_fs_mrtrix_labels()

    # get coordinates of the labels
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

def fetch_fs_mrtrix_labels(original_labels=False, drop_unknown=True, drop_duplicate_thalamus=True):
    """
    Reads the fs_default.txt file containing FreeSurfer labels and returns a dictionary mapping label names to indexes.
    Optionally modifies the labels to a specified format.
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

def fetch_fs_lut_labels(original_labels=False):
    """
    Reads the FreeSurferColorLUT.txt file containing FreeSurfer labels and returns a dictionary mapping label names to indexes.
    Optionally modifies the labels to a specified format.
    """
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    replacements = {
        'ctx-lh-': 'L_', 'ctx-rh-': 'R_', 'Left-': 'L_', 'Right-': 'R_',
        '-Cortex': '', '-Proper*': 'proper', '-area': 'area',
    }

    # Construct the path to the FreeSurferColorLUT.txt file
    file_path = os.path.join(current_dir, '..', 'misc', 'fs_labels', 'FreeSurferColorLUT.txt')

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
                label_name = parts[1]

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

# def sort_left_first(item):
#     key, value = item
#     suffix_priority = 0 if key.endswith('_L') else 1
#     return (suffix_priority, value)

def sortby_ordered_dict(item, dict_ordered):
    key, _ = item
    return dict_ordered.get(key, float('inf'))

def get_dk_fs_lut(sort=True):
    
    fs_lut_dict = fetch_fs_lut_labels()
    # _, dk_labels = fetch_dk_atlas()
    dk_labels = fetch_fs_mrtrix_labels()

    for label in list(fs_lut_dict):
        if label not in list(dk_labels):
            del fs_lut_dict[label]
    
    if sort:
        # fs_lut_dict = dict(sorted(fs_lut_dict.items(), key=sort_left_first))
        fs_lut_dict = dict(sorted(fs_lut_dict.items(), key=lambda item: sortby_ordered_dict(item, dk_labels)))
    
    return fs_lut_dict