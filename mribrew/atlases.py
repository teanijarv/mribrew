import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting, datasets
import abagen

def fetch_dk_atlas(coords=False):
    """Load Desikan-Killiany atlas' image and labels (and optionally coordinates)."""

    dk_dict = abagen.fetch_desikan_killiany()
    dk_img = nib.load(dk_dict['image'])
    dk_info = pd.read_csv(dk_dict['info'])
    dk_info['labelname'] = (dk_info['label'] + "_" + dk_info['hemisphere']).tolist()
    dk_labels = {label: idx+1 for idx, label in enumerate(dk_info['labelname'])}
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
