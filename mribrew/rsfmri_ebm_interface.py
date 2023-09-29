import numpy as np
import nibabel as nib
from nilearn import datasets, plotting
from nilearn.image import resample_to_img, load_img
from nilearn.connectome import ConnectivityMeasure
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def dice_coefficient(binary_img1, binary_img2):
    """Calculate Dice Coefficient for binary image data."""
    intersection = np.sum(binary_img1 * binary_img2)
    return 2. * intersection / (np.sum(binary_img1) + np.sum(binary_img2))

def jaccard_index(binary_img1, binary_img2):
    """Calculate Jaccard Index for binary image data."""
    intersection = np.sum(binary_img1 * binary_img2)
    union = np.sum(np.clip(binary_img1 + binary_img2, 0, 1))
    return intersection / union

def mutual_information(img1, img2):
    """Calculate Mutual Information for image data."""
    hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=20)
    return mutual_info_score(None, None, contingency=hist_2d)

def find_schaefer_at_ebm(ebm_stages_data, desikan_labels, desikan_img, schaefer_img, plot_style=None):
    # Co-register Schaefer atlas to Desikan-Killiany atlas
    resampled_schaefer_img = resample_to_img(schaefer_img, desikan_img, interpolation='nearest')
    
    # Extract image data
    desikan_data = desikan_img.get_fdata()
    schaefer_data = resampled_schaefer_img.get_fdata()

    overlap_dict = {}
    for stage, regions in ebm_stages_data.items():
        overlap_dict[stage] = []
        for region in regions:
            for hem in ['L', 'R']:
                # The region now includes hemisphere information, e.g. 'entorhinal_L'
                region_with_hem = f"{region}_{hem}"

                desikan_label_idx = desikan_labels.get(region_with_hem)
                if desikan_label_idx is not None:
                    overlapping_schaefer_labels = np.unique(schaefer_data[desikan_data == desikan_label_idx])
                    
                    # Remove the background label (typically 0)
                    overlapping_schaefer_labels = overlapping_schaefer_labels[overlapping_schaefer_labels != 0]
                    overlap_dict[stage].extend(overlapping_schaefer_labels)

    # Metrics calculation
    dsc = dice_coefficient((desikan_data > 0), (schaefer_data > 0))
    ji = jaccard_index((desikan_data > 0), (schaefer_data > 0))
    mi = mutual_information(desikan_data, schaefer_data)

    print(f"Dice Coefficient: {dsc:.4f}")
    print(f"Jaccard Index: {ji:.4f}")
    print(f"Mutual Information: {mi:.4f}")

    template_img = load_img(datasets.load_mni152_template(resolution=1))

    # Compute overlap data for plotting
    overlap_data = np.where(desikan_data > 0, 1, 0) + np.where(schaefer_data > 0, 2, 0)
    overlap_data = overlap_data.astype(np.int16)  # Ensure type compatibility
    overlap_img = nib.Nifti1Image(overlap_data, desikan_img.affine)

    colors = ['black', 'red', 'blue', 'magenta']
    cmap = ListedColormap(colors)
    labels = ['None', 'Desikan Only', 'Schaefer Only', 'Overlap']

    # Interactive Plot
    if plot_style=='interactive':
        view = plotting.view_img(overlap_img, bg_img=template_img, cmap=cmap, threshold=0.5, symmetric_cmap=False)
        view.open_in_browser()

    # Static Plots
    if plot_style=='static':
        plotting.plot_roi(overlap_img, title="Desikan and Schaefer Overlap", bg_img=template_img, cut_coords=(3, 5, 0), cmap=cmap, colorbar=True, vmin=0, vmax=3)
        for color, label in zip(colors, labels):
            plt.plot([0], [0], color=color, label=label)
        plt.legend()
        plt.show()

    return overlap_dict

def setup_ebm_rois():
    import abagen
    import pandas as pd
    import nibabel as nib
    from nilearn import datasets
    from mribrew.rsfmri_ebm_interface import find_schaefer_at_ebm

    # Load Desikan-Killiany atlas
    desikan_dict = abagen.fetch_desikan_killiany()
    desikan_img = nib.load(desikan_dict['image'])
    desikan_info = pd.read_csv(desikan_dict['info'])
    desikan_info['labelname'] = (desikan_info['label'] + "_" + desikan_info['hemisphere']).tolist()
    desikan_labels = {label: idx+1 for idx, label in enumerate(desikan_info['labelname'])}

    # Load Schaefer atlas
    schaefer_dict = datasets.fetch_atlas_schaefer_2018(n_rois=400) 
    schaefer_img = nib.load(schaefer_dict['maps'])
    schaefer_labels = {label.decode('utf-8'): float(idx+1) for idx, label in enumerate(schaefer_dict['labels'])}

    # EBM stages definition
    ebm_stages_data = {
        'ebm_I': ['entorhinal', 'amygdala', 'hippocampus'],
        'ebm_II': ['bankssts', 'fusiform', 'inferiortemporal', 'middletemporal', 'parahippocampal', 
                'superiortemporal', 'temporalpole'],
        'ebm_III': ['caudalmiddlefrontal', 'inferiorparietal', 'isthmuscingulate', 'lateraloccipital', 
                    'posteriorcingulate', 'precuneus', 'superiorparietal', 'supramarginal'],
        'ebm_IV': ['caudalanteriorcingulate', 'frontalpole', 'insula', 'lateralorbitofrontal', 
                'medialorbitofrontal', 'parsopercularis', 'parsorbitalis', 'parstriangularis', 
                'rostralanteriorcingulate', 'rostralmiddlefrontal', 'superiorfrontal'],
    }

    # Find and create dict of Schaefer ROIs at EBM stages
    ebm_rois_schaefer_dict = find_schaefer_at_ebm(ebm_stages_data, desikan_labels, desikan_img, schaefer_img)

    # Create a copy of Schaefer ROIs at different EBM stages
    ebm_rois_desikan_schaefer_dict = ebm_rois_schaefer_dict.copy()

    # Get all EBM I ROIs and add indeces of these ROIs to list
    new_ebm_I_indices = []
    for region in ebm_stages_data['ebm_I']:
        # Get label names and indeces from Desikan-Killiany atlas dict
        for label, idx in desikan_labels.items():
            # If the ROI is a substring of the label, then add to list
            if region in label:
                new_ebm_I_indices.append(idx)

    # Replace the EBM I ROIs with Desikan-Killiany atlas indeces
    ebm_rois_desikan_schaefer_dict['ebm_I'] = new_ebm_I_indices

    return ebm_rois_desikan_schaefer_dict, desikan_img, schaefer_img, desikan_labels, schaefer_labels

def plot_template_ebm_regions(ebm_time_series, desikan_img, schaefer_img, plot_style='static'):
    template_img = load_img(datasets.load_mni152_template(resolution=1))

    for stage, regions in ebm_time_series.items():
        # Based on the current stage, choose the atlas (EBM I = DK; else = Schaefer)
        current_atlas_img = desikan_img if stage=='ebm_I' else schaefer_img
        atlas_data = current_atlas_img.get_fdata()

        # Create a mask for regions in the current EBM stage and create an image
        mask = np.zeros_like(atlas_data)
        for idx, region in enumerate(regions.keys(), start=1):
            mask[atlas_data == region] = idx
        mask_img = nib.Nifti1Image(mask, current_atlas_img.affine)

        plot_title = f"{stage} (Desikan-Killiany)" if stage=='ebm_I' else f"{stage} (Schaefer 2018)"

        # Plot static plot for the stage regions on MNI
        if plot_style=='static':
            plotting.plot_stat_map(mask_img, colorbar=True, title=plot_title, bg_img=template_img, cmap='tab20')
            plotting.show()

        # Plot interactive plot for the stage regions on MNI
        if plot_style=='interactive':
            view = plotting.view_img(mask_img, title=plot_title, bg_img=template_img, cmap='tab20')
            view.open_in_browser()

def extract_ebm_timeseries(rsfmri_file, desikan_img, schaefer_img, desikan_labels, schaefer_labels,
                           ebm_rois_desikan_schaefer_dict):
    import nibabel as nib
    from nilearn.image import resample_to_img
    from nilearn.maskers import NiftiLabelsMasker

    # Load fMRI data
    rsfmri_img = nib.load(rsfmri_file)
    # Resample the fMRI data to match the Schaefer atlas resolution and space
    resampled_rsfmri_img = resample_to_img(rsfmri_img, schaefer_img, interpolation='nearest')

    # Loop through all the EBM stages and use the corresponding ROI indeces to extract ROI time series
    ebm_time_series = {}
    for stage, roi_indices in ebm_rois_desikan_schaefer_dict.items():
        if stage == 'ebm_I':
            # Adjust the masker to use the Desikan atlas and extract all time series data from fmri img
            masker = NiftiLabelsMasker(labels_img=desikan_img, standardize="zscore_sample")#, memory="nilearn_cache")
            all_time_series = masker.fit_transform(rsfmri_img)
        else:
            # Adjust the masker to use Schaefer atlas and extract all time series from resampled fmri img
            masker = NiftiLabelsMasker(labels_img=schaefer_img, standardize="zscore_sample")#, memory="nilearn_cache")
            all_time_series = masker.fit_transform(resampled_rsfmri_img)
        
        # Loop through all ROIs and extract time series of the ones relevant for the current stage
        ebm_time_series[stage] = {}
        for idx in roi_indices:
            ebm_time_series[stage][idx] = all_time_series[:, int(idx)-1]

    # Replace the label indeces with region label names for each stage
    ebm_time_series_labelled = {}
    for stage, stage_data in ebm_time_series.items():
        # Loop through all regions within the EBM stage
        labelname_data = {}
        for idx, time_series in stage_data.items():
            # For EBM I, use Desikan-Killiany atlas labels
            if stage == 'ebm_I':
                labelname_idx = [key for key, value in desikan_labels.items() if value == idx][0]
            # For all else, use Schaefer atlas labels
            else:
                labelname_idx = [key for key, value in schaefer_labels.items() if value == idx][0]
            # Merge the time series data with region label
            labelname_data[labelname_idx] = time_series
        # Add all EBM stage regions and their time series to to one dict
        ebm_time_series_labelled[stage] = labelname_data
    
    return ebm_time_series, ebm_time_series_labelled

def compute_connectivity(ebm_stage_data, kind):
    import numpy as np
    from nilearn.connectome import ConnectivityMeasure
    # Convert dictionary values to a list of time series and then transpose
    time_series_data = np.array(list(ebm_stage_data.values())).T

    connectome_measure = ConnectivityMeasure(kind=kind)
    connectivity_matrix = connectome_measure.fit_transform([time_series_data])[0]
    
    return connectivity_matrix

def drop_intrahemispheric_connections(matrix, stage_rois):
    matrix_hemi = matrix.copy()
    n = matrix_hemi.shape[0]
    for i in range(n):
        for j in range(n):
            if ('_LH_' in stage_rois[i] and '_LH_' in stage_rois[j]) or \
            ('_RH_' in stage_rois[i] and '_RH_' in stage_rois[j]) or \
            (stage_rois[i].endswith('_L') and stage_rois[j].endswith('_L')) or \
            (stage_rois[i].endswith('_R') and stage_rois[j].endswith('_R')):
                matrix_hemi[i, j] = 0  # Zero out the intra-hemispheric connections
    return matrix_hemi

def compute_corrmatrices(ebm_time_series_labelled):
    import os
    from mribrew.data_io import save_to_pickle
    from mribrew.rsfmri_ebm_interface import (compute_connectivity, drop_intrahemispheric_connections)
    # Compute connectivity (correlation and partial correlation) for each EBM stage
    sub_corrmatrices = {}
    sub_partial_corrmatrices = {}
    for stage, data in ebm_time_series_labelled.items():
        sub_corrmatrices[stage] = compute_connectivity(data, 'correlation')
        sub_partial_corrmatrices[stage] = compute_connectivity(data, 'partial correlation')
    
    sub_hemi_corrmatrices = {}
    sub_hemi_partial_corrmatrices = {}
    for stage in ebm_time_series_labelled.keys():
        stage_rois = list(ebm_time_series_labelled[stage].keys())
        sub_hemi_corrmatrices[stage] = drop_intrahemispheric_connections(sub_corrmatrices[stage], stage_rois)
        sub_hemi_partial_corrmatrices[stage] = drop_intrahemispheric_connections(sub_partial_corrmatrices[stage], stage_rois)

    files_dict = {
        "./sub_ebm_time_series_labelled.pkl": ebm_time_series_labelled,
        "./sub_corrmatrices.pkl": sub_corrmatrices,
        "./sub_partial_corrmatrices.pkl": sub_partial_corrmatrices,
        "./sub_hemi_corrmatrices.pkl": sub_hemi_corrmatrices,
        "./sub_hemi_partial_corrmatrices.pkl": sub_hemi_partial_corrmatrices
    }

    for fname, data in files_dict.items():
        save_to_pickle(data, fname)

    return tuple(os.path.abspath(fname) for fname, _ in files_dict.items())

def aggregate_matrices(subject_list, ebm_rois_desikan_schaefer_dict, 
                       sub_corrmatrices_file, sub_partial_corrmatrices_file, 
                       sub_hemi_corrmatrices_file, sub_hemi_partial_corrmatrices_file):
    import numpy as np
    import os
    from mribrew.data_io import read_pickle, save_to_pickle
    
    # Loop through subject_list and check if there is any path with sub's substring, if there is print message "found corr for sub"
    all_corrmatrices = {stage: [] for stage in ebm_rois_desikan_schaefer_dict.keys()}
    all_partial_corrmatrices = {stage: [] for stage in ebm_rois_desikan_schaefer_dict.keys()}
    all_hemi_corrmatrices = {stage: [] for stage in ebm_rois_desikan_schaefer_dict.keys()}
    all_hemi_partial_corrmatrices = {stage: [] for stage in ebm_rois_desikan_schaefer_dict.keys()}
    for i in range(len(subject_list)):
        print(f"Reading in {subject_list[i]} and {sub_corrmatrices_file[i]}...")
        sub_corrmatrices = read_pickle(sub_corrmatrices_file[i])
        sub_partial_corrmatrices = read_pickle(sub_partial_corrmatrices_file[i])
        sub_hemi_corrmatrices = read_pickle(sub_hemi_corrmatrices_file[i])
        sub_hemi_partial_corrmatrices = read_pickle(sub_hemi_partial_corrmatrices_file[i])

        # Append to all subjects' matrices
        for stage in ebm_rois_desikan_schaefer_dict.keys():
            all_corrmatrices[stage].append(sub_corrmatrices[stage])
            all_partial_corrmatrices[stage].append(sub_partial_corrmatrices[stage])
            all_hemi_corrmatrices[stage].append(sub_hemi_corrmatrices[stage])
            all_hemi_partial_corrmatrices[stage].append(sub_hemi_partial_corrmatrices[stage])

    # Convert to (N, N, P) for each stage
    for stage in ebm_rois_desikan_schaefer_dict.keys():
        all_corrmatrices[stage] = np.stack(all_corrmatrices[stage], axis=-1)
        all_partial_corrmatrices[stage] = np.stack(all_partial_corrmatrices[stage], axis=-1)
        all_hemi_corrmatrices[stage] = np.stack(all_hemi_corrmatrices[stage], axis=-1)
        all_hemi_partial_corrmatrices[stage] = np.stack(all_hemi_partial_corrmatrices[stage], axis=-1)

    # Save these all subjects' matrices to results folder
    all_corrmatrices_file = save_to_pickle(all_corrmatrices, "./all_corrmatrices.pkl")
    all_partial_corrmatrices_file = save_to_pickle(all_partial_corrmatrices, "./all_partial_corrmatrices.pkl")
    all_hemi_corrmatrices_file = save_to_pickle(all_hemi_corrmatrices, "./all_hemi_corrmatrices.pkl")
    all_hemi_partial_corrmatrices_file = save_to_pickle(all_hemi_partial_corrmatrices, "./all_hemi_partial_corrmatrices.pkl")

    return all_corrmatrices_file, all_partial_corrmatrices_file, all_hemi_corrmatrices_file, all_hemi_partial_corrmatrices_file