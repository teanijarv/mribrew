def setup_atlases():
    import abagen
    import pandas as pd
    import nibabel as nib
    from nilearn import datasets

    # Load Desikan-Killiany atlas
    desikan_dict = abagen.fetch_desikan_killiany()
    desikan_img = nib.load(desikan_dict['image'])
    desikan_info = pd.read_csv(desikan_dict['info'])
    desikan_info['labelname'] = (desikan_info['label'] + "_" + desikan_info['hemisphere']).tolist()
    #desikan_labels = {label: idx+1 for idx, label in enumerate(desikan_info['labelname'])}

    # Load Schaefer atlas
    schaefer_dict = datasets.fetch_atlas_schaefer_2018(n_rois=400) 
    schaefer_img = nib.load(schaefer_dict['maps'])
    #schaefer_labels = {label.decode('utf-8'): float(idx+1) for idx, label in enumerate(schaefer_dict['labels'])}

    return desikan_img, schaefer_img

def extract_timeseries(rsfmri_file, desikan_img, schaefer_img):
    import nibabel as nib
    from nilearn.maskers import NiftiLabelsMasker

    # Load fMRI data
    rsfmri_img = nib.load(rsfmri_file)

    # Adjust the masker to use the Desikan atlas and extract all time series data from fmri img
    desikan_masker = NiftiLabelsMasker(labels_img=desikan_img, resampling_target='data', standardize="zscore_sample")
    desikan_time_series = desikan_masker.fit_transform(rsfmri_img)

    # Adjust the masker to use Schaefer atlas and extract all time series from fmri img
    schaefer_masker = NiftiLabelsMasker(labels_img=schaefer_img, resampling_target='data', standardize="zscore_sample")
    schaefer_time_series = schaefer_masker.fit_transform(rsfmri_img)
    
    return desikan_time_series, schaefer_time_series

def compute_connectivity(desikan_time_series, schaefer_time_series):
    import os
    import numpy as np
    # from nilearn.connectome import ConnectivityMeasure
    from mribrew.data_io import save_to_pickle

    # corr_measure = ConnectivityMeasure(kind='correlation', standardize='zscore_sample')
    # sub_corrmat_desikan = corr_measure.fit_transform([desikan_time_series])[0]
    # sub_corrmat_schaefer = corr_measure.fit_transform([schaefer_time_series])[0]

    # part_corr_measure = ConnectivityMeasure(kind='partial correlation', standardize='zscore_sample')
    # sub_partial_corrmat_desikan = part_corr_measure.fit_transform([desikan_time_series])[0]
    # sub_partial_corrmat_schaefer = part_corr_measure.fit_transform([schaefer_time_series])[0]

    sub_corrmat_desikan = np.corrcoef(desikan_time_series.T)
    sub_corrmat_schaefer = np.corrcoef(schaefer_time_series.T)

    files_dict = {
        "./sub_corrmat_desikan.pkl": sub_corrmat_desikan,
        # "./sub_partial_corrmat_desikan.pkl": sub_partial_corrmat_desikan,
        "./sub_corrmat_schaefer.pkl": sub_corrmat_schaefer,
        # "./sub_partial_corrmat_schaefer.pkl": sub_partial_corrmat_schaefer
    }

    for fname, data in files_dict.items():
        save_to_pickle(data, fname)

    return tuple(os.path.abspath(fname) for fname, _ in files_dict.items())