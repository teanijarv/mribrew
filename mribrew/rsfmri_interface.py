def get_bold_timeseries(bold_img, atlas_img, standardize=False):
    from nilearn.maskers import NiftiLabelsMasker
    atlas_masker = NiftiLabelsMasker(labels_img=atlas_img, resampling_target='data', standardize=standardize)
    bold_timeseries = atlas_masker.fit_transform(bold_img)
    return bold_timeseries

def extract_dk_sch_bold_timeseries(rsfmri_file):
    import nibabel as nib
    from mribrew.atlases import (fetch_dk_atlas, fetch_schaefer_atlas)
    from mribrew.rsfmri_interface import get_bold_timeseries

    # Load fMRI data and DK+Schaefer atlases
    rsfmri_img = nib.load(rsfmri_file)
    dk_img, _ = fetch_dk_atlas(coords=False)
    sch_img, _ = fetch_schaefer_atlas(n_rois=400, coords=False)

    # Get BOLD time series for both atlases
    dk_timeseries = get_bold_timeseries(rsfmri_img, dk_img, standardize="zscore_sample")
    sch_timeseries = get_bold_timeseries(rsfmri_img, sch_img, standardize="zscore_sample")
    
    return dk_timeseries, sch_timeseries

def compute_fc(dk_timeseries, sch_timeseries):
    import os
    import numpy as np
    from nilearn.connectome import ConnectivityMeasure
    from mribrew.data_io import save_csv

    fc_measure = ConnectivityMeasure(kind='correlation', standardize='zscore_sample')
    dk_fc = fc_measure.fit_transform([dk_timeseries])[0]
    sch_fc = fc_measure.fit_transform([sch_timeseries])[0]

    # dk_fc = np.corrcoef(dk_timeseries.T)
    # sch_fc = np.corrcoef(sch_timeseries.T)

    files_dict = {
        "./dk_fc.csv": dk_fc,
        "./sch_fc.csv": sch_fc,
    }

    for fname, data in files_dict.items():
        save_csv(data, fname)
    
    return tuple(os.path.abspath(fname) for fname, _ in files_dict.items())
