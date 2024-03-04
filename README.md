# mribrew

This repository contains scripts and tools for (1) converting raw DWI DICOM to NIfTI, (2) pre-processing raw DWI data and (3) running Mean Apparent Propagator MRI (MAPMRI) or Anatomically Constrained Tractography (ACT). Literally all you need is three DICOM files (T1, DWI (dir-AP), and DWI (dir-PA)) and you can run all these scripts in order to get to the final results.

## Getting Started

### Overview

#### Main scripts
- `mribrew_dcm2nifti.py`: Script to convert DICOM to NIfTI.
- `mribrew_dwi_processing.py`: Script to run for pre-processing the raw DWI data.
- `mribrew_dwi_mapmri.py`: Script to run MAPMRI analysis using processed DWI data.
- `mribrew_dwi_act.py`: Script to run ACT using processed DWI data.
- `mribrew_rsfmri_ebmconnectivity.py` (adding more fMRI functionality in future): Script to run for functional connectivity analysis based on EBM stages with DK and Schaefer atlases using processed RSfMRI data.

### Data Folder Structure

Follow the underlying folder structure for scripts to work without having to make modifications. 

For `mribrew_dcm2nifti.py`, you need DICOM files in the `data/dcm/`. The output will be put to `data/raw/` which will be used by `mribrew_dwi_processing.py`. The processed files will be put to `data/proc/`. Finally, `mribrew_dwi_mapmri.py` or `mribrew_dwi_act.py` will use these processed files and their output will be at `data/res/`.

```
data/
└── dcm/
    ├── sub-01/
    │   ├── Serie_03_t1_mprage_sag_p2_iso_1.0.zip
    │   ├── Serie_08_ep2d_diff_hardi_s2_pa.zip
    │   └── Serie_10_ep2d_diff_hardi_s2.zip
    ├── sub-02/
    │   └── ...
    └── ...
└── raw/
    ├── sub-01/
    │   └── anat/
    │       ├── T1w.json
    │       └── T1w.nii.gz
    │   └── dwi/
    │       ├── dir-AP_dwi.bval
    │       ├── dir-AP_dwi.bvec
    │       ├── dir-AP_dwi.json
    │       ├── dir-AP_dwi.nii.gz
    │       ├── dir-PA_dwi.bval
    │       ├── dir-PA_dwi.bvec
    │       ├── dir-PA_dwi.json
    │       └── dir-PA_dwi.nii.gz
    ├── sub-02/
    │   └── ...
    └── ...
└── proc/
    ├── sub-01/
    │   └── dwi/
    │       ├── eddy_corrected.nii.gz
    │       ├── brain_dwi_mask.nii.gz
    │       ├── gradChecked.bval
    │       └── gradChecked.bvec
    ├── sub-02/
    │   └── ...
    └── ...
└── res/
    ├── sub-01/
    │   └── mapmri/
    │       ├── sub-01/
    │       │   ├── sub-01_MSE.nii.gz
    │       │   ├── sub-01_QIV.nii.gz
    │       │   └── ...
    │       ├── sub-02/
    │       │   └── ...
    │       └── ...
    │   └── act/
    │       ├── sc_sift_1000000.csv
    │       └── tracks_sift_1000000.tck
    ├── sub-02/
    │   └── ...
    └── ...
```

Notes:
- Small delta and large delta are pre-defined in the `mribrew_mapmri.py` script - make sure to change that according to your dataset.
- `misc/`folder contains various files like `acqp.txt` and `dcm2nii_config.json` which you may need to change based on your data acquisition.

## Running the Analysis

Run the `mribrew_dwi_*.py` script from the root directory of the project to begin the analysis:

```bash
python mribrew_dwi_*.py
```

### Dependencies
Make sure you have all the necessary dependencies installed and the data is organized as per the above structure.

- FSL
- nibabel
- Nipype
- Graphviz
- MRtrix3
- DIPY
- nilearn

## Contribution

If you'd like to contribute to this project or have any questions, please open an issue or submit a pull request.