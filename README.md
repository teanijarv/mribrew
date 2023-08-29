# mribrew

This repository contains scripts and tools for pre-processing DWI data and analyzing it using MAPMRI. In future, there are plans to include more diffusion and functional MRI analysis techniques.

## Getting Started

### Overview

#### Main scripts
- `mribrew_dwi_processing.py`: Script to run for pre-processing the raw DWI data.
- `mribrew_dwi_mapmri.py`: Script to run for MAPMRI analysis using processed DWI data.

#### Functions
- `mribrew/data_io.py`: All functions for data input-output.
- `mribrew/utils.py`: All utility functions.
- `mribrew/mapmri_funcs.py`: Contains all the MAPMRI-related functions required by the main MAPMRI script.
- `mribrew/dwiproc_interface.py`: Contains the helping interface for the DWI pre-processing main script.

### Data Folder Structure

For the scripts to run correctly by default, the following folder structure and naming conventions are expected:

```
data/
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
```

- The main data directory should be named `data/proc/`.
- Each subject should have their own sub-directory within `data/proc/` (e.g., `sub-01`, `sub-02`).
- Within each subject's directory, there should be a `dwi/` folder containing the required files:
    - `eddy_corrected.nii.gz`: The eddy current corrected DWI data.
    - `brain_dwi_mask.nii.gz`: The brain mask for the DWI data.
    - `gradChecked.bval`: The gradient b-values.
    - `gradChecked.bvec`: The gradient b-vectors.
- Note: small delta and large delta are pre-defined in the `mribrew_mapmri.py` script - make sure to change that according to your dataset.

### Results Folder Structure

The resulting metrics from the MAPMRI analysis will be saved in the following default structure:

```
data/
└── res/
    └── mapmri/
        ├── sub-01/
        │   ├── sub-01_MSE.nii.gz
        │   ├── sub-01_QIV.nii.gz
        │   └── ...
        ├── sub-02/
        │   └── ...
        └── ...
```

- The main results directory is `data/res/mapmri/`.
- Each subject's results will be saved in their respective sub-directory.
- The naming convention for the metrics is, e.g., `sub-01_MSE.nii.gz` for the MSE metric of subject 01.

## Running the Analysis

Run the `mribrew_dwi_mapmri.py` script from the root directory of the project to begin the analysis:

```bash
python mribrew_dwi_mapmri.py
```

### Dependencies
Make sure you have all the necessary dependencies installed and the data is organized as per the above structure.

- FSL
- Nipype
- Graphviz
- MRtrix3
- DIPY

## Contribution

If you'd like to contribute to this project or have any questions, please open an issue or submit a pull request.

## References

[1] Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2.

[2] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
           diffusion imaging method for mapping tissue microstructure",
           NeuroImage, 2013.

[3] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
           using Laplacian-regularized MAP-MRI and its application to HCP
           data." NeuroImage (2016).
