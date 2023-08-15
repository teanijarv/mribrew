import os
import numpy as np
from dipy.io.image import load_nifti
from dipy.core.gradients import gradient_table

from mribrew.utils import colours, Tee, should_use_subset_data
from mribrew.mapmri_funcs import correct_neg_data, mapmri_fit_and_save, metric_correction_and_save

# MAPMRI parameters and other constants
big_delta, small_delta = 0.0353, 0.0150
mapmri_params = dict(radial_order=4,
                     laplacian_regularization=True,
                     laplacian_weighting=0.2,
                     positivity_constraint=False,
                     global_constraints=False,
                     pos_grid=15,
                     pos_radius='adaptive',
                     anisotropic_scaling=True,
                     eigenvalue_threshold=1e-04,
                     dti_scale_estimation=True,
                     static_diffusivity=0.7e-3,
                     cvxpy_solver=None)

# Define folder directories and structures
cwd = os.getcwd()
data_dir = os.path.join(cwd, 'data')
raw_dir = os.path.join(data_dir, 'raw')
proc_dir = os.path.join(data_dir, 'proc')
res_dir = os.path.join(data_dir, 'res')
subj_dirs = next(os.walk(proc_dir))[1]  # processed subjects

# Ask the user whether they want to use only small part of the data (for testing)
use_subset_data = should_use_subset_data()

# Create log file in results directory and export all console output
Tee(res_dir)

# Start of the pipeline
print(f"\n{colours.UBOLD}{colours.CYELLOW}Starting the MAPMRI pipeline...{colours.CEND}")
print(f"[Info] For changing any model parameters etc, please make changes inside the code.")

# Iterate over each subject, process data and save results
for i, cur_subj in enumerate(subj_dirs):
    print(f"\n{colours.CGREEN}Running MAPMRI on {cur_subj} ({i+1}/{len(subj_dirs)}){colours.CEND}")
    
    # Set file paths for DWI, mask and bval/bvec for the current subject
    cur_subj_dir = os.path.join(proc_dir, cur_subj, 'dwi')
    data_file = os.path.join(cur_subj_dir, 'eddy_corrected.nii.gz')
    mask_file = os.path.join(cur_subj_dir, 'brain_dwi_mask.nii.gz')
    bval_file = os.path.join(cur_subj_dir, 'gradChecked.bval')
    bvec_file = os.path.join(cur_subj_dir, 'gradChecked.bvec')
    
    # Load the data
    data, affine = load_nifti(data_file)
    print("The data [shape = %s] has been read." % str(data.shape))

    # Load the gradient information
    bvals, bvecs = np.loadtxt(bval_file), np.loadtxt(bvec_file).T
    gtab = gradient_table(bvals, bvecs, big_delta, small_delta)
    print("The gradient table created [bvals shape = %s; bvecs shape = %s; "
          "big delta = %s; small delta = %s]" % (gtab.bvals.shape, gtab.bvecs.shape,
                                                 gtab.big_delta, gtab.small_delta))
    
    # Load the mask and use it to exclude non-brain regions
    mask = load_nifti(mask_file)[0]
    data = data * mask[..., np.newaxis]
    print("The mask [shape = %s] has been applied to the data." % str(mask.shape))

    # (TESTING) Extract only a subset of the data to speed up the computation
    if use_subset_data:
        data = data[:, 40:50, 30:40]
        print("Using subset of the data [shape = %s]" % str(data.shape))

    # Averaging negative values of the data
    print("Looking for negative voxels in the data...")
    data = correct_neg_data(data)

    # Create results folder if doesn't exist
    res_mapmri_subj_dir = os.path.join(res_dir, 'mapmri', cur_subj)
    os.makedirs(res_mapmri_subj_dir, exist_ok=True)

    # Fit MAPMRI model to the data and save metric
    mapmri_fit_and_save(data=data, affine=affine, gtab=gtab, mapmri_params=mapmri_params,
                        bval_threshold=np.inf, metrics_to_save=['MSD', 'QIV', 'RTOP', 'RTAP', 'RTPP'],
                        cur_subj=cur_subj, res_mapmri_subj_dir=res_mapmri_subj_dir)
    
    # Correct RTOP metric values and save the corrected metric
    metric_correction_and_save(metric_name='RTOP', threshold=2000000, cur_subj=cur_subj,
                           res_mapmri_subj_dir=res_mapmri_subj_dir, correct_neg=True, replace_with=0)

print(f"\n{colours.UBOLD}{colours.CYELLOW}MAPMRI pipeline has finished.{colours.CEND}")