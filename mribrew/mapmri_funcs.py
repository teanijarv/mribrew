def correct_neg_data(data):
    import numpy as np

    # Mask of voxels with negative for each time point
    neg_mask = data<0

    # Count and print how many negative voxels there are
    num_neg_voxels = np.sum(neg_mask)
    if num_neg_voxels > 0:
        print(f"[Info] Found {num_neg_voxels} negative voxels in the data - replacing with "
               "average volume value for that timepoint.")

    # Loop through timepoints
    for t in range(data.shape[3]):
        # Extract the 3D volume at timepoint t
        volume_at_t = data[..., t]
        # Compute the mean of the positive values in the volume
        volume_at_t_mean = volume_at_t[volume_at_t>0].mean()
        # Apply the mean value to the voxels that have negative values
        volume_at_t[neg_mask[..., t]] = volume_at_t_mean
        # Assign the corrected volume back to the data
        data[..., t] = volume_at_t

    return data

def fit_mapmri_model(data, gtab, mapmri_params=None):
    import numpy as np
    from dipy.reconst.mapmri import MapmriModel

    # Default parameters for MAPMRI
    if mapmri_params is None:
        mapmri_params = dict(radial_order=6,
                             laplacian_regularization=True,
                             laplacian_weighting=0.2,
                             positivity_constraint=False,
                             global_constraints=False,
                             pos_grid=15,
                             pos_radius='adaptive',
                             anisotropic_scaling=True,
                             eigenvalue_threshold=1e-04,
                             bval_threshold=np.inf,
                             dti_scale_estimation=True,
                             static_diffusivity=0.7e-3,
                             cvxpy_solver=None)

    # Fit MAPMRI model to the data
    map_model = MapmriModel(gtab, **mapmri_params)
    mapfit = map_model.fit(data)

    # Extract metrics from the model
    MSD = mapfit.msd()
    QIV = mapfit.qiv()
    RTOP = mapfit.rtop()
    RTAP = mapfit.rtap()
    RTPP = mapfit.rtpp()

    return MSD, QIV, RTOP, RTAP, RTPP

def metrics_to_nifti(affine, MSD, QIV, RTOP, RTAP, RTPP, out_file_prefix, res_dir):
    from dipy.io.image import save_nifti
    from mribrew.utils import colours
    import os

    # Define the metrics and list for exporting directories
    metrics = [MSD, QIV, RTOP, RTAP, RTPP]
    metric_names = ["MSD", "QIV", "RTOP", "RTAP", "RTPP"]
    out_files = []

    # Save all the metrics as NIfTI files
    subject_dir = os.path.join(res_dir, out_file_prefix)
    os.makedirs(os.path.join(res_dir, 'mapmri', subject_dir), exist_ok=True)
    for metric, name in zip(metrics, metric_names):
        out_file = os.path.join(subject_dir, f"{out_file_prefix}_{name}.nii.gz")
        save_nifti(out_file, metric, affine)
        out_files.append(out_file)
    print(f"{colours.CBLUE}All metrics saved to {subject_dir}.{colours.CEND}")

    return tuple(out_files)

def correct_metric_nifti(metric_path, threshold, correct_neg=True, replace_with=0):
    import numpy as np
    from dipy.io.image import load_nifti, save_nifti

    # Load the metric which needs correcting
    metric, affine = load_nifti(metric_path)

    # Replace NaN values
    metric[np.isnan(metric)] = replace_with

    # If correct_neg is True, replace negative values
    if correct_neg:
        metric[metric < 0] = replace_with

    # Replace values above the threshold
    metric[metric > threshold] = replace_with
    
    # Save the corrected metric to the same folder, but with modified name
    out_file = metric_path.replace(".nii.gz", "_corrected.nii.gz")
    save_nifti(out_file, metric, affine)

    return out_file