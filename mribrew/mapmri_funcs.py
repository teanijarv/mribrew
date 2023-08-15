import os
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.reconst.mapmri import MapmriModel

from mribrew.utils import colours

def correct_neg_data(data):
    """
    Correct the dMRI data by replacing negative voxel values with the total 
    volume average per timepoint.

    Parameters:
    -----------
    data : ndarray (4D)
        dMRI dataset containing volumes (first three dimensions) over time 
        (fourth dimension). Expected shape is (X, Y, Z, t).

    Returns:
    --------
    ndarray (4D)
        Corrected dMRI data with negative values replaced by the average 
        volume value for the respective timepoint.

    Notes:
    ------
    This function modifies the input data in-place if the data has any 
    negative values.
    """

    # Mask of voxels with negative for each time point
    neg_mask = data<0

    # Count and print how many negative voxels there are
    num_neg_voxels = np.sum(neg_mask)
    if num_neg_voxels > 0:
        print(f"Found {num_neg_voxels} negative voxels in the data - replacing with "
               "average volume value for that timepoint.")
    else:
        print("No negative voxels found in the data - all good!")

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

def mapmri_fit_and_save(data, affine, gtab, mapmri_params=None, bval_threshold=None,
                        metrics_to_save=None, cur_subj='unnamed', res_mapmri_subj_dir=None):
    """
    Fit the MAPMRI model to diffusion MRI data and save the desired metrics.

    Parameters:
    -----------
    data : ndarray
        dMRI dataset containing volumes (first three dimensions) over time 
        (fourth dimension). Expected shape is (X, Y, Z, t).
    
    affine : ndarray
        The affine matrix for the data. Used when saving the results.
    
    gtab : GradientTable object
        Object that contains information about the b-values and b-vectors.
    
    mapmri_params : dict, optional
        Parameters for the MAPMRI fitting. Defaults include various model-related parameters. 
        If not provided, default parameters are used.
    
    bval_threshold : float, optional
        Threshold for the b-values to be used in the model. If not provided, 
        set to infinity (use all b-values).
    
    metrics_to_save : list of str, optional
        List of metric names that you wish to compute and save. If not provided, 
        all available metrics are computed and saved. Available metrics are:
        ['MSD', 'QIV', 'RTOP', 'RTAP', 'RTPP', 'NG'].

    cur_subj : str, optional
        Name of the current subject. Used when naming the output files. If not provided,
        'unnamed' will be used.
    
    res_mapmri_subj_dir : str, optional
        Directory path where the metrics should be saved. If not provided, metrics 
        are saved in the current working directory.

    Returns:
    --------
    None
        The function saves the computed metrics to the specified directory and doesn't return any value.

    """

    # If results folder is not defined, set the current working directory for exporting
    if res_mapmri_subj_dir==None:
        res_mapmri_subj_dir = os.getcwd()
    
    # If bval_threshold is not defined, set it as infinite
    if bval_threshold==None:
        bval_threshold = np.inf

    # If not defined, use the following default parameters for MAPMRI
    if mapmri_params==None:
        mapmri_params = dict(radial_order=6,
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

    # Fit MAPMRI model to the data
    print(f"Fitting the MAPMRI model to the data (bval_threshold={bval_threshold}; {mapmri_params})")
    map_model = MapmriModel(gtab, bval_threshold=bval_threshold, **mapmri_params)
    mapfit = map_model.fit(data)

    # Definition of metric calculations as lambda functions to delay execution
    metrics = {
        'MSD': lambda: mapfit.msd(),
        'QIV': lambda: mapfit.qiv(),
        'RTOP': lambda: mapfit.rtop(),
        'RTAP': lambda: mapfit.rtap(),
        'RTPP': lambda: mapfit.rtpp(),
        'NG': lambda: mapfit.ng()
    }

    # If not defined which metrics to save, save all metrics
    if metrics_to_save is None:
        metrics_to_save = list(metrics.keys())

    # Compute and save all the metrics of interest
    for metric_name in metrics_to_save:
        # Ensure the metric name is valid
        if metric_name in metrics:
            # Compute the metric
            metric_value = metrics[metric_name]()
            # Save the metric
            metric_dir = os.path.join(res_mapmri_subj_dir, f'{cur_subj}_{metric_name}.nii.gz')
            save_nifti(metric_dir, metric_value, affine)
            print(f"{colours.CBLUE}Metric {metric_name} computed and saved to {metric_dir}{colours.CEND}")
        else:
            print(f"{colours.CRED}Warning: Metric {metric_name} not recognized and will not be computed.{colours.CEND}")

def metric_correction_and_save(metric_name, threshold, cur_subj, res_mapmri_subj_dir, correct_neg=True, replace_with=0):
    """
    Load a metric NIfTI file, correct its values based on specified criteria, and save the corrected NIfTI file.

    Parameters:
    -----------
    metric_name : str
        Name of the metric to be loaded and corrected (e.g., 'RTOP').
        
    threshold : float
        Threshold value above which the metric values are set to the replace_with value.

    cur_subj : str
        Identifier for the current subject. Used for loading and saving the NIfTI files.

    res_mapmri_subj_dir : str
        Directory path where the metric NIfTI files are stored.

    correct_neg : bool, optional (default=True)
        If True, negative values in the metric will be set to the replace_with value.

    replace_with : float, optional (default=0)
        Value with which to replace the metric values that are either NaN, negative (if correct_neg is True), or above the threshold.

    Returns:
    --------
    None
        The function saves the corrected metric to the specified directory and doesn't return any value.

    """

    # Print the correction parameters and initialize log list
    neg_msg = "and negative values" if correct_neg else ""
    print(f"Correcting the {metric_name} metric by replacing NaNs, values over {threshold} {neg_msg} with {replace_with}...")

    # Load the metric NIfTI file
    metric, affine = load_nifti(os.path.join(res_mapmri_subj_dir, f'{cur_subj}_{metric_name}.nii.gz'))

    # Count and replace NaN values
    num_nan_values = np.sum(np.isnan(metric))
    if num_nan_values > 0:
        metric[np.isnan(metric)] = replace_with
        print(f"{num_nan_values} NaN values found and replaced with {replace_with}")

    # Count and replace negative values if correct_neg is True
    num_neg_values = np.sum(metric < 0)
    if correct_neg and num_neg_values > 0:
        metric[metric < 0] = replace_with
        print(f"{num_neg_values} negative values found and replaced with {replace_with}")

    # Count and replace values above the threshold
    num_above_threshold = np.sum(metric > threshold)
    if num_above_threshold > 0:
        metric[metric > threshold] = replace_with
        print(f"{num_above_threshold} values above {threshold} found and replaced with {replace_with}")


    # Save the corrected metric NIfTI file
    save_nifti(os.path.join(res_mapmri_subj_dir, f'{cur_subj}_{metric_name}_corrected.nii.gz'), metric, affine)
    print(colours.CBLUE + f"Corrected metric saved to {cur_subj}_{metric_name}_corrected.nii.gz" + colours.CEND)
