def read_dwi_data(data_file, mask_file, bvec_file, bval_file,
                  big_delta, small_delta, use_subset_data):
    import numpy as np
    from dipy.io.image import load_nifti
    from dipy.core.gradients import gradient_table

    data, affine = load_nifti(data_file)
    mask = load_nifti(mask_file)[0]
    data = data * mask[..., np.newaxis]

    if use_subset_data==True:
        data = data[:, 40:50, 36:40]

    bvals, bvecs = np.loadtxt(bval_file), np.loadtxt(bvec_file).T
    gtab = gradient_table(bvals, bvecs, big_delta, small_delta)

    return data, affine, gtab

# Save matrices as pickle files
def save_to_pickle(data, fname):
    import os
    import pickle
    filepath = os.path.abspath(fname)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    return filepath

# Read in the corr matrices for this subject
def read_pickle(filepath):
    import os
    import pickle
    with open(filepath, 'rb') as f:
        var = pickle.load(f)
    return var

def save_csv(data, fname):
    import numpy as np
    return np.savetxt(fname, data, delimiter=",")

def read_csv(filepath):
    import pandas as pd
    return pd.read_csv(filepath, header=None).to_numpy()

def read_mat(filepath):
    """Import data from a Matlab file."""
    from scipy.io import loadmat
    var = loadmat(filepath)
    return list(var.values())[3]