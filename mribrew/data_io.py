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