import os
import numpy as np
from nipype import Function, Workflow, config, logging
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces import io

from mribrew.utils import colours, Tee, should_use_subset_data
from mribrew.data_io import read_dwi_data
from mribrew.mapmri_funcs import correct_neg_data

# ---------------------- Set up directory structures and constant variables ----------------------
cwd = os.getcwd()
data_dir = os.path.join(cwd, 'data')
raw_dir = os.path.join(data_dir, 'raw')
proc_dir = os.path.join(data_dir, 'proc')
wf_dir = os.path.join(cwd, 'wf')
res_dir = os.path.join(data_dir, 'res')
log_dir = os.path.join(wf_dir, 'log')

subject_list = next(os.walk(proc_dir))[1]  # processed subjects

# Computational variables
use_subset_data = False
processing_type = 'MultiProc' # or 'Linear'
n_cpus = 12

# MAPMRI variables
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
                     bval_threshold=np.inf,
                     dti_scale_estimation=True,
                     static_diffusivity=0.7e-3,
                     cvxpy_solver=None)

# Set up logging
os.makedirs(log_dir, exist_ok=True)
config.update_config({'logging': {'log_directory': log_dir,'log_to_file': True}})
logging.update_logging(config)

print(f"\n{colours.UBOLD}{colours.CYELLOW}Starting the MAPMRI pipeline...{colours.CEND}")
print(f"Using the following constants:\n"
      f"MAPMRI Parameters: {mapmri_params}\n"
      f"Small delta: {small_delta}\n"
      f"Big delta: {big_delta}\n"
      f"Using subset data: {use_subset_data}\n"
      f"Number of CPUs: {n_cpus}\n"
      f"Processing type: {processing_type}\n")

# ---------------------- INPUT SOURCE NODES ----------------------
print(colours.CGREEN + "Creating Source Nodes." + colours.CEND)

# Set up input files
info = dict(dwi_eddy_file=[['subject_id', 'dwi', 'eddy_corrected.nii.gz']],
            bvec_file=[['subject_id', 'dwi', 'gradChecked.bvec']],
            bval_file=[['subject_id', 'dwi', 'gradChecked.bval']],
            dwi_mask_file=[['subject_id', 'dwi', 'brain_dwi_mask.nii.gz']])

# Set up infosource node
infosource = pe.Node(niu.IdentityInterface(fields=['subject_id']), name='infosource')
infosource.iterables = [('subject_id', subject_list)]
infosource.inputs.big_delta = big_delta
infosource.inputs.small_delta = small_delta
infosource.inputs.use_subset_data = use_subset_data

# Set up datasource node
datasource = pe.Node(io.DataGrabber(infields=['subject_id'], outfields=list(info.keys())),
                                    name='datasource')
datasource.inputs.base_directory = proc_dir
datasource.inputs.template = "%s/%s/%s"
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True

# ---------------------- OUTPUT SOURCE NODE ----------------------
print(colours.CGREEN + "Creating Sink Node." + colours.CEND)

#### MAYBE CAN DELETE THIS IF NOT USING FOR EXPORTING
# Set up a sink node where all output is stored in subject folder
datasink = pe.Node(io.DataSink(parameterization=True), name='datasink')
datasink.inputs.base_directory = os.path.join(res_dir, 'mapmri')

# ---------------------- PROCESSING NODES ----------------------
print(colours.CGREEN + "Creating Processing Nodes." + colours.CEND)

# Set up a node for loading in DWI data (and apply mask) and gradient table
read_data = pe.Node(Function(input_names=['data_file', 'mask_file',
                                          'bvec_file', 'bval_file',
                                          'big_delta', 'small_delta',
                                          'use_subset_data'],
                             output_names=['data', 'affine', 'gtab'],
                             function=read_dwi_data),
                    name='read_data')
read_data.inputs.big_delta = big_delta
read_data.inputs.small_delta = small_delta
read_data.inputs.use_subset_data = use_subset_data

# Set up a node for correcting negative values to average volume value per timepoint
data_correction = pe.Node(Function(input_names=['data'],
                                   output_names=['data_corrected'],
                                   function=correct_neg_data),
                          name='data_correction')

# Set up a node for fitting data to MAPMRI model and saving the metrics
def fit_mapmri_model(data, gtab, mapmri_params=None):
    import numpy as np
    from dipy.reconst.mapmri import MapmriModel
    from dipy.data import get_sphere

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

# Set up a node for fitting data to the MAPMRI model
fit_mapmri = pe.Node(Function(input_names=['data', 'gtab', 'mapmri_params'],
                              output_names=['MSD', 'QIV', 'RTOP', 'RTAP', 'RTPP'],
                              function=fit_mapmri_model),
                     name='fit_mapmri')
fit_mapmri.inputs.mapmri_params = mapmri_params

def metrics_to_nifti(affine, MSD, QIV, RTOP, RTAP, RTPP, out_file_prefix, res_dir):
    """
    Convert the metrics to Nifti files and save them using Dipy's save_nifti.

    Parameters:
    - MSD, QIV, RTOP, RTAP, RTPP: The metrics values.
    - out_file_prefix: Prefix for the output files.
    - res_dir: Directory to save the metrics

    Returns:
    - File paths to the saved metrics.
    """
    from dipy.io.image import save_nifti
    import os
    import numpy as np

    metrics = [MSD, QIV, RTOP, RTAP, RTPP]
    metric_names = ["MSD", "QIV", "RTOP", "RTAP", "RTPP"]
    out_files = []

    for metric, name in zip(metrics, metric_names):
        subject_dir = os.path.join(res_dir, out_file_prefix)
        out_file = os.path.join(subject_dir, f"{out_file_prefix}_{name}.nii.gz")
        save_nifti(out_file, metric, affine)
        out_files.append(out_file)

    return tuple(out_files)


# Set up a node for converting metrics to NIfTI format
metrics_nii = pe.Node(Function(input_names=['affine', 'MSD', 'QIV', 'RTOP', 'RTAP', 'RTPP', 'out_file_prefix', 'res_dir'],
                              output_names=['MSD_file', 'QIV_file', 'RTOP_file', 'RTAP_file', 'RTPP_file'],
                              function=metrics_to_nifti),
                     name='metrics_nii')
metrics_nii.inputs.res_dir = os.path.join(res_dir, 'mapmri')

# ---------------------- CREATE WORKFLOW AND CONNECT NODES ----------------------
print(colours.CGREEN + 'Connect Nodes.' + colours.CEND)

workflow = Workflow(name='mapmri_wf', base_dir=f"{wf_dir}")
workflow.connect([
    # INPUT/OUTPUT STRUCTURE
    (infosource, datasource, [('subject_id', 'subject_id')]),
    (infosource, datasink, [('subject_id',  'container')]),
    (datasource, read_data, [('dwi_eddy_file', 'data_file'),
                             ('dwi_mask_file', 'mask_file'),
                             ('bvec_file', 'bvec_file'),
                             ('bval_file', 'bval_file')]),
    (read_data, data_correction, [('data', 'data')]),
    (data_correction, fit_mapmri, [('data_corrected', 'data')]),
    (read_data, fit_mapmri, [('gtab', 'gtab')]),
    (fit_mapmri, metrics_nii, [('MSD', 'MSD'),
                               ('QIV', 'QIV'),
                               ('RTOP', 'RTOP'),
                               ('RTAP', 'RTAP'),
                               ('RTPP', 'RTPP')]),
    (read_data, metrics_nii, [('affine', 'affine')]),
    (infosource, metrics_nii, [('subject_id', 'out_file_prefix')])
])

workflow.write_graph(graph2use='orig')
workflow.run(plugin=processing_type, plugin_args={'n_procs' : n_cpus})
