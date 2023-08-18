import os
import numpy as np
from nipype import Function, Workflow, config, logging
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces import io

from mribrew.utils import colours
from mribrew.data_io import read_dwi_data
from mribrew.mapmri_funcs import correct_neg_data, fit_mapmri_model, metrics_to_nifti, correct_metric_nifti

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
n_cpus = 2
os.environ['OMP_NUM_THREADS'] = '2'  # or whatever number of threads you desire

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

# Set up a node for fitting data to the MAPMRI model
fit_mapmri = pe.Node(Function(input_names=['data', 'gtab', 'mapmri_params'],
                              output_names=['MSD', 'QIV', 'RTOP', 'RTAP', 'RTPP'],
                              function=fit_mapmri_model),
                     name='fit_mapmri')
fit_mapmri.inputs.mapmri_params = mapmri_params

# ---------------------- OUTPUT NODES ----------------------
print(colours.CGREEN + "Creating Output Nodes." + colours.CEND)

# Set up a node for saving metrics as NIfTI
metrics_to_nii = pe.Node(Function(input_names=['affine', 'MSD', 'QIV', 'RTOP', 'RTAP', 
                                               'RTPP', 'out_file_prefix', 'res_dir'],
                              output_names=['MSD_file', 'QIV_file', 'RTOP_file', 
                                            'RTAP_file', 'RTPP_file'],
                              function=metrics_to_nifti),
                     name='metrics_to_nii')
metrics_to_nii.inputs.res_dir = os.path.join(res_dir, 'mapmri')

# Set up a node for correcting RTOP metric and saving as NIfTI 
rtop_corrected_to_nii = pe.Node(Function(input_names=['metric_path', 'threshold', 
                                                      'correct_neg', 'replace_with'],
                              output_names=['out_file'],
                              function=correct_metric_nifti),
                     name='rtop_corrected_to_nii')
rtop_corrected_to_nii.inputs.threshold = 2000000
rtop_corrected_to_nii.inputs.correct_neg = True
rtop_corrected_to_nii.inputs.replace_with = 0

# ---------------------- CREATE WORKFLOW AND CONNECT NODES ----------------------
print(colours.CGREEN + 'Connecting Nodes.\n' + colours.CEND)

workflow = Workflow(name='mapmri_wf', base_dir=f"{wf_dir}")
workflow.connect([
    (infosource, datasource, [('subject_id', 'subject_id')]),
    (datasource, read_data, [('dwi_eddy_file', 'data_file'),
                             ('dwi_mask_file', 'mask_file'),
                             ('bvec_file', 'bvec_file'),
                             ('bval_file', 'bval_file')]),
    (read_data, data_correction, [('data', 'data')]),
    (data_correction, fit_mapmri, [('data_corrected', 'data')]),
    (read_data, fit_mapmri, [('gtab', 'gtab')]),
    (fit_mapmri, metrics_to_nii, [('MSD', 'MSD'),
                               ('QIV', 'QIV'),
                               ('RTOP', 'RTOP'),
                               ('RTAP', 'RTAP'),
                               ('RTPP', 'RTPP')]),
    (read_data, metrics_to_nii, [('affine', 'affine')]),
    (infosource, metrics_to_nii, [('subject_id', 'out_file_prefix')]),
    (metrics_to_nii, rtop_corrected_to_nii, [('RTOP_file', 'metric_path')])
])

if __name__ == '__main__':
    workflow.write_graph(graph2use='orig')
    workflow.run(plugin=processing_type, plugin_args={'n_procs' : n_cpus})
