import os
from nipype import Function, JoinNode, config, logging
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces.io import DataSink, DataGrabber

from mribrew.utils import colours
from mribrew.rsfmri_ebm_interface import (setup_ebm_rois, extract_ebm_timeseries, 
                                          compute_corrmatrices, aggregate_matrices)

### ------ FILE DIRS & COMPUTATION VARIABLES

# Define constants and paths
cwd = os.getcwd()
data_dir = os.path.join(cwd, 'data')
proc_dir = os.path.join(data_dir, 'proc_test')
wf_dir = os.path.join(cwd, 'wf')
res_dir = os.path.join(data_dir, 'res')
log_dir = os.path.join(wf_dir, 'log')

subject_list = next(os.walk(proc_dir))[1]  # processed subjects

# Computational variables
processing_type = 'MultiProc' # or 'Linear'
total_memory = 6 # in GB
n_cpus = 6 # number of nipype processes to run at the same time
os.environ['OMP_NUM_THREADS'] = str(n_cpus)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cpus)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

plugin_args = {
    'n_procs': n_cpus,
    'memory_gb': total_memory,
    'raise_insufficient': True,
    'scheduler': 'mem_thread',
}

# Set up logging
os.makedirs(log_dir, exist_ok=True)
config.update_config({'logging': {'log_directory': log_dir,'log_to_file': True}})
logging.update_logging(config)

# ---------------------- INPUT SOURCE NODES ----------------------
print(colours.CGREEN + "Creating Source Nodes." + colours.CEND)

# Set up infosource node
infosource = pe.Node(niu.IdentityInterface(fields=['subject_id']),
                     name="infosource")
infosource.iterables = [('subject_id', subject_list)]

# Set up datasource node
datasource = pe.Node(interface=DataGrabber(infields=['subject_id'],
                                           outfields=['rsfmri_file']),
                     name='datasource')
datasource.inputs.base_directory = proc_dir
datasource.inputs.template = '%s/RSfMRI/processed_and_censored_32bit.nii.gz'
datasource.inputs.sort_filelist = True

# Set up a EBM ROIs setup node
setup_ebm_rois_node = pe.Node(Function(input_names=[],
                                       output_names=['ebm_rois_desikan_schaefer_dict', 
                                                     'desikan_img', 'schaefer_img',
                                                     'desikan_labels', 'schaefer_labels'],
                                       function=setup_ebm_rois),
                              name="setup_ebm_rois")

# ---------------------- PROCESSING NODES ----------------------
print(colours.CGREEN + "Creating Processing Nodes." + colours.CEND)

# Set up a node for extracting time series data for each subject
extract_ebm_timeseries_node = pe.Node(Function(input_names=['rsfmri_file', 
                                                            'desikan_img', 'schaefer_img',
                                                            'desikan_labels', 'schaefer_labels', 
                                                            'ebm_rois_desikan_schaefer_dict'],
                                               output_names=['ebm_time_series', 
                                                             'ebm_time_series_labelled'],
                                               function=extract_ebm_timeseries),
                                      name="extract_ebm_timeseries")

# Set up a node for computing correlation matrices for each subject
compute_corrmatrices_node = pe.Node(Function(input_names=['ebm_time_series_labelled'],
                                            output_names=['sub_ebm_time_series_labelled_file', 
                                                          'sub_corrmatrices_file', 
                                                          'sub_partial_corrmatrices_file',
                                                          'sub_hemi_corrmatrices_file', 
                                                          'sub_hemi_partial_corrmatrices_file'],
                                            function=compute_corrmatrices),
                                    name="compute_corrmatrices")

# Set up a node for merging all subjects' correlation matrices together
aggregate_matrices_node = JoinNode(Function(input_names=['subject_list', 
                                                         'ebm_rois_desikan_schaefer_dict', 
                                                         'sub_corrmatrices_file', 
                                                         'sub_partial_corrmatrices_file',
                                                         'sub_hemi_corrmatrices_file', 
                                                         'sub_hemi_partial_corrmatrices_file'],
                                            output_names=['all_corrmatrices_file', 
                                                          'all_partial_corrmatrices_file', 
                                                          'all_hemi_corrmatrices_file', 
                                                          'all_hemi_partial_corrmatrices_file'],
                                            function=aggregate_matrices),
                                   joinsource='infosource',
                                   joinfield=['subject_list',
                                              'sub_corrmatrices_file', 
                                              'sub_partial_corrmatrices_file',
                                              'sub_hemi_corrmatrices_file', 
                                              'sub_hemi_partial_corrmatrices_file'],
                                   name="aggregate_matrices")
aggregate_matrices_node.inputs.subject_list = subject_list

# ---------------------- OUTPUT NODES ----------------------
print(colours.CGREEN + "Creating Output Nodes." + colours.CEND)

# DataSink
datasink = pe.Node(DataSink(base_directory=res_dir, container='connectivity'), name='datasink')

# ---------------------- CREATE WORKFLOW AND CONNECT NODES ----------------------
print(colours.CGREEN + 'Connecting Nodes.\n' + colours.CEND)

# Workflow setup
workflow = pe.Workflow(name='ebmconnectivity_wf', base_dir=wf_dir)
workflow.connect([
    (infosource, datasource, [('subject_id', 'subject_id')]),

    # Extract time series for all EBM ROIs
    (setup_ebm_rois_node, extract_ebm_timeseries_node, [('desikan_img', 'desikan_img')]),
    (setup_ebm_rois_node, extract_ebm_timeseries_node, [('schaefer_img', 'schaefer_img')]),
    (setup_ebm_rois_node, extract_ebm_timeseries_node, [('desikan_labels', 'desikan_labels')]),
    (setup_ebm_rois_node, extract_ebm_timeseries_node, [('schaefer_labels', 'schaefer_labels')]),
    (setup_ebm_rois_node, extract_ebm_timeseries_node, [('ebm_rois_desikan_schaefer_dict', 
                                                         'ebm_rois_desikan_schaefer_dict')]),
    (datasource, extract_ebm_timeseries_node, [('rsfmri_file', 'rsfmri_file')]),

    # Compute correlation matrices
    (extract_ebm_timeseries_node, compute_corrmatrices_node, [('ebm_time_series_labelled', 
                                                               'ebm_time_series_labelled')]),
    
    # Export all correlation matrices
    (compute_corrmatrices_node, datasink, [('sub_ebm_time_series_labelled_file', 
                                            '@sub_ebm_time_series_labelled_file')]),
    (compute_corrmatrices_node, datasink, [('sub_corrmatrices_file', 
                                            '@sub_corrmatrices_file')]),
    (compute_corrmatrices_node, datasink, [('sub_partial_corrmatrices_file', 
                                            '@sub_partial_corrmatrices_file')]),
    (compute_corrmatrices_node, datasink, [('sub_hemi_corrmatrices_file', 
                                            '@sub_hemi_corrmatrices_file')]),
    (compute_corrmatrices_node, datasink, [('sub_hemi_partial_corrmatrices_file', 
                                            '@sub_hemi_partial_corrmatrices_file')]),

    # Merge all subjects' correlation matrices
    (infosource, aggregate_matrices_node, [('subject_id', 'subject_list')]),
    (setup_ebm_rois_node, aggregate_matrices_node, [('ebm_rois_desikan_schaefer_dict', 
                                                     'ebm_rois_desikan_schaefer_dict')]),
    (compute_corrmatrices_node, aggregate_matrices_node, [('sub_corrmatrices_file', 
                                                           'sub_corrmatrices_file')]),
    (compute_corrmatrices_node, aggregate_matrices_node, [('sub_partial_corrmatrices_file', 
                                                           'sub_partial_corrmatrices_file')]),
    (compute_corrmatrices_node, aggregate_matrices_node, [('sub_hemi_corrmatrices_file', 
                                                           'sub_hemi_corrmatrices_file')]),
    (compute_corrmatrices_node, aggregate_matrices_node, [('sub_hemi_partial_corrmatrices_file', 
                                                           'sub_hemi_partial_corrmatrices_file')]),

    # Export the merged correlation matrices
    (aggregate_matrices_node, datasink, [('all_corrmatrices_file', 
                                          '@all_corrmatrices_file')]),
    (aggregate_matrices_node, datasink, [('all_partial_corrmatrices_file', 
                                          '@all_partial_corrmatrices_file')]),
    (aggregate_matrices_node, datasink, [('all_hemi_corrmatrices_file', 
                                          '@all_hemi_corrmatrices_file')]),
    (aggregate_matrices_node, datasink, [('all_hemi_partial_corrmatrices_file', 
                                          '@all_hemi_partial_corrmatrices_file')]),
])

if __name__ == '__main__':
    workflow.write_graph(graph2use='orig')
    workflow.run(plugin=processing_type, plugin_args=plugin_args)