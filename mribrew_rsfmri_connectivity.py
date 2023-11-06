import os
import pandas as pd
from nipype import Function, config, logging
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces.io import DataSink, DataGrabber

from mribrew.utils import colours
from mribrew.rsfmri_interface import (setup_atlases, extract_timeseries, 
                                      compute_connectivity)

### ------ FILE DIRS & COMPUTATION VARIABLES

# Define constants and paths
cwd = os.getcwd()
data_dir = os.path.join(cwd, 'data')
proc_dir = '/mnt/raid1/RSfMRI' # os.path.join(data_dir, 'proc')
wf_dir = os.path.join(cwd, 'wf')
res_dir = os.path.join(data_dir, 'res')
log_dir = os.path.join(wf_dir, 'log')

# subject_list = next(os.walk(proc_dir))[1]
all_subjects = next(os.walk(proc_dir))[1]
subjects_of_interest = pd.read_csv(os.path.join(data_dir, 'temp_csv/subjects_for_fmri_analysis.csv'), header=None)[0].to_list()

subject_list = set(
    subject for subject in all_subjects
    for sub in subjects_of_interest if sub in subject
)

# Computational variables
processing_type = 'MultiProc' # or 'Linear'
total_memory = 32 # in GB
n_cpus = 32 # number of nipype processes to run at the same time
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
# datasource.inputs.template = '%s/RSfMRI/processed_and_censored_32bit.nii.gz'
datasource.inputs.template = '%s/processed_and_censored_32bit.nii.gz'
datasource.inputs.sort_filelist = True

# Set up a EBM ROIs setup node
setup_atlases_node = pe.Node(Function(input_names=[],
                                       output_names=['desikan_img', 
                                                     'schaefer_img'],
                                       function=setup_atlases),
                              name="setup_atlases")

# ---------------------- PROCESSING NODES ----------------------
print(colours.CGREEN + "Creating Processing Nodes." + colours.CEND)

# Set up a node for extracting time series data for each subject
extract_timeseries_node = pe.Node(Function(input_names=['rsfmri_file', 
                                                        'desikan_img', 
                                                        'schaefer_img'],
                                               output_names=['desikan_time_series', 
                                                             'schaefer_time_series'],
                                               function=extract_timeseries),
                                      name="extract_timeseries")

# Set up a node for computing correlation matrices for each subject
compute_connectivity_node = pe.Node(Function(input_names=['desikan_time_series',
                                                     'schaefer_time_series'],
                                            output_names=['sub_corrmat_desikan_file', 
                                                        #   'sub_partial_corrmat_desikan_file', 
                                                          'sub_corrmat_schaefer_file'],
                                                        #   'sub_partial_corrmat_schaefer_file'],
                                            function=compute_connectivity),
                                    name="compute_connectivity")

# ---------------------- OUTPUT NODES ----------------------
print(colours.CGREEN + "Creating Output Nodes." + colours.CEND)

# DataSink
datasink = pe.Node(DataSink(base_directory=res_dir, container='connectivity'), name='datasink')

# ---------------------- CREATE WORKFLOW AND CONNECT NODES ----------------------
print(colours.CGREEN + 'Connecting Nodes.\n' + colours.CEND)

# Workflow setup
workflow = pe.Workflow(name='connectivity_wf', base_dir=wf_dir)
workflow.connect([
    (infosource, datasource, [('subject_id', 'subject_id')]),

    # Extract time series for all EBM ROIs
    (setup_atlases_node, extract_timeseries_node, [('desikan_img', 'desikan_img')]),
    (setup_atlases_node, extract_timeseries_node, [('schaefer_img', 'schaefer_img')]),
    (datasource, extract_timeseries_node, [('rsfmri_file', 'rsfmri_file')]),

    # Compute correlation matrices
    (extract_timeseries_node, compute_connectivity_node, [('desikan_time_series', 'desikan_time_series')]),
    (extract_timeseries_node, compute_connectivity_node, [('schaefer_time_series', 'schaefer_time_series')]),
    
    # Export all correlation matrices
    (compute_connectivity_node, datasink, [('sub_corrmat_desikan_file', 
                                            '@sub_corrmat_desikan_file')]),
    # (compute_connectivity_node, datasink, [('sub_partial_corrmat_desikan_file', 
    #                                         '@sub_partial_corrmat_desikan_file')]),
    (compute_connectivity_node, datasink, [('sub_corrmat_schaefer_file', 
                                            '@sub_corrmat_schaefer_file')]),
    # (compute_connectivity_node, datasink, [('sub_partial_corrmat_schaefer_file', 
    #                                         '@sub_partial_corrmat_schaefer_file')]),
])

if __name__ == '__main__':
    workflow.write_graph(graph2use='orig')
    workflow.run(plugin=processing_type, plugin_args=plugin_args)