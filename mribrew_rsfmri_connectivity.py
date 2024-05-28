import os
import pandas as pd
from nipype import Function, config, logging
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces import io

from mribrew.utils import colours
from mribrew.rsfmri_interface import (extract_dk_sch_bold_timeseries, 
                                      compute_fc)

### ------ FILE DIRS & COMPUTATION VARIABLES

# Define constants and paths
cwd = os.getcwd()
data_dir = os.path.join(cwd, 'data')
proc_dir = os.path.join(data_dir, 'proc') # '/mnt/raid1/RSfMRI'
wf_dir = os.path.join(cwd, 'wf')
res_dir = os.path.join(data_dir, 'res', 'fc')
log_dir = os.path.join(wf_dir, 'log')

# subject_list = next(os.walk(proc_dir))[1]
all_subjects = next(os.walk(proc_dir))[1]
subjects_of_interest = all_subjects #pd.read_csv(os.path.join(data_dir, 'temp_csv/subjects_for_fmri_analysis.csv'), header=None)[0].to_list()

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
datasource = pe.Node(interface=io.DataGrabber(infields=['subject_id'],
                                           outfields=['rsfmri_file']),
                     name='datasource')
datasource.inputs.base_directory = proc_dir
datasource.inputs.template = '%s/processed_and_censored_32bit.nii.gz'
datasource.inputs.sort_filelist = True

# ---------------------- PROCESSING NODES ----------------------
print(colours.CGREEN + "Creating Processing Nodes." + colours.CEND)

# Set up a node for extracting time series data for each subject
extract_timeseries_node = pe.Node(Function(input_names=['rsfmri_file'],
                                            output_names=['dk_timeseries', 
                                                            'sch_timeseries'],
                                            function=extract_dk_sch_bold_timeseries),
                                      name="extract_timeseries")

# Set up a node for computing correlation matrices for each subject
compute_fc_node = pe.Node(Function(input_names=['dk_timeseries',
                                                'sch_timeseries'],
                                            output_names=['dk_fc', 
                                                          'sch_fc'],
                                            function=compute_fc),
                                    name="compute_fc")

# ---------------------- OUTPUT NODES ----------------------
print(colours.CGREEN + "Creating Output Nodes." + colours.CEND)

# DataSink
# datasink = pe.Node(DataSink(base_directory=res_dir, container='connectivity'), name='datasink')
datasink = pe.Node(io.DataSink(parameterization=False), name='datasink')
datasink.inputs.base_directory = res_dir

# ---------------------- CREATE WORKFLOW AND CONNECT NODES ----------------------
print(colours.CGREEN + 'Connecting Nodes.\n' + colours.CEND)

# Workflow setup
workflow = pe.Workflow(name='connectivity_wf', base_dir=wf_dir)
workflow.connect([
    (infosource, datasource, [('subject_id', 'subject_id')]),
    (infosource, datasink, [('subject_id',  'container')]),

    # Extract time series for all EBM ROIs
    (datasource, extract_timeseries_node, [('rsfmri_file', 'rsfmri_file')]),

    # Compute correlation matrices
    (extract_timeseries_node, compute_fc_node, [('dk_timeseries', 'dk_timeseries')]),
    (extract_timeseries_node, compute_fc_node, [('sch_timeseries', 'sch_timeseries')]),
    
    # Export all correlation matrices
    (compute_fc_node, datasink, [('dk_fc', 
                                '@dk_fc')]),
    (compute_fc_node, datasink, [('sch_fc', 
                                '@sch_fc')]),
])

if __name__ == '__main__':
    workflow.write_graph(graph2use='orig')
    workflow.run(plugin=processing_type, plugin_args=plugin_args)