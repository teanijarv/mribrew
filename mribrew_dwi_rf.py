# %%
import os
import pandas as pd
from nipype import config, logging
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces import io, mrtrix3

from mribrew.utils import (colours, split_subject_scan_list)
from mribrew.act_interface import ResponseMean

# ---------------------- Set up directory structures and constant variables ----------------------
cwd = os.getcwd()
misc_dir = os.path.join(cwd, 'misc')
data_dir = os.path.join(cwd, 'data')
proc_dir = os.path.join(data_dir, 'proc', 'dwi_proc')
wf_dir = os.path.join(cwd, 'wf')
res_dir = os.path.join(data_dir, 'res', 'act')
log_dir = os.path.join(wf_dir, 'log_rf')

# load csv with all the subjects of interest
df_rf_subs = pd.read_csv(os.path.join(misc_dir, 'rf_subscans.csv'), index_col=0)

# generate a list of all subject-scans and only the ones used for response function based on dataframe
subscans_indir = []
subscans_indir_rf = []
for sub in next(os.walk(proc_dir))[1]:
    for scan in next(os.walk(os.path.join(proc_dir, sub)))[1]:
        subscans_indir.append([sub, scan])
        if f'{sub}__{scan[:-2]}' in df_rf_subs['mri_date__index'].to_list():
            subscans_indir_rf.append([sub, scan])

print(f'all sub-scans with files = {len(subscans_indir)}')
print(f'all rf sub-scans = {len(df_rf_subs)}')
print(f'rf sub-scans with files = {len(subscans_indir_rf)}')

# Computational variables
processing_type = 'MultiProc' # or 'Linear'
plugin_args = {
    'n_procs': 4,
    'memory_gb': 96,
    'raise_insufficient': True,
}

# Set up logging
os.makedirs(log_dir, exist_ok=True)
config.update_config({'logging': {'log_directory': log_dir, 'log_to_file': True}})
logging.update_logging(config)

# ---------------------- INPUT SOURCE NODES ----------------------
print(colours.CGREEN + "Creating Source Nodes." + colours.CEND)

# Set up input files
info = dict(
    dwi_eddy_file=[['data', 'proc', 'dwi_proc', 'subject_id', 'scan_id', 'dwi', 'eddy_corrected.nii.gz']],
    bvec_file=[['data', 'proc', 'dwi_proc', 'subject_id', 'scan_id','dwi', 'gradChecked.bvecs']],
    bval_file=[['data', 'proc', 'dwi_proc', 'subject_id', 'scan_id','dwi', 'gradChecked.bvals']],
    dwi_mask_file=[['data', 'proc', 'dwi_proc', 'subject_id', 'scan_id', 'dwi', 'dwi_mask.nii.gz']]
)

## RESPONSE FUNCTION WORKFLOW

# Set up infosource node for response function wf
infosource_rf = pe.Node(niu.IdentityInterface(fields=['subject_scan']), name='infosource_rf')
infosource_rf.iterables = [('subject_scan', subscans_indir_rf)]

splitSubjectScanList_rf = pe.Node(niu.Function(input_names=['subject_scan'],
                                      output_names=['subject_id', 'scan_id'],
                                      function=split_subject_scan_list),
                             name='splitSubjectScanList_rf')

# Set up datasource node for response function wf
datasource_rf = pe.Node(io.DataGrabber(infields=['subject_id', 'scan_id'], outfields=list(info.keys())),
                                    name='datasource_rf')
datasource_rf.inputs.base_directory = cwd
datasource_rf.inputs.template = "%s/%s/%s/%s/%s/%s/%s"
datasource_rf.inputs.field_template = {
    'dwi_eddy_file': '%s/%s/%s/%s/%s/%s/%s',
    'bvec_file': '%s/%s/%s/%s/%s/%s/%s',
    'bval_file': '%s/%s/%s/%s/%s/%s/%s',
    'dwi_mask_file': '%s/%s/%s/%s/%s/%s/%s'
}
datasource_rf.inputs.template_args = info
datasource_rf.inputs.sort_filelist = True

# # ---------------------- OUTPUT SINK NODE ----------------------
# print(colours.CGREEN + "Creating Sink Node." + colours.CEND)

# createSubjectScanContainer = pe.Node(niu.Function(input_names=['subject_scan'],
#                                               output_names=['container'],
#                                               function=create_subject_scan_container),
#                                      name='createSubjectScanContainer')

# # Set up sink node where all output is stored in subject folder
# datasink = pe.Node(io.DataSink(parameterization=False), name='datasink')
# datasink.inputs.base_directory = res_dir

# ---------------------- PROCESSING NODES ----------------------
print(colours.CGREEN + "Creating Processing Nodes." + colours.CEND)

# Response function estimation
response_sd = pe.Node(mrtrix3.ResponseSD(), name='response_sd')
response_sd.inputs.algorithm = 'dhollander'
response_sd.inputs.wm_file = 'wm.txt'
response_sd.inputs.gm_file = 'gm.txt'
response_sd.inputs.csf_file = 'csf.txt'

# Join nodes for merging response functions
join_wm_responses = pe.JoinNode(niu.IdentityInterface(fields=['wm_files']),
                             joinsource='infosource_rf',
                             joinfield='wm_files',
                             name='join_wm_responses')
join_gm_responses = pe.JoinNode(niu.IdentityInterface(fields=['gm_files']),
                             joinsource='infosource_rf',
                             joinfield='gm_files',
                             name='join_gm_responses')
join_csf_responses = pe.JoinNode(niu.IdentityInterface(fields=['csf_files']),
                              joinsource='infosource_rf',
                              joinfield='csf_files',
                              name='join_csf_responses')

# Average response function across subjects for each tissue type
response_mean_wm = pe.Node(ResponseMean(), name='response_mean_wm')
response_mean_wm.inputs.out_txt = 'avg_wm.txt'
response_mean_gm = pe.Node(ResponseMean(), name='response_mean_gm')
response_mean_gm.inputs.out_txt = 'avg_gm.txt'
response_mean_csf = pe.Node(ResponseMean(), name='response_mean_csf')
response_mean_csf.inputs.out_txt = 'avg_csf.txt'


# ---------------------- CREATE WORKFLOW AND CONNECT NODES ----------------------
print(colours.CGREEN + 'Connecting Nodes.\n' + colours.CEND)

# Response function workflow
workflow = pe.Workflow(name='rf_wf', base_dir=wf_dir)
workflow.connect([
# ---------------------- INPUT/OUTPUT STRUCTURE (RF)

    (infosource_rf, splitSubjectScanList_rf, [('subject_scan', 'subject_scan')]),

    # Connect to datasource
    (splitSubjectScanList_rf, datasource_rf, [('subject_id', 'subject_id')]),
    (splitSubjectScanList_rf, datasource_rf, [('scan_id', 'scan_id')]),

# ---------------------- FIBER ORIENTATION DISTRIBUTION (RF average)
    
    # Estimate response functions for different tissue types
    (datasource_rf, response_sd, [('dwi_eddy_file', 'in_file'),
                               ('bvec_file', 'in_bvec'),
                               ('bval_file', 'in_bval'),
                               ('dwi_mask_file', 'in_mask')]),
    
    (response_sd, join_wm_responses, [('wm_file', 'wm_files')]),
    (response_sd, join_gm_responses, [('gm_file', 'gm_files')]),
    (response_sd, join_csf_responses, [('csf_file', 'csf_files')]),

    # Connect JoinNodes to ResponseMean nodes
    (join_wm_responses, response_mean_wm, [('wm_files', 'in_txts')]),
    (join_gm_responses, response_mean_gm, [('gm_files', 'in_txts')]),
    (join_csf_responses, response_mean_csf, [('csf_files', 'in_txts')]),
])

if __name__ == '__main__':
    workflow.write_graph(graph2use='orig')
    workflow.run(plugin=processing_type, plugin_args=plugin_args)
