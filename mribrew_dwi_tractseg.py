# %%
import os
import pandas as pd
from nipype import config, logging
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces import io, mrtrix, mrtrix3

from mribrew.utils import (colours, split_subject_scan_list, create_subject_scan_container)
from mribrew.tractseg_interface import (RawTractSeg)

# ---------------------- Set up directory structures and constant variables ----------------------
cwd = os.getcwd()
misc_dir = os.path.join(cwd, 'misc')
data_dir = os.path.join(cwd, 'data')
proc_dir = os.path.join(data_dir, 'proc', 'dwi_proc')
wf_dir = os.path.join(cwd, 'wf')
res_dir = os.path.join(data_dir, 'res', 'tractseg')
log_dir = os.path.join(wf_dir, 'log_tractseg')

wf_name = 'wf_tractseg'

# // TO-DO: read from CSV & potentially check for similar names (some have _1 or sth in the end)
subject_list = next(os.walk(proc_dir))[1]

# Generate a list of all [subject, scan] sublists and list of controls whose response function will be averaged
subject_scan_list = []
for sub in subject_list:
    scans = next(os.walk(os.path.join(proc_dir, sub)))[1]
    for scan in scans:
        subject_scan_list.append([sub, scan])

subject_scan_list = [subject_scan_list[0]]

print(f'n subjects running: {len(subject_scan_list)}')
print(subject_scan_list)

# %%

# Computational variables
processing_type = 'MultiProc' # or 'Linear'
plugin_args = {
    'n_procs': 1,
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
    dwi_mask_file=[['data', 'proc', 'dwi_proc', 'subject_id', 'scan_id', 'dwi', 'dwi_mask.nii.gz']],
)

# Set up infosource node
infosource = pe.Node(niu.IdentityInterface(fields=['subject_scan']), name='infosource')
infosource.iterables = [('subject_scan', subject_scan_list)]

splitSubjectScanList = pe.Node(niu.Function(input_names=['subject_scan'],
                                      output_names=['subject_id', 'scan_id'],
                                      function=split_subject_scan_list),
                             name='splitSubjectScanList')

# Set up datasource node
datasource = pe.Node(io.DataGrabber(infields=['subject_id', 'scan_id'], outfields=list(info.keys())),
                                    name='datasource')
datasource.inputs.base_directory = cwd
datasource.inputs.template = "%s/%s/%s/%s/%s/%s/%s"
datasource.inputs.field_template = {
    'dwi_eddy_file': '%s/%s/%s/%s/%s/%s/%s',
    'bvec_file': '%s/%s/%s/%s/%s/%s/%s',
    'bval_file': '%s/%s/%s/%s/%s/%s/%s',
    'dwi_mask_file': '%s/%s/%s/%s/%s/%s/%s',
}
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True

# ---------------------- OUTPUT SINK NODE ----------------------
print(colours.CGREEN + "Creating Sink Node." + colours.CEND)

createSubjectScanContainer = pe.Node(niu.Function(input_names=['subject_scan'],
                                              output_names=['container'],
                                              function=create_subject_scan_container),
                                     name='createSubjectScanContainer')

# Set up sink node where all output is stored in subject folder
datasink = pe.Node(io.DataSink(parameterization=False), name='datasink')
datasink.inputs.base_directory = res_dir

# ---------------------- PROCESSING NODES ----------------------
print(colours.CGREEN + "Creating Processing Nodes." + colours.CEND)


# Tractseg with standard tract definitions
tractSeg = pe.Node(RawTractSeg(), name='tractSeg')
tractSeg.inputs.args = '--raw_diffusion_input --single_output_file'

# Tractseg segmentation with Xtract tract definitions 
tractSegXtract = pe.Node(RawTractSeg(), name='tractSegXtract')
tractSegXtract.inputs.args = '--raw_diffusion_input --single_output_file  --tract_definition xtract'

# # Fit DKI model
# dkifit = pe.Node(DKIfit(), name='dkifit')

# # Estimate DKI metrics in the WM tracts
# # Standard definitions
# tractMetrics = pe.Node(TractMetrics(), name='tractMetrics')
# tractMetrics.inputs.thresh1 = 1.2e-3 # lower thresh for MD in WM lesion
# tractMetrics.inputs.thresh2 = 2.5e-3 # higher thresh for MD in WM lesion
# # Xtract definitions
# tractMetricsXtract = tractMetrics.clone(name='tractMetricsXtract')

# ---------------------- CREATE WORKFLOW AND CONNECT NODES ----------------------
print(colours.CGREEN + 'Connecting Nodes.\n' + colours.CEND)

# Response function workflow
workflow = pe.Workflow(name=wf_name, base_dir=wf_dir)
workflow.connect([
# ---------------------- INPUT/OUTPUT STRUCTURE (all subjects)
    
    (infosource, splitSubjectScanList, [('subject_scan', 'subject_scan')]),

    # Connect to datasource
    (splitSubjectScanList, datasource, [('subject_id', 'subject_id')]),
    (splitSubjectScanList, datasource, [('scan_id', 'scan_id')]),

    # Connect to datasink
    (infosource, createSubjectScanContainer, [('subject_scan', 'subject_scan')]),
    (createSubjectScanContainer, datasink, [('container', 'container')]),

# ---------------------- WM PARCELLATION WITH TRACTSEG
    (datasource, tractSeg, [('dwi_eddy_file', 'in_file'),
                            ('bval_file', 'in_bvals'), 
                            ('bvec_file', 'in_bvecs'),
                            ('dwi_mask_file', 'in_mask')]),
    # (datasource, tractSegXtract, [('dwi_eddy_file', 'in_file'),
    #                               ('bval_file', 'in_bvals'), 
    #                               ('bvec_file', 'in_bvecs'),
    #                               ('dwi_mask_file', 'in_mask')]),

# ---------------------- DKI FITTING
    # (datasource, dkifit, [('dwi_eddy_file', 'in_file'),
    #                         ('bval_file', 'in_bvals'), 
    #                         ('bvec_file', 'in_bvecs'),
    #                         ('dwi_mask_file', 'in_mask')]),

# ---------------------- DKI METRICS IN WM TRACTS
    # (infosource, tractMetrics, [('subject_id',  'subject_id')]),
    # (tractSeg, tractMetrics, [('out_binary_atlas', 'in_binary_atlas')]),
    # (dkifit, tractMetrics, [('out_fa', 'in_fa'), ('out_md', 'in_md'),('out_mk', 'in_mk'), ('out_ak', 'in_ak'),('out_rk', 'in_rk'),('out_rd', 'in_rd'),('out_ad', 'in_ad')]),

    # (infosource, tractMetricsXtract, [('subject_id',  'subject_id')]),
    # (tractSegXtract, tractMetricsXtract, [('out_binary_atlas', 'in_binary_atlas')]),
    # (dkifit, tractMetricsXtract, [('out_fa', 'in_fa'), ('out_md', 'in_md'),('out_mk', 'in_mk'), ('out_ak', 'in_ak'),('out_rk', 'in_rk'),('out_rd', 'in_rd'),('out_ad', 'in_ad')]),

# ---------------------- DATASINK
    # (dkifit, datasink, [('out_fa', 'tractseg.dki.@fa'), ('out_md', 'tractseg.dki.@md'),('out_mk', 'tractseg.dki.@mk'), ('out_ak', 'tractseg.dki.@ak'),
    # ('out_rk', 'tractseg.dki.@rk'),('out_rd', 'tractseg.dki.@rd'),('out_ad', 'tractseg.dki.@ad')]), 
    # (dkifit, QC_MD, [('out_md', 'in_md')]),
])

# Run the script and generate a graph of the workflow
if __name__ == '__main__':
    workflow.write_graph(graph2use='orig')
    workflow.run(plugin=processing_type, plugin_args=plugin_args)
# %%
