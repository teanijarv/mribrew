# %%
import os
import pandas as pd
from nipype import config, logging
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces import io, mrtrix, mrtrix3

from mribrew.utils import (colours, split_subject_scan_list, create_subject_scan_container)

# ---------------------- Set up directory structures and constant variables ----------------------
cwd = os.getcwd()
misc_dir = os.path.join(cwd, 'misc')
data_dir = os.path.join(cwd, 'data')
proc_dir = os.path.join(data_dir, 'proc', 'dwi_proc')
wf_dir = os.path.join(cwd, 'wf')
res_dir = os.path.join(data_dir, 'res', 'dti')
log_dir = os.path.join(wf_dir, 'log_dti')

wf_name = 'wf_dti'

# // TO-DO: read from CSV & potentially check for similar names (some have _1 or sth in the end)
subject_list = next(os.walk(proc_dir))[1]

# Generate a list of all [subject, scan] sublists and list of controls whose response function will be averaged
subject_scan_list = []
for sub in subject_list:
    scans = next(os.walk(os.path.join(proc_dir, sub)))[1]
    for scan in scans:
        subject_scan_list.append([sub, scan])

print(f'n subjects running: {len(subject_scan_list)}')
print(subject_scan_list)

# %%

# Computational variables
processing_type = 'MultiProc' # or 'Linear'
plugin_args = {
    'n_procs': 6,
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

# convert NII with grad vals/vecs to mif format
nii2mif = pe.Node(mrtrix3.MRConvert(), name='nii2mif')
nii2mif.inputs.out_file = 'dwi.mif'

# exclude b2500 shell (not well-suited for DTI)
dwiextract = pe.Node(mrtrix3.DWIExtract(), name='dwiextract')
dwiextract.inputs.shell = [0, 100, 1000]
dwiextract.inputs.out_file = 'dwi_no_b2500.mif'

# convert dwi to tensor image
dwi2tensor = pe.Node(mrtrix.DWI2Tensor(), name='dwi2tensor')
# dwi2tensor.inputs.out_file = 'dwi_tensor.mif'

# calculate dti metrics
tensor2metrics = pe.Node(mrtrix3.TensorMetrics(), name='tensor2metrics')
tensor2metrics.inputs.out_fa = 'fa.mif'
tensor2metrics.inputs.out_ad = 'ad.mif'
tensor2metrics.inputs.out_adc = 'adc.mif'
tensor2metrics.inputs.out_rd = 'rd.mif'
tensor2metrics.inputs.out_cl = 'cl.mif'
tensor2metrics.inputs.out_cp = 'cp.mif'
tensor2metrics.inputs.out_cs = 'cs.mif'

# ---------------------- CREATE WORKFLOW AND CONNECT NODES ----------------------
print(colours.CGREEN + 'Connecting Nodes.\n' + colours.CEND)

# Response function workflow
workflow = pe.Workflow(name=wf_name, base_dir=wf_dir)
workflow.connect([
# ---------------------- INPUT/OUTPUT STRUCTURE
    
    (infosource, splitSubjectScanList, [('subject_scan', 'subject_scan')]),

    # Connect to datasource
    (splitSubjectScanList, datasource, [('subject_id', 'subject_id')]),
    (splitSubjectScanList, datasource, [('scan_id', 'scan_id')]),

    # Connect to datasink
    (infosource, createSubjectScanContainer, [('subject_scan', 'subject_scan')]),
    (createSubjectScanContainer, datasink, [('container', 'container')]),

# ---------------------- DTI

    # convert to mif
    (datasource, nii2mif, [('dwi_eddy_file', 'in_file'),
                           ('bvec_file', 'in_bvec'),
                           ('bval_file', 'in_bval')]),
    
    # exclude b2500 shell
    (nii2mif, dwiextract, [('out_file', 'in_file')]),

    # perform dti
    (dwiextract, dwi2tensor, [('out_file', 'in_file')]),
    # (datasource, dwi2tensor, [('dwi_mask_file', 'mask')]),

    (dwi2tensor, tensor2metrics, [('tensor', 'in_file')]),

# ---------------------- DATASINK

    # export dti metrics
    (tensor2metrics, datasink, [('out_fa', '@fa'),
                                ('out_ad', '@ad'),
                                ('out_adc', '@adc'),
                                ('out_rd', '@rd'),
                                ('out_cl', '@cl'),
                                ('out_cp', '@cp'),
                                ('out_cs', '@cs')]), 
])

if __name__ == '__main__':
    workflow.write_graph(graph2use='orig')
    workflow.run(plugin=processing_type, plugin_args=plugin_args)

# %%
