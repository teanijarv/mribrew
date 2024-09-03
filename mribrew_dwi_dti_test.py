# %%
import os
import pandas as pd
from nipype import config, logging
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces import io, mrtrix, mrtrix3

from mribrew.utils import (colours, split_subject_scan_list, create_subject_scan_container)
from mribrew.dwiproc_interface import adjustBval

# ---------------------- Set up directory structures and constant variables ----------------------
cwd = os.getcwd()
misc_dir = os.path.join(cwd, 'misc')
data_dir = os.path.join(cwd, 'data')
proc_dir = os.path.join(data_dir, 'proc', 'dwi_proc')
wf_dir = os.path.join(cwd, 'wf')
res_dir = os.path.join(data_dir, 'res', 'dti')
log_dir = os.path.join(wf_dir, 'log_dti')

wf_name = 'wf_dti_b1000b2500'

# // TO-DO: read from CSV & potentially check for similar names (some have _1 or sth in the end)
subject_list = next(os.walk(proc_dir))[1]

# Generate a list of all [subject, scan] sublists and list of controls whose response function will be averaged
subject_scan_list = []
for sub in subject_list:
    scans = next(os.walk(os.path.join(proc_dir, sub)))[1]
    for scan in scans:
        subject_scan_list.append([sub, scan])

print(f'n subjects running: {len(subject_scan_list)}')
# subject_scan_list = [['BOF112_BioFINDER2_216', '20190928_1']]
print(subject_scan_list)


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

# %%

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



# Adjust b0 to b-100
adjustb0 = pe.Node(adjustBval(), name='adjustb0')
adjustb0.inputs.valold = 0
adjustb0.inputs.valnew = -100

# Adjust b1000 to b0
adjustb1000 = pe.Node(adjustBval(), name='adjustb1000')
adjustb1000.inputs.valold = 1000
adjustb1000.inputs.valnew = 0

# Adjust b2500 to b1000
adjustb2500 = pe.Node(adjustBval(), name='adjustb2500')
adjustb2500.inputs.valold = 2500
adjustb2500.inputs.valnew = 1000

# include only the fake b0 and b1000
dwiextract = pe.Node(mrtrix3.DWIExtract(), name='dwiextract')
dwiextract.inputs.shell = [0, 1000]
dwiextract.inputs.out_file = 'dwi_fakeb0b1000.mif'

# convert NII with grad vals/vecs to mif format
nii2mif = pe.Node(mrtrix3.MRConvert(), name='nii2mif')
nii2mif.inputs.out_file = 'dwi_fakebvals.mif'

# convert dwi to tensor image
dwi2tensor = pe.Node(mrtrix.DWI2Tensor(), name='dwi2tensor')

# export MD
tensor2metrics = pe.Node(mrtrix3.TensorMetrics(), name='tensor2metrics')
tensor2metrics.inputs.out_adc = 'md_b1000b2500.mif'

# ---------------------- CREATE WORKFLOW AND CONNECT NODES ----------------------
print(colours.CGREEN + 'Connecting Nodes.\n' + colours.CEND)

# Response function workflow
workflow = pe.Workflow(name=wf_name, base_dir=wf_dir)
workflow.connect([
# ---------------------- INPUT/OUTPUT STRUCTURE
    
    (infosource, splitSubjectScanList, [('subject_scan', 'subject_scan')]),

    # datasource
    (splitSubjectScanList, datasource, [('subject_id', 'subject_id')]),
    (splitSubjectScanList, datasource, [('scan_id', 'scan_id')]),

    # datasink
    (infosource, createSubjectScanContainer, [('subject_scan', 'subject_scan')]),
    (createSubjectScanContainer, datasink, [('container', 'container')]),

# ---------------------- DTI

    # adjust b0 -> b-100 -> b1000 -> b0 and b2500 -> b1000
    (datasource, adjustb0, [('bval_file', 'in_bval')]), ### issue with using adjustbvalue multiple times; would give empty file
    # (adjustb0, adjustb1000, [('out_bval', 'in_bval')]),
#     (adjustb1000, adjustb2500, [('out_bval', 'in_bval')]),

#     # convert to mif
#     (datasource, nii2mif, [('dwi_eddy_file', 'in_file'),
#                            ('bvec_file', 'in_bvec')]),
#     (adjustb2500, nii2mif, [('out_bval', 'in_bval')]),

#     # exclude b100 and b-100 (ie include only fake b0 and b1000)
#     (nii2mif, dwiextract, [('out_file', 'in_file')]),

#     # perform dti on and extract metrics
#     (dwiextract, dwi2tensor, [('out_file', 'in_file')]),
#     (dwi2tensor, tensor2metrics, [('tensor', 'in_file')]),

# # ---------------------- DATASINK

#     # export dti metrics
#     (tensor2metrics, datasink, [('out_adc', '@md_b1000b2500')]), 
])

if __name__ == '__main__':
    workflow.write_graph(graph2use='orig')
    workflow.run(plugin=processing_type, plugin_args=plugin_args)

# %%
