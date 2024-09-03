# %%
import os
import pandas as pd
from nipype import config, logging
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces import io, mrtrix, mrtrix3

from mribrew.utils import (colours, split_subject_scan_list, create_subject_scan_container)
from mribrew.tractseg_interface import (RawTractSeg, TractMetrics, QCgetBrainMaskVol,
                                        QC_MD, QC_wm_vol, sessionSummaryCSV, cohortSummaryCSV)

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

subject_scan_list = [subject_scan_list[0], subject_scan_list[1]]

print(f'n subjects running: {len(subject_scan_list)}')
print(subject_scan_list)

# %%

# Computational variables
processing_type = 'MultiProc' # or 'Linear'
plugin_args = {
    'n_procs': 8,
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
dwiextract2 = pe.Node(mrtrix3.DWIExtract(), name='dwiextract2')
dwiextract2.inputs.shell = [0, 100, 1000]
dwiextract2.inputs.out_file = 'dwi_no_b2500.mif'

# convert dwi to tensor image
dwi2tensor = pe.Node(mrtrix.DWI2Tensor(), name='dwi2tensor')

# calculate dti metrics
tensor2metrics = pe.Node(mrtrix3.TensorMetrics(), name='tensor2metrics')
tensor2metrics.inputs.out_adc = 'md.nii.gz'
tensor2metrics.inputs.out_fa = 'fa.nii.gz'
tensor2metrics.inputs.out_ad = 'ad.nii.gz'
tensor2metrics.inputs.out_rd = 'rd.nii.gz'

# tractseg
# standard definitions
tractSeg = pe.Node(RawTractSeg(), name='tractSeg')
tractSeg.inputs.args = '--raw_diffusion_input --single_output_file --csd_type csd_msmt'
tractSeg.inputs.tract_definition = 'TractQuerier+'
# xtract definitions 
tractSegXtract = pe.Node(RawTractSeg(), name='tractSegXtract')
tractSegXtract.inputs.args = '--raw_diffusion_input --single_output_file --csd_type csd_msmt'
tractSegXtract.inputs.tract_definition = 'xtract'

# estimate DTI metrics in the WM tracts
# standard definitions
tractMetrics = pe.Node(TractMetrics(), name='tractMetrics')
tractMetrics.inputs.thresh1 = 1.2e-3 # lower thresh for MD in WM lesion
tractMetrics.inputs.thresh2 = 2.5e-3 # higher thresh for MD in WM lesion
# xtract definitions
tractMetricsXtract = tractMetrics.clone(name='tractMetricsXtract')

# ---------------------- QC NODES ----------------------
# mask volume check
QC_mask = pe.Node(QCgetBrainMaskVol(), name='QC_mask')

# MD % HIGHER THAN 2.5 check
QC_MD = pe.Node(QC_MD(), name='QC_MD')
QC_MD.inputs.in_thres = 2.5e-3

# WM volume check
QC_wm_vol = pe.Node(QC_wm_vol(), name='QC_wm_vol')

# ---------------------- SUMMARY NODES ----------------------
# session summary
# standard definitions
sessionSummary = pe.Node(sessionSummaryCSV(), name='sessionSummary') 
sessionSummary.inputs.out_filename = 'tractseg_summary.csv'
# xtract definitions
sessionSummaryXtract = pe.Node(sessionSummaryCSV(), name = 'sessionSummaryXtract') 
sessionSummaryXtract.inputs.out_filename = 'tractseg_summary_xtract.csv'

# cohort summary   
# standard definitions 
cohortSummary = pe.Node(cohortSummaryCSV(), name = 'cohortSummary')
cohortSummary.inputs.in_csv_c = os.path.join(res_dir, 'tractseg_cohort_summary.csv')
# xtract definitions
cohortSummaryXtract = pe.Node(cohortSummaryCSV(), name = 'cohortSummaryXtract')
cohortSummaryXtract.inputs.in_csv_c = os.path.join(res_dir, 'tractseg_xtract_cohort_summary.csv')


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

# ---------------------- DTI
    # convert to mif
    (datasource, nii2mif, [('dwi_eddy_file', 'in_file'),
                           ('bvec_file', 'in_bvec'),
                           ('bval_file', 'in_bval')]),
    
    # exclude b2500 shell
    (nii2mif, dwiextract2, [('out_file', 'in_file')]),

    # perform dti
    (dwiextract2, dwi2tensor, [('out_file', 'in_file')]),

    # calculate metrics
    (dwi2tensor, tensor2metrics, [('tensor', 'in_file')]),

# ---------------------- WM PARCELLATION WITH TRACTSEG
    (datasource, tractSeg, [('dwi_eddy_file', 'in_file'),
                            ('bval_file', 'in_bvals'), 
                            ('bvec_file', 'in_bvecs'),
                            ('dwi_mask_file', 'in_mask')]),
    (datasource, tractSegXtract, [('dwi_eddy_file', 'in_file'),
                                  ('bval_file', 'in_bvals'), 
                                  ('bvec_file', 'in_bvecs'),
                                  ('dwi_mask_file', 'in_mask')]),

# ---------------------- DTI METRICS IN WM TRACTS
    # DTI metrics in WM tracts
    (infosource, tractMetrics, [('subject_scan', 'subject_scan')]),
    (tractSeg, tractMetrics, [('out_binary_atlas', 'in_binary_atlas'),
                              ('out_labels', 'tract_labels')]),
    (tensor2metrics, tractMetrics, [('out_adc', 'in_md'),
                                    ('out_fa', 'in_fa'), 
                                    ('out_ad', 'in_ad'),
                                    ('out_rd', 'in_rd')]),
    
    # DTI metrics in WM tracts (xtract)
    (infosource, tractMetricsXtract, [('subject_scan', 'subject_scan')]),
    (tractSegXtract, tractMetricsXtract, [('out_binary_atlas', 'in_binary_atlas'),
                                          ('out_labels', 'tract_labels')]),
    (tensor2metrics, tractMetricsXtract, [('out_adc', 'in_md'),
                                          ('out_fa', 'in_fa'), 
                                          ('out_ad', 'in_ad'),
                                          ('out_rd', 'in_rd')]),

# ---------------------- QUALITY CONTROL
    # mask check
    (datasource, QC_mask, [('dwi_mask_file', 'in_mask')]),

    # WM volume check
    (tractSeg, QC_wm_vol, [('out_binary_atlas', 'in_file')]),

    # MD check
    (tensor2metrics, QC_MD, [('out_adc', 'in_md')]),
    
# ---------------------- SUMMARY
    # summary for standard tracts
    (QC_mask, sessionSummary, [('out_maskvolume', 'in_maskvolume')]),
    (QC_wm_vol, sessionSummary, [('out_wmvolume', 'in_wmvolume')]),
    (QC_MD, sessionSummary, [('out_perc', 'in_mdperc')]),
    (tractMetrics, sessionSummary, [('out_csv_summary', 'in_csv')]),

    # summary for xtract tracts
    (QC_mask, sessionSummaryXtract, [('out_maskvolume', 'in_maskvolume')]),
    (QC_wm_vol, sessionSummaryXtract, [('out_wmvolume', 'in_wmvolume')]),
    (QC_MD, sessionSummaryXtract, [('out_perc', 'in_mdperc')]),
    (tractMetricsXtract, sessionSummaryXtract, [('out_csv_summary', 'in_csv')]),

# ---------------------- DATASINK

    (sessionSummary, datasink, [('out_csv_summary', '@tractseg')]),
    (sessionSummaryXtract, datasink, [('out_csv_summary', '@tractseg_xtract')]),

    # add session summary to cohort summary
    (sessionSummary, cohortSummary, [('out_csv_summary', 'in_csv_p')]),
    (sessionSummaryXtract, cohortSummaryXtract, [('out_csv_summary', 'in_csv_p')]),

])

# Run the script and generate a graph of the workflow
if __name__ == '__main__':
    workflow.write_graph(graph2use='orig')
    workflow.run(plugin=processing_type, plugin_args=plugin_args)
# %%
