# NEWEST VERSION
import os
import pandas as pd
from nipype import config, logging
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces import io, mrtrix, mrtrix3, fsl

from mribrew.utils import (colours, split_subject_scan_list, create_subject_scan_container)
from mribrew.act_interface import (MRTransform, MTNormalise, Generate5tt, SIFT2, BuildConnectome, TckSample)

# ---------------------- Set up directory structures and constant variables ----------------------
job_name = 'act_b3'

cwd = os.getcwd()
misc_dir = os.path.join(cwd, 'misc')
data_dir = os.path.join(cwd, 'data')
wf_dir = '/proj/sens2023026/to8050an/wf/'
res_dir = os.path.join(data_dir, 'res', 'act')
log_dir = os.path.join(wf_dir, f'log_{job_name}')

fs_lut_file = os.path.join(misc_dir, 'fs_labels', 'FreeSurferColorLUT.txt')
fs_default_file = os.path.join(misc_dir, 'fs_labels', 'fs_default.txt')
rf_wm = os.path.join(misc_dir, 'hc_rf', 'avg_wm.txt')
rf_gm = os.path.join(misc_dir, 'hc_rf', 'avg_gm.txt')
rf_csf = os.path.join(misc_dir, 'hc_rf', 'avg_csf.txt')

# // TO-DO: read from CSV & potentially check for similar names (some have _1 or sth in the end)
subject_list = next(os.walk(os.path.join(data_dir, 'proc', 'dwi_proc')))[1]

# Generate a list of all [subject, scan] sublists
subject_scan_list = []
for sub in subject_list:
    scans = next(os.walk(os.path.join(data_dir, 'proc', 'dwi_proc', sub)))[1]
    for scan in scans:
        subject_scan_list.append([sub, scan])

# Select all non-HC and select only one batch
#subject_scan_list = [[sub, scan] for [sub, scan] in subject_scan_list]
batch_len = len(subject_scan_list) // 5
batch1 = subject_scan_list[0:batch_len]
batch2 = subject_scan_list[1*batch_len:2*batch_len]
batch3 = subject_scan_list[2*batch_len:3*batch_len]
batch4 = subject_scan_list[3*batch_len:4*batch_len]
batch5 = subject_scan_list[4*batch_len:]
subject_scan_list = batch3
print(f'(b3) n subjects running: {len(subject_scan_list)}')

# Computational variables
processing_type = 'MultiProc' # or 'Linear'

# ACT parameters
n_tracks = 10000000

plugin_args = {
    'n_procs': 8,
    'raise_insufficient': True,
    #'scheduler': 'mem_thread',  # Prioritize jobs by memory consumption then nr of threads
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
    freesurfer_dir=[['data', 'proc', 'freesurfer', 'subject_id', 'scan_id']], 
    t1_file=[['data', 'proc', 'freesurfer', 'subject_id', 'scan_id', 'mri', 'T1.mgz']],
    parc_file=[['data', 'proc', 'freesurfer', 'subject_id', 'scan_id', 'mri', 'aparc+aseg.mgz']]
)

## ACT WORKFLOW

# Set up infosource node for ACT for all subjects
infosource = pe.Node(niu.IdentityInterface(fields=['subject_scan']), name='infosource')
infosource.iterables = [('subject_scan', subject_scan_list)]
infosource.inputs.fs_lut_file = fs_lut_file
infosource.inputs.fs_default_file = fs_default_file

splitSubjectScanList = pe.Node(niu.Function(input_names=['subject_scan'],
                                      output_names=['subject_id', 'scan_id'],
                                      function=split_subject_scan_list),
                             name='splitSubjectScanList')

# Set up datasource node for ACT
datasource = pe.Node(io.DataGrabber(infields=['subject_id', 'scan_id'], outfields=list(info.keys())),
                                    name='datasource')
datasource.inputs.base_directory = cwd
datasource.inputs.template = "%s/%s/%s/%s/%s/%s/%s"
datasource.inputs.field_template = {
    'dwi_eddy_file': '%s/%s/%s/%s/%s/%s/%s',
    'bvec_file': '%s/%s/%s/%s/%s/%s/%s',
    'bval_file': '%s/%s/%s/%s/%s/%s/%s',
    'dwi_mask_file': '%s/%s/%s/%s/%s/%s/%s',
    'freesurfer_dir': '%s/%s/%s/%s/%s',
    't1_file': '%s/%s/%s/%s/%s/%s/%s',
    'parc_file': '%s/%s/%s/%s/%s/%s/%s'
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

# Fiber orientation distribution estimation
dwi2fod = pe.Node(mrtrix3.ConstrainedSphericalDeconvolution(), name='dwi2fod')
dwi2fod.inputs.algorithm = 'msmt_csd'
dwi2fod.inputs.wm_txt = rf_wm
dwi2fod.inputs.gm_txt = rf_gm
dwi2fod.inputs.csf_txt = rf_csf
dwi2fod.inputs.wm_odf = 'wmfod.mif'
dwi2fod.inputs.gm_odf = 'gmfod.mif'
dwi2fod.inputs.csf_odf = 'csffod.mif'

# Intensity normalisation
mtn = pe.Node(MTNormalise(), name='mtn')
mtn.inputs.out_file_wm = 'wmfod_norm.mif'
mtn.inputs.out_file_gm = 'gmfod_norm.mif'
mtn.inputs.out_file_csf = 'csffod_norm.mif'

# 5-tissue-types image generation
gen5tt = pe.Node(Generate5tt(), name='gen5tt')
gen5tt.inputs.algorithm = 'hsvs'
#gen5tt.inputs.nthreads = 12
gen5tt.inputs.out_file = '5tt_nocoreg.mif'

# Extract b0 image
dwiextract = pe.Node(mrtrix3.DWIExtract(), name='dwiextract')
dwiextract.inputs.bzero = True
dwiextract.inputs.out_file = 'b0vols.mif'

# Average all b0 images
mrmath = pe.Node(mrtrix3.MRMath(), name='mrmath')
mrmath.inputs.operation = 'mean'
mrmath.inputs.axis = 3
mrmath.inputs.out_file = 'b0vols_mean.mif'

# Covert T1 and average b0 images to NIfTI
t1_nii = pe.Node(mrtrix3.MRConvert(), name='t1_nii')
t1_nii.inputs.out_file = 't1.nii.gz'
b0vols_mean_nii = pe.Node(mrtrix3.MRConvert(), name='b0vols_mean_nii')
b0vols_mean_nii.inputs.out_file = 'b0vols_mean.nii.gz'

# Flirt
flt = pe.Node(fsl.FLIRT(), name='flt')
flt.inputs.dof = 6
flt.inputs.out_matrix_file = 'diff2struct_fsl.mat'

# Transformconvert
transconv = pe.Node(mrtrix3.TransformFSLConvert(), name='transconv')
transconv.inputs.flirt_import = True
transconv.inputs.out_transform = 'diff2struct_mrtrix.txt'

# Coregister T1 and 5TT to DWI image
mrxform2_t1 = pe.Node(MRTransform(), name='mrxform2_t1')
mrxform2_t1.inputs.invert = True
mrxform2_t1.inputs.out_file = 'T1_coreg.mif'
mrxform2_5tt = pe.Node(MRTransform(), name='mrxform2_5tt')
mrxform2_5tt.inputs.invert = True
mrxform2_5tt.inputs.out_file = '5tt_coreg.mif'

# Convert parc file to MIF, replace labels, and coregister to DWI
parc_mif = pe.Node(mrtrix3.MRConvert(), name='parc_mif')
parc_mif.inputs.args = '-datatype uint32'
parc_mif.inputs.out_file = 'aparcaseg.mif'

labelconv = pe.Node(mrtrix3.LabelConvert(), name='labelconv')
labelconv.inputs.out_file = 'aparcaseg_label.mif'

mrxform2_parc = pe.Node(MRTransform(), name='mrxform2_parc')
mrxform2_parc.inputs.args = '-datatype uint32'
mrxform2_parc.inputs.invert = True
mrxform2_parc.inputs.out_file = 'aparcaseg_label_coreg.mif'

# Anatomically constrained tractography (ACT)
tk = pe.Node(mrtrix3.Tractography(), name='tk')
#tk.inputs.args = '-nthreads 12'
tk.inputs.backtrack = True
tk.inputs.select = n_tracks
tk.inputs.out_file = f'tracks_{n_tracks}.tck'

# SIFT2
sift2 = pe.Node(SIFT2(), name='sift2')
#sift2.inputs.args = '-nthreads 12'
sift2.inputs.out_file = 'sift2_tck_weights.txt'

# Tractograms to structural connectomes
tck2conn_sift2 = pe.Node(BuildConnectome(), name='tck2conn_sift2')
tck2conn_sift2.inputs.args = '-symmetric -zero_diagonal -stat_edge sum'
tck2conn_sift2.inputs.out_assignments = 'assignments.txt'
tck2conn_sift2.inputs.out_file = f'sc_sift2_{int(n_tracks)}.csv'

# convert NII with grad vals/vecs to mif format
nii2mif = pe.Node(mrtrix3.MRConvert(), name='nii2mif')
nii2mif.inputs.out_file = 'dwi.mif'

# exclude b2500 shell (not well-suited for DTI)
dwiextract2 = pe.Node(mrtrix3.DWIExtract(), name='dwiextract2')
dwiextract2.inputs.shell = [0, 100, 1000]
dwiextract2.inputs.out_file = 'dwi_no_b2500.mif'

# convert dwi to tensor image
dwi2tensor = pe.Node(mrtrix.DWI2Tensor(), name='dwi2tensor')
# dwi2tensor.inputs.out_file = 'dwi_tensor.mif'

# calculate dti metrics
tensor2metrics = pe.Node(mrtrix3.TensorMetrics(), name='tensor2metrics')
tensor2metrics.inputs.out_fa = 'fa.mif'
tensor2metrics.inputs.out_ad = 'ad.mif'
tensor2metrics.inputs.out_adc = 'adc.mif'
tensor2metrics.inputs.out_rd = 'rd.mif'
# tensor2metrics.inputs.out_cl = 'cl.mif'
# tensor2metrics.inputs.out_cp = 'cp.mif'
# tensor2metrics.inputs.out_cs = 'cs.mif'

# FA tcksample + tck2connectome
tcksample_fa = pe.Node(TckSample(), name='tcksample_fa')
tcksample_fa.inputs.args = '-stat_tck mean'
tcksample_fa.inputs.out_samples = 'tck_fa.csv'
tck2conn_fa = pe.Node(BuildConnectome(), name='tck2conn_fa')
tck2conn_fa.inputs.args = '-symmetric -zero_diagonal -stat_edge mean'
tck2conn_fa.inputs.out_file = f'sc_fa_{int(n_tracks)}.csv'

# AD tcksample + tck2connectome
tcksample_ad = pe.Node(TckSample(), name='tcksample_ad')
tcksample_ad.inputs.args = '-stat_tck mean'
tcksample_ad.inputs.out_samples = 'tck_ad.csv'
tck2conn_ad = pe.Node(BuildConnectome(), name='tck2conn_ad')
tck2conn_ad.inputs.args = '-symmetric -zero_diagonal -stat_edge mean'
tck2conn_ad.inputs.out_file = f'sc_ad_{int(n_tracks)}.csv'

# ADC tcksample + tck2connectome
tcksample_adc = pe.Node(TckSample(), name='tcksample_adc')
tcksample_adc.inputs.args = '-stat_tck mean'
tcksample_adc.inputs.out_samples = 'tck_adc.csv'
tck2conn_adc = pe.Node(BuildConnectome(), name='tck2conn_adc')
tck2conn_adc.inputs.args = '-symmetric -zero_diagonal -stat_edge mean'
tck2conn_adc.inputs.out_file = f'sc_adc_{int(n_tracks)}.csv'

# RD tcksample + tck2connectome
tcksample_rd = pe.Node(TckSample(), name='tcksample_rd')
tcksample_rd.inputs.args = '-stat_tck mean'
tcksample_rd.inputs.out_samples = 'tck_rd.csv'
tck2conn_rd = pe.Node(BuildConnectome(), name='tck2conn_rd')
tck2conn_rd.inputs.args = '-symmetric -zero_diagonal -stat_edge mean'
tck2conn_rd.inputs.out_file = f'sc_rd_{int(n_tracks)}.csv'

# ---------------------- CREATE WORKFLOW AND CONNECT NODES ----------------------
print(colours.CGREEN + 'Connecting Nodes.\n' + colours.CEND)

# Response function workflow
workflow = pe.Workflow(name=f'wf_{job_name}', base_dir=wf_dir)
workflow.connect([

# ---------------------- INPUT/OUTPUT STRUCTURE (all subjects)
    
    (infosource, splitSubjectScanList, [('subject_scan', 'subject_scan')]),

    # Connect to datasource
    (splitSubjectScanList, datasource, [('subject_id', 'subject_id')]),
    (splitSubjectScanList, datasource, [('scan_id', 'scan_id')]),

    # Connect to datasink
    (infosource, createSubjectScanContainer, [('subject_scan', 'subject_scan')]),
    (createSubjectScanContainer, datasink, [('container', 'container')]),

# ---------------------- FIBER ORIENTATION DISTRIBUTION (average used on all subjects)

    # Estimate the orientation of all fibers crossing every voxel
    (datasource, dwi2fod, [('dwi_eddy_file', 'in_file'),
                           ('dwi_mask_file', 'mask_file'),
                           ('bvec_file', 'in_bvec'),
                           ('bval_file', 'in_bval')]),

    # Correct for global intensity differences
    (dwi2fod, mtn, [('wm_odf', 'wm_fod'),
                    ('gm_odf', 'gm_fod'),
                    ('csf_odf', 'csf_fod')]),
    (datasource, mtn, [('dwi_mask_file', 'mask')]),

# ---------------------- 5 TISSUE TYPES (5TT & T1 images with coregistration to DWI)

    # Generate a 5TT image
    (datasource, gen5tt, [('freesurfer_dir', 'in_file')]),
    # Extract b0 image
    (datasource, dwiextract, [('dwi_eddy_file', 'in_file'),
                              ('bvec_file', 'in_bvec'),
                              ('bval_file', 'in_bval')]),
    # Average all b0 volumes
    (dwiextract, mrmath, [('out_file', 'in_file')]),
    # Convert Freesurfer T1 image and average b0 to NIfTI
    (datasource, t1_nii, [('t1_file', 'in_file')]),
    (mrmath, b0vols_mean_nii, [('out_file', 'in_file')]),
    # Flirt b0 to T1
    (t1_nii, flt, [('out_file', 'reference')]),
    (b0vols_mean_nii, flt, [('out_file', 'in_file')]),
    # transformconvert flirt import
    (flt, transconv, [('out_matrix_file', 'in_transform')]),
    (datasource, transconv, [('t1_file', 'reference')]),
    (b0vols_mean_nii, transconv, [('out_file', 'in_file')]),
    # Coregister T1 to DWI
    (datasource, mrxform2_t1, [('t1_file', 'in_files')]),
    (transconv, mrxform2_t1, [('out_transform', 'linear_transform')]),
    # Coregister 5TT to DWI
    (gen5tt, mrxform2_5tt, [('out_file', 'in_files')]),
    (transconv, mrxform2_5tt, [('out_transform', 'linear_transform')]),

# ---------------------- PARCELLATION PREP (parcellation image label edits & coregister to DWI)

    (datasource, parc_mif, [('parc_file', 'in_file')]),

    (parc_mif, labelconv, [('out_file', 'in_file')]),
    (infosource, labelconv, [('fs_lut_file', 'in_lut')]),
    (infosource, labelconv, [('fs_default_file', 'in_config')]),

    # Coregister parcellations to DWI
    (labelconv, mrxform2_parc, [('out_file', 'in_files')]),
    # (datasource, mrxform_parc, [('parc_file', 'in_files')]),
    (transconv, mrxform2_parc, [('out_transform', 'linear_transform')]),

# ---------------------- ANATOMICALLY CONSTRAINED TRACTOGRAPHY (ACT) - SIFT2

    # Perform ACT using 5TT image and determine seed points dynamically using WM FOD
    (mrxform2_5tt, tk, [('out_file', 'act_file')]),
    (mtn, tk, [('out_file_wm', 'seed_dynamic')]),
    (mtn, tk, [('out_file_wm', 'in_file')]),
    # SIFT2
    (mrxform2_5tt, sift2, [('out_file', 'act_file')]),
    (tk, sift2, [('out_file', 'in_tracks')]),
    (mtn, sift2, [('out_file_wm', 'in_fod')]),
    # Generate structural connectomes of SIFT2 tractograms
    (tk, tck2conn_sift2, [('out_file', 'in_file')]),
    (sift2, tck2conn_sift2, [('out_weights', 'in_weights')]),
    (mrxform2_parc, tck2conn_sift2, [('out_file', 'in_parc')]),

# ---------------------- DTI

    # convert to mif
    (datasource, nii2mif, [('dwi_eddy_file', 'in_file'),
                           ('bvec_file', 'in_bvec'),
                           ('bval_file', 'in_bval')]),
    
    # exclude b2500 shell
    (nii2mif, dwiextract2, [('out_file', 'in_file')]),

    # perform dti
    (dwiextract2, dwi2tensor, [('out_file', 'in_file')]),
    # (datasource, dwi2tensor, [('dwi_mask_file', 'mask')]),

    # calculate metrics
    (dwi2tensor, tensor2metrics, [('tensor', 'in_file')]),

    # estimate FA weights on ACT and find structural connectome
    (tk, tcksample_fa, [('out_file', 'in_tracks')]),
    (tensor2metrics, tcksample_fa, [('out_fa', 'in_img')]),
    (tk, tck2conn_fa, [('out_file', 'in_file')]),
    (sift2, tck2conn_fa, [('out_weights', 'in_weights')]), # new
    (tcksample_fa, tck2conn_fa, [('out_samples', 'scale_file')]), # new
    (mrxform2_parc, tck2conn_fa, [('out_file', 'in_parc')]),

    # estimate AD weights on ACT and find structural connectome
    (tk, tcksample_ad, [('out_file', 'in_tracks')]),
    (tensor2metrics, tcksample_ad, [('out_ad', 'in_img')]),
    (tk, tck2conn_ad, [('out_file', 'in_file')]),
    (sift2, tck2conn_ad, [('out_weights', 'in_weights')]), # new
    (tcksample_ad, tck2conn_ad, [('out_samples', 'scale_file')]), # new
    (mrxform2_parc, tck2conn_ad, [('out_file', 'in_parc')]),

    # estimate ADC weights on ACT and find structural connectome
    (tk, tcksample_adc, [('out_file', 'in_tracks')]),
    (tensor2metrics, tcksample_adc, [('out_adc', 'in_img')]),
    (tk, tck2conn_adc, [('out_file', 'in_file')]),
    (sift2, tck2conn_adc, [('out_weights', 'in_weights')]), # new
    (tcksample_adc, tck2conn_adc, [('out_samples', 'scale_file')]), # new
    (mrxform2_parc, tck2conn_adc, [('out_file', 'in_parc')]),

    # estimate RD weights on ACT and find structural connectome
    (tk, tcksample_rd, [('out_file', 'in_tracks')]),
    (tensor2metrics, tcksample_rd, [('out_rd', 'in_img')]),
    (tk, tck2conn_rd, [('out_file', 'in_file')]),
    (sift2, tck2conn_rd, [('out_weights', 'in_weights')]), # new
    (tcksample_rd, tck2conn_rd, [('out_samples', 'scale_file')]), # new
    (mrxform2_parc, tck2conn_rd, [('out_file', 'in_parc')]),

# ---------------------- DATASINK (save tractograms and structural connectomes)

    # Save structural connectomes
    # (tk, datasink, [('out_file', '@tracks')]), 
    (tck2conn_sift2, datasink, [('out_file', '@sc_sift2')]), 
    # (tck2conn_sift2, datasink, [('out_assignments', '@assignments')]), 

    (tck2conn_fa, datasink, [('out_file', '@sc_fa')]), 
    (tck2conn_ad, datasink, [('out_file', '@sc_ad')]), 
    (tck2conn_adc, datasink, [('out_file', '@sc_adc')]),
    (tck2conn_rd, datasink, [('out_file', '@sc_rd')]),
    
])

if __name__ == '__main__':
    workflow.write_graph(graph2use='orig')
    workflow.run(plugin=processing_type, plugin_args=plugin_args)
    #workflow.run(plugin='SLURMGraph', plugin_args={'dont_resubmit_completed_jobs':True})
