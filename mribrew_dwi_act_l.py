import os
from nipype import config, logging
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces import io, mrtrix3, fsl

from mribrew.utils import (colours, split_subject_scan_list, create_subject_scan_container)
from mribrew.act_interface import Generate5tt, SIFT, SIFT2

# ---------------------- Set up directory structures and constant variables ----------------------
cwd = os.getcwd()
misc_dir = os.path.join(cwd, 'misc')
data_dir = os.path.join(cwd, 'data')
#proc_dir = os.path.join(data_dir, 'proc')
wf_dir = os.path.join(cwd, 'wf')
res_dir = os.path.join(data_dir, 'res', 'act')
log_dir = os.path.join(wf_dir, 'log')

fs_lut_file = os.path.join(misc_dir, 'fs_labels', 'FreeSurferColorLUT.txt')
fs_default_file = os.path.join(misc_dir, 'fs_labels', 'fs_default.txt')

# // TO-DO: read from CSV & potentially check for similar names (some have _1 or sth in the end)
subject_list = next(os.walk(os.path.join(data_dir, 'proc', 'dwi_proc')))[1]
# Generate a list of all (subject, scan) tuples
subject_scan_list = []
for sub in subject_list:
    scans = next(os.walk(os.path.join(data_dir, 'proc', 'dwi_proc', sub)))[1] ## maybe error!
    for scan in scans:
        subject_scan_list.append([sub, scan])

# Computational variables
processing_type = 'MultiProc' # or 'Linear'
total_memory = 6 # in GB
n_cpus = 6 # number of nipype processes to run at the same time
os.environ['OMP_NUM_THREADS'] = str(n_cpus)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cpus)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ACT parameters
n_tracks = 1000000

plugin_args = {
    'n_procs': n_cpus,
    'memory_gb': total_memory,
    'raise_insufficient': True,
    'scheduler': 'mem_thread',  # Prioritize jobs by memory consumption then nr of threads
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
    parc_file=[['data', 'proc', 'freesurfer', 'subject_id', 'scan_id', 'mri', 'aparc+aseg.mgz']],
)

# Set up infosource node
infosource = pe.Node(niu.IdentityInterface(fields=['subject_scan']), name='infosource')
infosource.iterables = [('subject_scan', subject_scan_list)]
infosource.inputs.fs_lut_file = fs_lut_file
infosource.inputs.fs_default_file = fs_default_file

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
    'freesurfer_dir': '%s/%s/%s/%s/%s',
    't1_file': '%s/%s/%s/%s/%s/%s/%s',
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

# Response function estimation
response_sd = pe.Node(mrtrix3.ResponseSD(), name='response_sd')
response_sd.inputs.algorithm = 'dhollander'
response_sd.inputs.wm_file = 'wm.txt'
response_sd.inputs.gm_file = 'gm.txt'
response_sd.inputs.csf_file = 'csf.txt'

# Fiber orientation distribution estimation
dwi2fod = pe.Node(mrtrix3.ConstrainedSphericalDeconvolution(), name='dwi2fod')
dwi2fod.inputs.algorithm = 'msmt_csd'
dwi2fod.inputs.wm_odf = 'wmfod.mif'
dwi2fod.inputs.gm_odf = 'gmfod.mif'
dwi2fod.inputs.csf_odf = 'csffod.mif'

# Intensity normalisation
mtn = pe.Node(mrtrix3.MTNormalise(), name='mtn')
mtn.inputs.out_file_wm = 'wmfod_norm.mif'
mtn.inputs.out_file_gm = 'gmfod_norm.mif'
mtn.inputs.out_file_csf = 'csffod_norm.mif'

# 5-tissue-types image generation
gen5tt = pe.Node(Generate5tt(), name='gen5tt')
gen5tt.inputs.algorithm = 'hsvs'
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
mrxform_t1 = pe.Node(mrtrix3.MRTransform(), name='mrxform_t1')
mrxform_t1.inputs.invert = True
mrxform_t1.inputs.out_file = 'T1_coreg.mif'
mrxform_5tt = pe.Node(mrtrix3.MRTransform(), name='mrxform_5tt')
mrxform_5tt.inputs.invert = True
mrxform_5tt.inputs.out_file = '5tt_coreg.mif'

# Convert parc file to MIF, replace labels, and coregister to DWI
parc_mif = pe.Node(mrtrix3.MRConvert(), name='parc_mif')
parc_mif.inputs.args = '-datatype uint32'
parc_mif.inputs.out_file = 'aparcaseg.mif'

labelconv = pe.Node(mrtrix3.LabelConvert(), name='labelconv')
labelconv.inputs.out_file = 'aparcaseg_label.mif'

mrxform_parc = pe.Node(mrtrix3.MRTransform(), name='mrxform_parc')
mrxform_parc.inputs.args = '-datatype uint32'
mrxform_parc.inputs.invert = True
mrxform_parc.inputs.out_file = 'aparcaseg_label_coreg.mif'

# Anatomically constrained tractography (ACT)
tk = pe.Node(mrtrix3.Tractography(), name='tk')
tk.inputs.args = '-nthreads 50'
tk.inputs.backtrack = True
tk.inputs.select = n_tracks
tk.inputs.out_file = f'tracks_{n_tracks}.tck'

# # Spherical-deconvolution informed filtering of tractograms (SIFT)
# sift = pe.Node(SIFT(), name='sift')
# sift.inputs.args = f'–term_number {int(n_tracks/10)} -nthreads 50'
# sift.inputs.out_file = f'tracks_sift_{int(n_tracks/10)}.tck'

# # Tractograms to structural connectomes
# tck2conn = pe.Node(mrtrix3.BuildConnectome(), name='tck2conn')
# tck2conn.inputs.args = '–symmetric –zero_diagonal -scale_invnodevol'
# tck2conn.inputs.out_file = f'sc_sift_{int(n_tracks/10)}.csv'

# SIFT2
sift2 = pe.Node(SIFT2(), name='sift2')
sift2.inputs.args = '-nthreads 50'
sift2.inputs.out_file = 'sift2_tck_weights.txt'

# Tractograms to structural connectomes
tck2conn = pe.Node(mrtrix3.BuildConnectome(), name='tck2conn')
tck2conn.inputs.args = '–symmetric –zero_diagonal -scale_invnodevol'
tck2conn.inputs.out_file = f'sc_sift2_{int(n_tracks)}.csv'


# ---------------------- CREATE WORKFLOW AND CONNECT NODES ----------------------
print(colours.CGREEN + 'Connecting Nodes.\n' + colours.CEND)

workflow = pe.Workflow(name='act_wf', base_dir=f"{wf_dir}")
workflow.connect([
# ---------------------- INPUT/OUTPUT STRUCTURE (Handling input/output directories)
    
    # # Linking subject's information to data source
    # (infosource, datasource, [('subject_id', 'subject_id')]),
    # # Setting the output directory structure based on the subject ID
    # (infosource, datasink, [('subject_id',  'container')]),

    (infosource, splitSubjectScanList, [('subject_scan', 'subject_scan')]),

    # Connect to datasource
    (splitSubjectScanList, datasource, [('subject_id', 'subject_id')]),
    (splitSubjectScanList, datasource, [('scan_id', 'scan_id')]),

    # Connect to datasink
    (infosource, createSubjectScanContainer, [('subject_scan', 'subject_scan')]),
    (createSubjectScanContainer, datasink, [('container', 'container')]),

# ---------------------- FIBER ORIENTATION DISTRIBUTION (FOD for WM, GM, CSF)
    
    # Estimate response functions for different tissue types
    (datasource, response_sd, [('dwi_eddy_file', 'in_file'),
                               ('bvec_file', 'in_bvec'),
                               ('bval_file', 'in_bval')]),
    # Estimate the orientation of all fibers crossing every voxel
    (datasource, dwi2fod, [('dwi_eddy_file', 'in_file'),
                           ('dwi_mask_file', 'mask_file'),
                           ('bvec_file', 'in_bvec'),
                           ('bval_file', 'in_bval')]),
    (response_sd, dwi2fod, [('wm_file', 'wm_txt'),
                            ('gm_file', 'gm_txt'),
                            ('csf_file', 'csf_txt')]),
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
    (datasource, mrxform_t1, [('t1_file', 'in_files')]),
    (transconv, mrxform_t1, [('out_transform', 'linear_transform')]),
    # Coregister 5TT to DWI
    (gen5tt, mrxform_5tt, [('out_file', 'in_files')]),
    (transconv, mrxform_5tt, [('out_transform', 'linear_transform')]),

# ---------------------- PARCELLATION PREP (parcellation image label edits & coregister to DWI)

    (datasource, parc_mif, [('parc_file', 'in_file')]),

    (parc_mif, labelconv, [('out_file', 'in_file')]),
    (infosource, labelconv, [('fs_lut_file', 'in_lut')]),
    (infosource, labelconv, [('fs_default_file', 'in_config')]),

    # Coregister parcellations to DWI
    (labelconv, mrxform_parc, [('out_file', 'in_files')]),
    # (datasource, mrxform_parc, [('parc_file', 'in_files')]),
    (transconv, mrxform_parc, [('out_transform', 'linear_transform')]),

# # ---------------------- ANATOMICALLY CONSTRAINED TRACTOGRAPHY (ACT) - SIFT

#     # Perform ACT using 5TT image and determine seed points dynamically using WM FOD
#     (mrxform_5tt, tk, [('out_file', 'act_file')]),
#     (mtn, tk, [('out_file_wm', 'seed_dynamic')]),
#     (mtn, tk, [('out_file_wm', 'in_file')]),
#     # Spherical-deconvolution informed filtering of tractograms (SIFT)
#     (mrxform_5tt, sift, [('out_file', 'act_file')]),
#     (tk, sift, [('out_file', 'in_tracks')]),
#     (mtn, sift, [('out_file_wm', 'in_fod')]),
#     # Generate structural connectomes of SIFT tractograms
#     (sift, tck2conn, [('out_tracks', 'in_file')]),
#     (mrxform_parc, tck2conn, [('out_file', 'in_parc')]),

# # ---------------------- DATASINK (save tractograms and structural connectomes)

#     # Save the tractograms and structural connectomes
#     (sift, datasink, [('out_tracks', '@tracks')]), 
#     (tck2conn, datasink, [('out_file', '@sc')]), 

# ---------------------- ANATOMICALLY CONSTRAINED TRACTOGRAPHY (ACT) - SIFT2

    # Perform ACT using 5TT image and determine seed points dynamically using WM FOD
    (mrxform_5tt, tk, [('out_file', 'act_file')]),
    (mtn, tk, [('out_file_wm', 'seed_dynamic')]),
    (mtn, tk, [('out_file_wm', 'in_file')]),
    # SIFT2
    (mrxform_5tt, sift2, [('out_file', 'act_file')]),
    (tk, sift2, [('out_file', 'in_tracks')]),
    (mtn, sift2, [('out_file_wm', 'in_fod')]),
    # Generate structural connectomes of SIFT2 tractograms
    (tk, tck2conn, [('out_file', 'in_file')]),
    (sift2, tck2conn, [('out_weights', 'in_weights')]),
    (mrxform_parc, tck2conn, [('out_file', 'in_parc')]),

# ---------------------- DATASINK (save tractograms and structural connectomes)

    # Save the tractograms and structural connectomes
    (tk, datasink, [('out_file', '@tracks')]), 
    (tck2conn, datasink, [('out_file', '@sc')]), 
])

if __name__ == '__main__':
    workflow.write_graph(graph2use='orig')
    workflow.run(plugin=processing_type, plugin_args=plugin_args)
