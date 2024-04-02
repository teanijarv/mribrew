# %%
import os
from nipype import config, logging
from nipype.interfaces import io, fsl, mrtrix3
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces.utility import Function

from mribrew.utils import (colours, split_subject_scan_list, create_subject_scan_container)
import mribrew.dwiproc_interface as ProcInterface

# ---------------------- Set up directory structures and constant variables ----------------------
cwd = os.getcwd()
misc_dir = os.path.join(cwd, 'misc')
data_dir = os.path.join(cwd, 'data')
raw_dir = os.path.join(data_dir, 'raw')
proc_dir = os.path.join(data_dir, 'proc', 'dwi_proc')
wf_dir = os.path.join(cwd, 'wf')
log_dir = os.path.join(wf_dir, 'log')

acqp_file = os.path.join(misc_dir, 'acqp.txt')

# DWI sequence file names
dwi_name = 'dir-AP_dwi'
dwipa_name = 'dir-PA_dwi'

# List of all subjects
subject_list = next(os.walk(raw_dir))[1]

# Generate a list of all (subject, scan) tuples
subject_scan_list = []
for sub in subject_list:
    scans = next(os.walk(os.path.join(raw_dir, sub)))[1]
    for scan in scans:
        subject_scan_list.append([sub, scan])

# Filter the subject_scan_list based on PA and AP presence
filtered_subject_scan_list = []
for sub_sc in subject_scan_list:
    scan_dir = os.path.join(raw_dir, sub_sc[0], sub_sc[1], 'dwi')
    files_in_scan = os.listdir(scan_dir) if os.path.exists(scan_dir) else []

    # Check if the scan contains the required files
    has_dwi = any(dwi_name in file for file in files_in_scan)
    has_dwipa = any(dwipa_name in file for file in files_in_scan)

    if has_dwi and has_dwipa:
        filtered_subject_scan_list.append(sub_sc)
subject_scan_list = filtered_subject_scan_list

# Computational variables
processing_type = 'MultiProc' # or 'Linear'
cuda_processing = False
total_memory = 24 # in GB
n_cpus = 24 # number of nipype processes to run at the same time
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

# Set up input files
info = dict(dwi_file=[['subject_id', 'scan_id', 'dwi', '*%s.nii.gz' % dwi_name]],
        bvec_file=[['subject_id', 'scan_id', 'dwi','*%s.bvec' % dwi_name]],
        bval_file=[['subject_id', 'scan_id', 'dwi','*%s.bval' % dwi_name]],
        dwiPA_file=[['subject_id', 'scan_id','dwi', '*%s.nii.gz' % dwipa_name]])

# Set up infosource node to iterate over subject_ids (each subject the amount of their scans)
infosource = pe.Node(niu.IdentityInterface(fields=['subject_scan']), name='infosource')
infosource.iterables = [('subject_scan', subject_scan_list)]
infosource.inputs.acqp_file = acqp_file

splitSubjectScanList = pe.Node(Function(input_names=['subject_scan'],
                                      output_names=['subject_id', 'scan_id'],
                                      function=split_subject_scan_list),
                             name='splitSubjectScanList')

# Set up datasource node
datasource = pe.Node(io.DataGrabber(infields=['subject_id', 'scan_id'], outfields=list(info.keys())),
                                    name='datasource')
datasource.inputs.base_directory = raw_dir
datasource.inputs.template = "%s/%s/%s/%s"
datasource.inputs.template_args = info
datasource.inputs.field_template = dict(
    dwi_file="%s/%s/%s/%s", 
    bvec_file="%s/%s/%s/%s", 
    bval_file="%s/%s/%s/%s",
    dwiPA_file="%s/%s/%s/%s"
)
datasource.inputs.sort_filelist = True

# ---------------------- OUTPUT SINK NODE ----------------------
print(colours.CGREEN + "Creating Sink Node." + colours.CEND)

createSubjectScanContainer = pe.Node(Function(input_names=['subject_scan'],
                                              output_names=['container'],
                                              function=create_subject_scan_container),
                                     name='createSubjectScanContainer')

# Set up sink node where all output is stored in subject folder
datasink = pe.Node(io.DataSink(parameterization=False), name='datasink')
datasink.inputs.base_directory = proc_dir

# ---------------------- PROCESSING NODES ----------------------
print(colours.CGREEN + "Creating Processing Nodes." + colours.CEND)

### DENOISE & DEGIBBS

# Using MRtrix3's DWIDenoise to reduce random noise
mrtrixDenoise = pe.Node(mrtrix3.DWIDenoise(), name='mrtrixDenoise')
mrtrixDenoise.inputs.nthreads = 3

# Removing Gibbs ringing artifacts using MRtrix3's MRDeGibbs function
mrtrixDegibbs = pe.Node(mrtrix3.MRDeGibbs(), name='mrtrixDegibbs')
mrtrixDegibbs.inputs.nthreads = 3

### BRAIN MASK 1 (pre-topup/eddy)

# Extracting the brain mask from the raw data prior to any corrections using FSL's BET function
betMask1 = pe.Node(fsl.BET(), name = 'betMask1')
betMask1.inputs.mask = True
betMask1.inputs.output_type = 'NIFTI_GZ'
betMask1.inputs.no_output = True
betMask1.inputs.functional = True

### TOPUP

# Ensuring the 3D dimensionality of the input images.
checkDimension = pe.Node(ProcInterface.checkDimension(), name='checkDimension')

# Select 1st volume of dwi-ap for topup
dwiB0 = pe.Node(fsl.ExtractROI(), name='dwiB0')
dwiB0.inputs.t_min = 0
dwiB0.inputs.t_size = 1
dwiB0.inputs.output_type = 'NIFTI_GZ'

# Select 1st volume of dwi-pa for topup
dwiPAB0 = pe.Node(fsl.ExtractROI(), name='dwiPAB0')
dwiPAB0.inputs.t_min = 0
dwiPAB0.inputs.t_size = 1
dwiPAB0.inputs.output_type = 'NIFTI_GZ'

# Handle odd dimensions by cutting off a slice from all three planes 
cutOddB0 = pe.Node(fsl.ExtractROI(), name='cutOddB0') 
cutOddB0.inputs.x_size = -1
cutOddB0.inputs.y_size = -1
cutOddB0.inputs.z_size = -1
cutOddB0.inputs.roi_file = 'cutOddB0.nii.gz'
cutOddB0.inputs.output_type = 'NIFTI_GZ'
cutOddPA = cutOddB0.clone(name = 'cutOddPA')
cutOddPA.inputs.roi_file = 'cutOddPA.nii.gz'

# Merge b0s of dwi-ap and dwi-pa
listAPPA = pe.Node(niu.Merge(2), name='listAPPA')
mergeAPPA = pe.Node(fsl.Merge(), name='mergeAPPA')
mergeAPPA.inputs.dimension = 't'
mergeAPPA.inputs.merged_file = 'mergeAPPA.nii.gz'
mergeAPPA.inputs.output_type = 'NIFTI_GZ'

# Topup correction
topup = pe.Node(fsl.TOPUP(), name='topup')
topup.inputs.output_type = "NIFTI_GZ"

### EDDY

# Adjust b-values if both b100 and b0 exist
adjustBval = pe.Node(ProcInterface.adjustBval(), name='adjustBval')
adjustBval.inputs.valold = 100
adjustBval.inputs.valnew = 0

# Create index file for Eddy
eddyIndex = pe.Node(ProcInterface.eddyIndex(), name='eddyIndex')

# Eddy correction
eddy = pe.Node(fsl.Eddy(), name='eddy')
eddy.inputs.interp = 'spline'
eddy.inputs.use_cuda = cuda_processing
eddy.inputs.is_shelled = True
eddy.inputs.args = '--ol_nstd=5 --repol'
eddy.inputs.output_type = 'NIFTI_GZ'

### BRAIN MASK 2 (post-topup/eddy)

# Using FSL's BET function after corrections
betMask2 = betMask1.clone(name='betMask2')

# Creating a brain mask using MRtrix
mrtrixMask = pe.Node(ProcInterface.MRTRIX3BrainMask(), name='mrtrixMask')
mrtrixMask.inputs.out_name = 'mrtrix_mask.nii.gz'

# Combine different masks to create a final DWI brain mask
dwiMask = pe.Node(ProcInterface.combineDWIBrainMask(), name='dwiMask')
dwiMask.inputs.out_name = 'dwi_mask.nii.gz'

### GRADIENT CHECK

# Check gradient directions using MRtrix
mrtrixGradCheck = pe.Node(ProcInterface.MRTRIX3GradCheck(), name='mrtrixGradCheck')

# ---------------------- CREATE WORKFLOW AND CONNECT NODES ----------------------
print(colours.CGREEN + 'Connecting Nodes.\n' + colours.CEND)

workflow = pe.Workflow(name='dwiproc_wf', base_dir=f"{wf_dir}")
workflow.connect([
# ---------------------- INPUT/OUTPUT STRUCTURE (Handling input/output directories)

    (infosource, splitSubjectScanList, [('subject_scan', 'subject_scan')]),

    # Connect to datasource
    (splitSubjectScanList, datasource, [('subject_id', 'subject_id')]),
    (splitSubjectScanList, datasource, [('scan_id', 'scan_id')]),

    # Connect to datasink
    (infosource, createSubjectScanContainer, [('subject_scan', 'subject_scan')]),
    (createSubjectScanContainer, datasink, [('container', 'container')]),

# ---------------------- DENOISE (noise reduction)

    # Apply denoising to the DWI data using mrtrixDenoise
    (datasource, mrtrixDenoise, [('dwi_file', 'in_file')]),

# ---------------------- DEGIBBS (Correction for Gibbs ringing artifacts)

    # Use denoised data to correct for Gibbs ringing artifacts with mrtrixDegibbs
    (mrtrixDenoise, mrtrixDegibbs, [('out_file', 'in_file')]),

# ---------------------- BRAIN MASK 1 (creation of initial brain mask pre-topup/eddy)

    # Extract initial brain mask from the DWI data using BET
    (datasource, betMask1, [('dwi_file', 'in_file')]),

# ---------------------- TOPUP (distrortion correction)
    
    # Extract B0 image (AP direction) for susceptibility correction
    (datasource, dwiB0, [('dwi_file', 'in_file')]),
    # Check the dimensions of the B0 (AP direction)
    (dwiB0, checkDimension, [('roi_file', 'in_file')]),
    # If dimensions are odd, cut slices to ensure compatibility with TOPUP 
    (dwiB0, cutOddB0, [('roi_file', 'in_file')]),
    (checkDimension, cutOddB0, [('axialCutX', 'x_min'),
                                ('axialCutY', 'y_min'),
                                ('axialCutZ', 'z_min')]),
    # Extract B0 image (PA direction)
    (datasource, dwiPAB0, [('dwiPA_file', 'in_file')]),
    # If dimensions are odd for the PA B0, cut slices to ensure compatibility with TOPUP 
    (dwiPAB0, cutOddPA, [('roi_file', 'in_file')]),
    (checkDimension, cutOddPA, [('axialCutX', 'x_min'),
                                ('axialCutY', 'y_min'),
                                ('axialCutZ', 'z_min')]),
    # Merge the B0 images from both AP and PA phase-encode directions
    (cutOddB0, listAPPA, [('roi_file', 'in1')]),
    (cutOddPA, listAPPA, [('roi_file', 'in2')]),
    (listAPPA, mergeAPPA, [('out', 'in_files')]),
    # Execute TOPUP for susceptibility-induced distortion correction
    (mergeAPPA, topup, [('merged_file', 'in_file')]),
    (infosource, topup, [('acqp_file', 'encoding_file')]),

# ---------------------- EDDY (motion & eddy current corrections)

    # Use field coefficients and movement parameters from topup for eddy
    (topup, eddy, [('out_fieldcoef', 'in_topup_fieldcoef'),
                   ('out_movpar', 'in_topup_movpar')]),
    # Provide initial brain mask for eddy
    (betMask1, eddy, [('mask_file', 'in_mask')]), 
    # Provide b-vectors for eddy correction
    (datasource, eddy, [('bvec_file', 'in_bvec')]),
    # Provide denoised, degibbsed DWI data for eddy
    (mrtrixDegibbs, eddy, [('out_file', 'in_file')]),
    # Provide phase-encode information for eddy
    (infosource, eddy, [('acqp_file', 'in_acqp')]),
    # Adjust b-values before using them in eddy
    (datasource, adjustBval, [('bval_file', 'in_bval')]),
    # Create an index file for eddy
    (datasource, eddyIndex, [('bval_file', 'in_bval')]),
    # Provide adjusted b-values to eddy
    (adjustBval, eddy, [('out_bval', 'in_bval')]),
    # Provide index file to eddy
    (eddyIndex, eddy, [('out_file', 'in_index')]), 

# ---------------------- BRAIN MASK 2 (post-topup/eddy)

    # Create a mask with mrtrix based on corrected DWI data
    (eddy, mrtrixMask, [('out_corrected', 'in_file')]), 
    # Use rotated b-vectors for mrtrix mask generation
    (eddy, mrtrixMask, [('out_rotated_bvecs', 'in_bvec')]), 
    # Provide b-values for mask generation with mrtrix
    (datasource, mrtrixMask, [('bval_file', 'in_bval')]), 
    # Create a brain mask with BET based on eddy corrected data
    (eddy, betMask2, [('out_corrected', 'in_file')]),  
    # Combine masks from mrtrix and BET
    (mrtrixMask, dwiMask, [('out_mask', 'in_mask1')]),  
    (betMask2, dwiMask, [('mask_file', 'in_mask2')]), 

# ---------------------- GRADIENT CHECK (ensuring consistency in b-values/vectors)
    
    # Check gradient consistency of the eddy-corrected DWI
    (eddy, mrtrixGradCheck, [('out_corrected', 'in_file')]),
    # Use rotated b-vectors for gradient consistency check
    (eddy, mrtrixGradCheck, [('out_rotated_bvecs', 'in_bvecs')]),
    # Provide original b-values for gradient consistency check
    (datasource, mrtrixGradCheck, [('bval_file', 'in_bvals')]),

# ---------------------- DATASINK (saving results)
    
    # Save the final DWI brain mask
    (dwiMask, datasink, [('out_mask', 'dwi.@dwi_mask')]), 
    # Save the eddy-corrected DWI
    (eddy, datasink, [('out_corrected', 'dwi.@eddy_corrected')]),
    # Save the checked b-values post gradient check
    (mrtrixGradCheck, datasink, [('out_bvals', 'dwi.@bvals')]),
    # Save the checked b-vectors post gradient check
    (mrtrixGradCheck, datasink, [('out_bvecs', 'dwi.@bvecs')]),
])

# Run the script and generate a graph of the workflow
if __name__ == '__main__':
    workflow.write_graph(graph2use='orig')
    workflow.run(plugin=processing_type, plugin_args=plugin_args)
# %%
