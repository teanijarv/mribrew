import os
import re
import numpy as np
import nibabel as nib
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec, 
                                    traits, TraitedSpec, File, isdefined, 
                                    CommandLine, CommandLineInputSpec)
from nipype.utils.filemanip import split_filename



# Input specification for the checkDimension interface
class checkDimensionInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='Path to the DWI file', mandatory=True)
    
# Output specification for the checkDimension interface
class checkDimensionOutputSpec(TraitedSpec):
    axialCutX = traits.Int(desc='Flag indicating parity of X dimension')
    axialCutY = traits.Int(desc='Flag indicating parity of Y dimension')
    axialCutZ = traits.Int(desc='Flag indicating parity of Z dimension')

class checkDimension(BaseInterface):
    """
    Interface to check if the dimensions of a given DWI file are even or odd.

    This interface evaluates the spatial dimensions of a provided DWI file to determine their parity. 
    If a dimension is odd, a corresponding flag is set to 1; otherwise, it's set to 0.
    
    Attributes:
    -----------
    in_file: str
        Path to the input DWI file.
    axialCutX: int
        Flag indicating if X dimension is odd (1 if odd, 0 if even).
    axialCutY: int
        Flag indicating if Y dimension is odd (1 if odd, 0 if even).
    axialCutZ: int
        Flag indicating if Z dimension is odd (1 if odd, 0 if even).
    """
    input_spec = checkDimensionInputSpec
    output_spec = checkDimensionOutputSpec
        
    def _run_interface(self, runtime):
        """Check the parity of dimensions of the input DWI file."""
        # Initialize flags to indicate even dimensions
        self.axialCutX, self.axialCutY, self.axialCutZ = 0, 0, 0
        
        # Load the file and retrieve its spatial dimensions
        dimX, dimY, dimZ = nib.load(self.inputs.in_file).shape[:-1]
        
        # Set flags to 1 if the respective dimension is odd
        if dimX % 2: self.axialCutX = 1
        if dimY % 2: self.axialCutY = 1
        if dimZ % 2: self.axialCutZ = 1
        
        return runtime

    def _list_outputs(self):
        """Return flags that indicate the parity of each spatial dimension."""
        return {'axialCutX': self.axialCutX, 'axialCutY': self.axialCutY, 'axialCutZ': self.axialCutZ}



# Input specification for the adjustBval interface
class adjustBvalInputSpec(BaseInterfaceInputSpec):
    in_bval = traits.File(exists=True, desc='Input bval file', mandatory=True)
    valold = traits.Int(desc='B-value to be replaced') 
    valnew = traits.Int(desc='New b-value to replace the old one')

# Output specification for the adjustBval interface
class adjustBvalOutputSpec(TraitedSpec):
    out_bval = traits.File(exists=True, desc='Output bval file with adjusted b-values')

class adjustBval(BaseInterface):
    """
    Interface for adjusting b-values in a bval file (necessary for eddy).

    This interface replaces specified b-values (valold) with new specified b-values (valnew) 
    in the provided bval file. The adjusted b-values are then written to a new bval file.

    Attributes:
    -----------
    in_bval: str
        Path to the input bval file.
    valold: int
        B-value in the input file to be replaced.
    valnew: int
        New b-value to replace the old one.
    out_bval: str
        Path to the output bval file with adjusted b-values.
    """
    input_spec = adjustBvalInputSpec
    output_spec = adjustBvalOutputSpec

    def _run_interface(self, runtime):
        """Replace old b-values with new ones and write to a new bval file."""
        # Setting output to the same directory as the input for convenience
        output_dir = os.path.dirname(self.inputs.in_bval)  
        self.out_bval = os.path.join(output_dir, 'out_bval.bval')

        # Reading the input bval file, modifying its content, and writing the result
        with open(self.inputs.in_bval, 'r') as infile, open(self.out_bval, 'w') as outfile:
            bvals = infile.read().split()
            # Replace old b-values with new ones
            bvals = [str(self.inputs.valnew) if val == str(self.inputs.valold) else val for val in bvals]
            outfile.write(" ".join(bvals))

        return runtime

    def _list_outputs(self):
        """Return the path to the adjusted bval file."""
        outputs = self._outputs().get()
        outputs['out_bval'] = self.out_bval
        return outputs



# Input specification for the eddyIndex interface
class eddyIndexInputSpec(BaseInterfaceInputSpec):
    in_bval = traits.File(exists=True, desc='Path to the bval file', mandatory=True)
    
# Output specification for the eddyIndex interface
class eddyIndexOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True, desc='Path to the output eddy index file')
    
class eddyIndex(BaseInterface):
    """
    Interface to create an index file for the EDDY correction step.
    
    This interface reads a bval file, processes its contents, and then generates 
    an index file required for the EDDY correction step in diffusion MRI preprocessing.
    
    Attributes:
    -----------
    in_bval: str
        Path to the input bval file.
    out_file: str
        Path to the output index file for EDDY correction.
    """
    input_spec = eddyIndexInputSpec
    output_spec = eddyIndexOutputSpec

    def _run_interface(self, runtime):
        """Generate the index file for EDDY correction."""
        # Set the output file path
        output_dir = os.path.dirname(self.inputs.in_bval)  # Setting output to the same directory as the input
        self.out_file = os.path.join(output_dir, 'eddy_index.txt')

        # Read and process the bval file contents
        with open(self.inputs.in_bval, 'r') as f:
            content = f.read()
        # Replace coma and tab delimiter with space
        content = re.sub(r'(\t|,)', ' ', content)
        bvals = np.squeeze(np.fromstring(content, sep=' '))

        # Create the index file for EDDY correction
        eddyIndex = np.ones((1, len(bvals)), int)
        np.savetxt(self.out_file, eddyIndex, fmt='%.i')
        
        return runtime

    def _list_outputs(self):
        """Return the path to the generated index file."""
        outputs = self._outputs().get()
        outputs['out_file'] = self.out_file
        return outputs



# Input specification for the MRTRIX3BrainMask interface
class MRTRIX3BrainMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='Path to the input diffusion image', mandatory=True)
    in_bval = File(exists=True, desc='Path to the bval file', mandatory=True)
    in_bvec = File(exists=True, desc='Path to the bvec file', mandatory=True)
    out_name = traits.Str(desc='Name for the output mask file')
    
# Output specification for the MRTRIX3BrainMask interface
class MRTRIX3BrainMaskOutputSpec(TraitedSpec):
    out_mask = File(exists=True, desc='Path to the generated brain mask')

class MRTRIX3BrainMask(BaseInterface):
    """
    Interface to generate a brain mask using MRTRIX3.
    
    This interface wraps around the MRTRIX3 'dwi2mask' utility to produce a 
    brain mask. It uses gradient files (bvec and bval) in conjunction with 
    an input diffusion image to generate the mask.
    
    Attributes:
    -----------
    in_file: str
        Path to the input diffusion image.
    in_bval: str
        Path to the bval file.
    in_bvec: str
        Path to the bvec file.
    out_name: str
        Name for the output mask file.
    out_mask: str
        Path to the generated brain mask.
    """
    input_spec = MRTRIX3BrainMaskInputSpec
    output_spec = MRTRIX3BrainMaskOutputSpec

    def _run_interface(self, runtime):
        """Generate the brain mask using MRTRIX3 dwi2mask."""
        _, base, _ = split_filename(self.inputs.in_file)
        output_dir = os.path.abspath('')

        # Define output mask name
        mask_name = self.inputs.out_name if isdefined(self.inputs.out_name) else 'MRtrix3_brain_mask.nii.gz'
        self.out_mask = os.path.join(output_dir, mask_name)

        # Combine gradient files
        gradFiles = ' '.join((self.inputs.in_bvec, self.inputs.in_bval))

        # Initialise and run BRAINMASK
        _brainmask = BRAINMASK(in_grad=gradFiles, input_file=self.inputs.in_file, output_file=self.out_mask)
        _brainmask.run()

        return runtime

    def _list_outputs(self):
        """Return the path to the generated brain mask."""
        outputs = self._outputs().get()
        outputs['out_mask'] = self.out_mask
        return outputs

    def _gen_filename(self, name):
        """Generate filename for output files."""
        if name == 'out_mask':
            return self._gen_outfilename()
        return None

class BRAINMASKInputSpec(CommandLineInputSpec):
    in_grad = traits.Str(exists=True, desc='Gradient files (bvec and bval) concatenated', mandatory=True, argstr="-fslgrad %s", position=0)
    input_file = traits.Str(exists=True, desc='Path to the input diffusion image', mandatory=True, argstr="%s", position=1)
    output_file = traits.Str(desc='Desired path for the output mask', mandatory=True, argstr="%s", position=2)

class BRAINMASK(CommandLine):
    """
    BRAINMASK is a CommandLine interface wrapper around MRTRIX3 'dwi2mask' command.
    
    Attributes:
    -----------
    in_grad: str
        Concatenated gradient file paths (bvec and bval).
    input_file: str
        Path to the input diffusion image.
    output_file: str
        Desired path for the output mask.
    """
    input_spec = BRAINMASKInputSpec
    _cmd = 'dwi2mask -force'



# Input specification for the combineDWIBrainMask interface
class combineDWIBrainMaskInputSpec(BaseInterfaceInputSpec):
    in_mask1 = traits.File(exists=True, desc='Path to the MRTRIX mask', mandatory=True)
    in_mask2 = traits.File(exists=True, desc='Path to the BET mask', mandatory=True)
    out_name = traits.Str(desc='Name for the combined output mask file', mandatory=True)
    
# Output specification for the combineDWIBrainMask interface
class combineDWIBrainMaskOutputSpec(TraitedSpec):
    out_mask = File(exists=True, desc='Path to the generated combined mask')
    
class combineDWIBrainMask(BaseInterface):
    """
    Interface to combine two brain masks (MRTRIX and BET).
    
    This interface combines two input masks, primarily from MRTRIX and BET. The 
    resulting mask is primarily based on the BET mask and has its values set to 
    1 where the MRTRIX mask has a value greater than 0.
    
    Attributes:
    -----------
    in_mask1: str
        Path to the MRTRIX mask.
    in_mask2: str
        Path to the BET mask.
    out_name: str
        Name for the combined output mask file.
    out_mask: str
        Path to the generated combined mask.
    """
    input_spec = combineDWIBrainMaskInputSpec
    output_spec = combineDWIBrainMaskOutputSpec
        
    def _run_interface(self, runtime):
        """Combine the two brain masks."""
        output_dir = os.path.abspath('')
        
        # Load both masks
        mask1_img = nib.load(self.inputs.in_mask1).get_fdata()
        mask2_img = nib.load(self.inputs.in_mask2).get_fdata()

        # Create the combined mask based on the BET mask
        combined_mask = np.where(mask1_img > 0, 1, mask2_img)
        
        # Save the combined mask
        combined_mask_file = nib.Nifti1Image(combined_mask, mask2_img.affine, mask2_img.header)
        self.out_mask = os.path.join(output_dir, self.inputs.out_name)
        nib.save(combined_mask_file, self.out_mask)

        return runtime

    def _list_outputs(self):
        """Return the path to the generated combined mask."""
        outputs = self._outputs().get()
        outputs['out_mask'] = self.out_mask
        return outputs



# Input specification for the MRTRIX3GradCheck interface
class MRTRIX3GradCheckInputSpec(BaseInterfaceInputSpec): 
    in_file = File(exists=True, desc='DWI volume to be analysed', mandatory=True)
    in_bvals = File(exists=True, desc='corresponding b-values', mandatory=True)
    in_bvecs = File(exists=True, desc='corresponding b-vectors', mandatory=True)

# Output specification for the MRTRIX3GradCheck interface
class MRTRIX3GradCheckOutputSpec(TraitedSpec): 
    out_bvecs = File(desc='gradient direction checked b-vectors')
    out_bvals = File(desc='corresponding b-values')
    
class MRTRIX3GradCheck(BaseInterface): 
    """
    Interface for MRTRIX3 DWI Gradient Check.
    
    This interface is designed to check the gradient of DWI volumes using 
    MRTRIX3. The output of this interface includes checked gradient directions 
    (b-vectors) and their corresponding b-values.
    
    Attributes:
    -----------
    in_file: str
        Path to the DWI volume to be analysed.
    in_bvals: str
        Path to the corresponding b-values.
    in_bvecs: str
        Path to the corresponding b-vectors.
    out_bvecs: str
        Path to the gradient direction checked b-vectors.
    out_bvals: str
        Path to the corresponding b-values.
    """
    input_spec = MRTRIX3GradCheckInputSpec
    output_spec = MRTRIX3GradCheckOutputSpec
        
    def _run_interface(self, runtime): 
        """Check DWI gradient and generate outputs."""
        output_dir = os.path.abspath('')

        # Define names for output files
        self.out_bvecs = os.path.join(output_dir, 'gradChecked.bvecs')    
        self.out_bvals = os.path.join(output_dir, 'gradChecked.bvals')
        
        in_gradFiles = ' '.join((self.inputs.in_bvecs, self.inputs.in_bvals))
        out_gradFiles = ' '.join((self.out_bvecs, self.out_bvals))

        # Initialize and run DWI gradient check
        _gradcheck = DWIGRADCHECK(input_file=self.inputs.in_file, 
                                  in_grad=in_gradFiles, 
                                  out_grad=out_gradFiles)
        _gradcheck.run()

        return runtime

    def _list_outputs(self):
        """Return the paths to the generated b-vectors and b-values."""
        outputs = self._outputs().get()
        outputs['out_bvecs'] = self.out_bvecs
        outputs['out_bvals'] = self.out_bvals

        return outputs
    
    def _gen_filename(self, name):
        if name == 'out_bvecs':
            return self._gen_outfilename()
        return None

# Input specification for DWIGRADCHECK
class DWIGRADCHECKinputSpec(CommandLineInputSpec):
    input_file = traits.Str(exists=True, desc='input file', 
                            mandatory=True, argstr="%s", position=0)
    in_grad = traits.Str(exists=True, desc='input bvecs and bvals', 
                            mandatory=True, argstr="-fslgrad %s", position=1)
    out_grad = traits.Str(desc='output bvecs and bvals', 
                            mandatory=True, argstr="-export_grad_fsl %s", position=2)

class DWIGRADCHECK(CommandLine):
    """
    Command-line Interface for dwigradcheck utility in MRTRIX3.
    
    This class represents the dwigradcheck command from MRTRIX3. It checks the 
    gradient of DWI volumes.
    """
    input_spec = DWIGRADCHECKinputSpec
    _cmd = 'dwigradcheck'
