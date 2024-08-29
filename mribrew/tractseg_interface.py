import os, csv, re
import nibabel as nib
import numpy as np

from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec,
    traits, TraitedSpec, File, isdefined,
    CommandLine, CommandLineInputSpec, DynamicTraitedSpec, Undefined
)

from nipype.utils.filemanip import split_filename

from nibabel.tmpdirs import InTemporaryDirectory


#-----------------------------------------------------------------------------------------------------#
# TRACTSEG WM PARCELLATION
#-----------------------------------------------------------------------------------------------------#

class RawTractSegInputSpec(BaseInterfaceInputSpec):
    in_bvals = File(exists=True, desc='corresponding b-values ', mandatory=True)
    in_bvecs = File(exists=True, desc='corresponding b-vectors', mandatory=True)
    in_mask = File(desc='brain mask to calculate tracts in')
    in_file = File(exists=True, desc='DWI image', mandatory=True)
    args = traits.Str(exists=True, desc='extra arguments')

class RawTractSegOutputSpec(TraitedSpec):
  
    out_binary_atlas = File(desc='tract segmentation atlas 4D')


class RawTractSeg(BaseInterface):
    input_spec = RawTractSegInputSpec
    output_spec = RawTractSegOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.in_mask):
            maskInput = "--brain_mask " + self.inputs.in_mask
        else:
            maskInput = ""

        # define output folders
        output_dir = os.path.abspath('') 
        
        # define names for output files
        self.binary_atlas = os.path.join(output_dir, 'bundle_segmentations.nii.gz')
        
     
        # initialise and run tractseg
        if self.inputs.args:
            _tractseg = TRACTSEG(input_bvals= self.inputs.in_bvals,
                    input_bvecs=self.inputs.in_bvecs, 
                    input_file=self.inputs.in_file, 
                    input_mask= maskInput,
                    output_dir= output_dir, 
                    extra_arg=self.inputs.args)
        
        _tractseg.run()       
        
        return runtime
        
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_binary_atlas'] = self.binary_atlas
        return outputs

    def _gen_filename(self, name):
        if name == 'out_binary_atlas':
            return self._gen_outfilename()
        return None

class TRACTSEGinputSpec(CommandLineInputSpec):
    input_file = traits.Str(exists=True, desc='input file', mandatory=True, argstr="-i %s", position=0)
    input_bvals = traits.Str(exists=True, desc='input bvals', mandatory=True, argstr="--bvals %s", position=1)
    input_bvecs = traits.Str(exists=True, desc='input bvecs', mandatory=True, argstr="--bvecs %s", position=2)
    input_mask = traits.Str(exists=True, desc='input mask', mandatory=True, argstr="%s", position=3)
    output_dir = traits.Str(desc='output directory', mandatory=True, argstr="-o %s", position=4)
    extra_arg = traits.Str(desc='extra arguments', argstr="%s", position=5)
    
class TRACTSEG(CommandLine):
    input_spec = TRACTSEGinputSpec
    _cmd = 'TractSeg'

