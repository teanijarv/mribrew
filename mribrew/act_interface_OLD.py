"""MRtrix3 interfaces for commands adapted from NiPype and cmtklib."""

import os.path as op
from nipype.interfaces.base import (
    CommandLineInputSpec,
    CommandLine,
    traits,
    TraitedSpec,
    InputMultiPath,
    File,
    Directory,
    isdefined
)
from nipype.interfaces.mrtrix3.base import (MRTrix3Base, MRTrix3BaseInputSpec)

class ResponseMeanInputSpec(CommandLineInputSpec):
    in_txts = InputMultiPath(File(exists=True), mandatory=True, argstr='%s',
                             sep=' ', desc='The input response functions')
    out_txt = File(argstr='%s', position=-1,
                   desc='The output mean response function')
class ResponseMeanOutputSpec(TraitedSpec):
    out_txt = File(exists=True, desc='The output mean response function')
class ResponseMean(MRTrix3Base):
    """Calculate the mean response function from a set of text files using `responsemean`."""

    _cmd = 'responsemean'
    input_spec = ResponseMeanInputSpec
    output_spec = ResponseMeanOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()

        if not isdefined(self.inputs.out_txt):
            outputs['out_txt'] = op.abspath('out.txt')
        else:
            outputs['out_txt'] = op.abspath(self.inputs.out_txt)

        return outputs

class Generate5ttInputSpec(MRTrix3BaseInputSpec):
    algorithm = traits.Enum(
        "fsl",
        "gif",
        "freesurfer",
        "hsvs",
        argstr="%s",
        position=-3,
        mandatory=True,
        desc="tissue segmentation algorithm",
    )
    in_file = traits.Either(
        File(exists=True),
        Directory(exists=True),
        argstr="%s", mandatory=True, position=-2, desc="input image / directory"
    )
    out_file = File(argstr="%s", mandatory=True, position=-1, desc="output image")
    nthreads = traits.Int(
        argstr="-nthreads %d",
        desc="number of threads. if zero, the number of available cpus will be used",
        nohash=True,
    )
class Generate5ttOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output image")
class Generate5tt(MRTrix3Base):
    """
    Generate a 5TT image suitable for ACT using the selected algorithm


    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> gen5tt = mrt.Generate5tt()
    >>> gen5tt.inputs.in_file = 'T1.nii.gz'
    >>> gen5tt.inputs.algorithm = 'fsl'
    >>> gen5tt.inputs.out_file = '5tt.mif'
    >>> gen5tt.cmdline                             # doctest: +ELLIPSIS
    '5ttgen fsl T1.nii.gz 5tt.mif'
    >>> gen5tt.run()                               # doctest: +SKIP
    """

    _cmd = "5ttgen"
    input_spec = Generate5ttInputSpec
    output_spec = Generate5ttOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = op.abspath(self.inputs.out_file)
        return outputs

class SIFTInputSpec(CommandLineInputSpec):
    in_tracks = File(exists=True, mandatory=True, argstr='%s',
                     position=-3, desc='Input track file in TCK format')
    in_fod = File(exists=True, mandatory=True, argstr='%s', position=-2,
                  desc='Input image containing the spherical harmonics of the fibre orientation distributions')
    act_file = File(exists=True, argstr='-act %s',
                    position=-4, desc='ACT 5TT image file')
    out_file = File(argstr='%s', position=-1,
                    desc='Output filtered tractogram')
class SIFTOutputSpec(TraitedSpec):
    out_tracks = File(
        exists=True, desc='Output filtered tractogram')
class SIFT(MRTrix3Base):
    """Spherical-deconvolution informed filtering of tractograms using `tcksift` [Smith2013SIFT]_.

    References
    ----------
    .. [Smith2013SIFT] R.E. Smith et al., NeuroImage 67 (2013), pp. 298–312, <https://www.ncbi.nlm.nih.gov/pubmed/23238430>.


    Example
    -------
    >>> import cmtklib.interfaces.mrtrix3 as cmp_mrt
    >>> mrtrix_sift = cmp_mrt.SIFT()
    >>> mrtrix_sift.inputs.in_tracks = 'tractogram.tck'
    >>> mrtrix_sift.inputs.in_fod = 'spherical_harmonics_image.nii.gz'
    >>> mrtrix_sift.inputs.out_file = 'sift_tractogram.tck'
    >>> mrtrix_sift.run()   # doctest: +SKIP

    """

    _cmd = 'tcksift'
    input_spec = SIFTInputSpec
    output_spec = SIFTOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()

        if not isdefined(self.inputs.out_file):
            outputs['out_tracks'] = op.abspath('SIFT-filtered_tractogram.tck')
        else:
            outputs['out_tracks'] = op.abspath(self.inputs.out_file)

        return outputs


class SIFT2InputSpec(CommandLineInputSpec):
    in_tracks = File(exists=True, mandatory=True, argstr='%s',
                     position=-3, desc='Input track file in TCK format')
    in_fod = File(exists=True, mandatory=True, argstr='%s', position=-2,
                  desc='Input image containing the spherical harmonics of the fibre orientation distributions')
    act_file = File(exists=True, argstr='-act %s',
                    position=-4, desc='ACT 5TT image file')
    out_file = File(argstr='%s', position=-1,
                    desc='Output text file containing the weighting factor for each streamline')
class SIFT2OutputSpec(TraitedSpec):
    out_weights = File(
            exists=True, desc='Output text file containing the weighting factor for each streamline')
class SIFT2(MRTrix3Base):
    """Determine an appropriate cross-sectional area multiplier for each streamline using `tcksift2` [Smith2015SIFT2]_.

    References
    ----------
    .. [Smith2015SIFT2] Smith RE et al., Neuroimage, 2015, 119:338-51. <https://doi.org/10.1016/j.neuroimage.2015.06.092>.


    Example
    -------
    >>> import cmtklib.interfaces.mrtrix3 as cmp_mrt
    >>> mrtrix_sift2 = cmp_mrt.SIFT2()
    >>> mrtrix_sift2.inputs.in_tracks = 'tractogram.tck'
    >>> mrtrix_sift2.inputs.in_fod = 'spherical_harmonics_image.nii.gz'
    >>> mrtrix_sift2.inputs.out_file = 'sift2_fiber_weights.txt'
    >>> mrtrix_sift2.run()  # doctest: +SKIP

    """

    _cmd = 'tcksift2'
    input_spec = SIFT2InputSpec
    output_spec = SIFT2OutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()

        if not isdefined(self.inputs.out_file):
            outputs['out_weights'] = op.abspath('streamlines_weights.txt')
        else:
            outputs['out_weights'] = op.abspath(self.inputs.out_file)

        return outputs

class MTNormaliseInputSpec(MRTrix3BaseInputSpec):
    wm_fod = File(
        argstr="%s",
        exists=True,
        mandatory=False,
        position=1,
        desc="input fod of white matter tissue compartment"
    )
    out_file_wm = File(
        argstr="%s",
        mandatory=False,
        position=2,
        desc="output file of white matter tissue compartment"
    )
    gm_fod = File(
        argstr="%s",
        exists=True,
        mandatory=False,
        position=3,
        desc="input fod of grey matter tissue compartment"
    )
    out_file_gm = File(
        argstr="%s",
        mandatory=False,
        position=4,
        desc="output file of grey matter tissue compartment"
    )
    csf_fod = File(
        argstr="%s",
        exists=True,
        mandatory=False,
        position=5,
        desc="input fod of CSF tissue compartment"
    )
    out_file_csf = File(
        argstr="%s",
        mandatory=False,
        position=6,
        desc="output file of CSF tissue compartment 3"
    )
    mask = File(
        argstr="-mask %s",
        exists=True,
        position=-1,
        desc="input brain mask"
    )
class MTNormaliseOutputSpec(TraitedSpec):
    out_file_wm = File(exists=True, desc="the normalized white matter fod")
    out_file_gm = File(exists=True, desc="the normalized grey matter fod")
    out_file_csf = File(exists=True, desc="the normalized csf fod")
class MTNormalise(CommandLine):
    """
    Multi-tissue informed log-domain intensity normalisation
    Example
    -------
    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> mtn = mrt.MTnormalise()
    >>> mtn.inputs.fod_wm = 'wmfod.mif'
    >>> mtn.inputs.fod_gm = 'gmfod.mif'
    >>> mtn.inputs.fod_csf = 'csffod.mif'
    >>> mtn.inputs.out_file_wm = 'wmfod_norm.mif'
    >>> mtn.inputs.out_file_gm = 'gmfod_norm.mif'
    >>> mtn.inputs.out_file_csf = 'csffod_norm.mif'
    >>> mtn.inputs.mask = 'mask.mif'
    >>> mtn.cmdline                      
    'mtnormalise wmfod.mif wmfod_norm.mif gmfod.mif gmfod_norm.mif csffod.mif csffod_norm.mif -mask mask.mif'
    >>> mtn.run()                                 
    """

    _cmd = "mtnormalise"
    input_spec = MTNormaliseInputSpec
    output_spec = MTNormaliseOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file_wm"] = op.abspath(self.inputs.out_file_wm)
        outputs["out_file_gm"] = op.abspath(self.inputs.out_file_gm)
        outputs["out_file_csf"] = op.abspath(self.inputs.out_file_csf)
        return outputs