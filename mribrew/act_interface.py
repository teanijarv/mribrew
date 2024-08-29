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

def split_filename(fname):
    """Split a filename into parts: path, base filename and extension."""

    special_extensions = [".nii.gz", ".tar.gz", ".niml.dset"]

    pth = op.dirname(fname)
    fname = op.basename(fname)

    ext = None
    for special_ext in special_extensions:
        ext_len = len(special_ext)
        if (len(fname) > ext_len) and (fname[-ext_len:].lower() == special_ext.lower()):
            ext = fname[-ext_len:]
            fname = fname[:-ext_len]
            break
    if not ext:
        fname, ext = op.splitext(fname)

    return pth, fname, ext

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
    

class MRTransformInputSpec(MRTrix3BaseInputSpec):
    in_files = InputMultiPath(
        File(exists=True),
        argstr="%s",
        mandatory=True,
        position=-2,
        desc="Input images to be transformed",
    )
    out_file = File(
        genfile=True,
        argstr="%s",
        position=-1,
        desc="Output image",
    )
    invert = traits.Bool(
        argstr="-inverse",
        desc="Invert the specified transform before using it",
    )
    linear_transform = File(
        exists=True,
        argstr="-linear %s",
        desc=(
            "Specify a linear transform to apply, in the form of a 3x4 or 4x4 ascii file. "
            "Note the standard reverse convention is used, "
            "where the transform maps points in the template image to the moving image. "
            "Note that the reverse convention is still assumed even if no -template image is supplied."
        ),
    )
    replace_transform = traits.Bool(
        argstr="-replace",
        desc="replace the current transform by that specified, rather than applying it to the current transform",
    )
    transformation_file = File(
        exists=True,
        argstr="-transform %s",
        desc="The transform to apply, in the form of a 4x4 ascii file.",
    )
    template_image = File(
        exists=True,
        argstr="-template %s",
        desc="Reslice the input image to match the specified template image.",
    )
    reference_image = File(
        exists=True,
        argstr="-reference %s",
        desc="in case the transform supplied maps from the input image onto a reference image, use this option to specify the reference. Note that this implicitly sets the -replace option.",
    )
    flip_x = traits.Bool(
        argstr="-flipx",
        desc="assume the transform is supplied assuming a coordinate system with the x-axis reversed relative to the MRtrix convention (i.e. x increases from right to left). This is required to handle transform matrices produced by FSL's FLIRT command. This is only used in conjunction with the -reference option.",
    )
    quiet = traits.Bool(
        argstr="-quiet",
        desc="Do not display information messages or progress status.",
    )
    debug = traits.Bool(argstr="-debug", desc="Display debugging messages.")
class MRTransformOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the output image of the transformation")
class MRTransform(MRTrix3Base):
    """
    Apply spatial transformations or reslice images

    Example
    -------

    >>> MRxform = MRTransform()
    >>> MRxform.inputs.in_files = 'anat_coreg.mif'
    >>> MRxform.run()                                   # doctest: +SKIP
    """

    _cmd = "mrtransform"
    input_spec = MRTransformInputSpec
    output_spec = MRTransformOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = self.inputs.out_file
        if not isdefined(outputs["out_file"]):
            outputs["out_file"] = op.abspath(self._gen_outfilename())
        else:
            outputs["out_file"] = op.abspath(outputs["out_file"])
        return outputs

    def _gen_filename(self, name):
        if name == "out_file":
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_files[0])
        return name + "_MRTransform.mif"


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

# new

class BuildConnectomeInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True, argstr="%s", mandatory=True, position=-3, desc="input tractography"
    )
    in_parc = File(exists=True, argstr="%s", position=-2, desc="parcellation file")
    out_file = File(
        "connectome.csv",
        argstr="%s",
        mandatory=True,
        position=-1,
        usedefault=True,
        desc="output file after processing",
    )

    out_assignments = File( # new addition
        "assignments.txt",
        argstr="-out_assignments %s",
        mandatory=False,
        usedefault=True,
        desc="output the node assignments of each streamline to a file",
    )

    nthreads = traits.Int(
        argstr="-nthreads %d",
        desc="number of threads. if zero, the number of available cpus will be used",
        nohash=True,
    )

    vox_lookup = traits.Bool(
        argstr="-assignment_voxel_lookup",
        desc="use a simple voxel lookup value at each streamline endpoint",
    )
    search_radius = traits.Float(
        argstr="-assignment_radial_search %f",
        desc="perform a radial search from each streamline endpoint to locate "
        "the nearest node. Argument is the maximum radius in mm; if no node is"
        " found within this radius, the streamline endpoint is not assigned to"
        " any node.",
    )
    search_reverse = traits.Float(
        argstr="-assignment_reverse_search %f",
        desc="traverse from each streamline endpoint inwards along the "
        "streamline, in search of the last node traversed by the streamline. "
        "Argument is the maximum traversal length in mm (set to 0 to allow "
        "search to continue to the streamline midpoint).",
    )
    search_forward = traits.Float(
        argstr="-assignment_forward_search %f",
        desc="project the streamline forwards from the endpoint in search of a"
        "parcellation node voxel. Argument is the maximum traversal length in "
        "mm.",
    )

    metric = traits.Enum(
        "count",
        "meanlength",
        "invlength",
        "invnodevolume",
        "mean_scalar",
        "invlength_invnodevolume",
        argstr="-metric %s",
        desc="specify the edge weight metric",
    )

    in_scalar = File(
        exists=True,
        argstr="-image %s",
        desc="provide the associated image for the mean_scalar metric",
    )

    scale_file = File( # new
        exists=True,
        argstr="-scale_file %s",
        desc="scale each contribution to the connectome edge according to the values in a vector file",
    )

    in_weights = File(
        exists=True,
        argstr="-tck_weights_in %s",
        desc="specify a text scalar file containing the streamline weights",
    )

    keep_unassigned = traits.Bool(
        argstr="-keep_unassigned",
        desc="By default, the program discards the"
        " information regarding those streamlines that are not successfully "
        "assigned to a node pair. Set this option to keep these values (will "
        "be the first row/column in the output matrix)",
    )
    zero_diagonal = traits.Bool(
        argstr="-zero_diagonal",
        desc="set all diagonal entries in the matrix "
        "to zero (these represent streamlines that connect to the same node at"
        " both ends)",
    )


class BuildConnectomeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the output response file")
    out_assignments = File(exists=True, desc="the output assignments file")


class BuildConnectome(MRTrix3Base):
    """
    Generate a connectome matrix from a streamlines file and a node
    parcellation image

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> mat = mrt.BuildConnectome()
    >>> mat.inputs.in_file = 'tracks.tck'
    >>> mat.inputs.in_parc = 'aparc+aseg.nii'
    >>> mat.cmdline                               # doctest: +ELLIPSIS
    'tck2connectome tracks.tck aparc+aseg.nii connectome.csv'
    >>> mat.run()                                 # doctest: +SKIP
    """

    _cmd = "tck2connectome"
    input_spec = BuildConnectomeInputSpec
    output_spec = BuildConnectomeOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = op.abspath(self.inputs.out_file)
        outputs["out_assignments"] = op.abspath(self.inputs.out_assignments)
        return outputs

class TckSampleInputSpec(CommandLineInputSpec):
    in_tracks = File(exists=True, mandatory=True, argstr='%s',
                     position=-3, desc='Input track file in TCK format')
    in_img = File(exists=True, mandatory=True, argstr='%s', position=-2,
                  desc='Input image to be sampled in MIF format')
    out_samples = File(argstr='%s', position=-1,
                    desc='Output sampled tractogram')
class TckSampleOutputSpec(TraitedSpec):
    out_samples = File(
        exists=True, desc='Output sampled tractogram')
class TckSample(MRTrix3Base):
    """Sample values of an associated image along tracks using `tcksample`.
    """

    _cmd = 'tcksample'
    input_spec = TckSampleInputSpec
    output_spec = TckSampleOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()

        if not isdefined(self.inputs.out_samples):
            outputs['out_samples'] = op.abspath('tck_samples.txt')
        else:
            outputs['out_samples'] = op.abspath(self.inputs.out_samples)

        return outputs