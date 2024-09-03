import os, csv, re
import nibabel as nib
import numpy as np
import pandas as pd

from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec,
    traits, TraitedSpec, File, isdefined,
    CommandLine, CommandLineInputSpec, DynamicTraitedSpec, Undefined
)
from nipype.utils.filemanip import split_filename
from nibabel.tmpdirs import InTemporaryDirectory

from filelock import SoftFileLock

### TRACTSEG WM PARCELLATION

class RawTractSegInputSpec(BaseInterfaceInputSpec):
    in_bvals = File(exists=True, desc='corresponding b-values ', mandatory=True)
    in_bvecs = File(exists=True, desc='corresponding b-vectors', mandatory=True)
    in_mask = File(desc='brain mask to calculate tracts in')
    in_file = File(exists=True, desc='DWI image', mandatory=True)
    output_dir = traits.Str(exists=True, desc='output directory')
    tract_definition = traits.Str(desc='tract definition')
    args = traits.Str(exists=True, desc='extra arguments')

class RawTractSegOutputSpec(TraitedSpec):
    out_binary_atlas = File(desc='tract segmentation atlas 4D')
    out_labels = traits.Dict(desc='dictionary of tract labels')

class RawTractSeg(BaseInterface):
    input_spec = RawTractSegInputSpec
    output_spec = RawTractSegOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.in_mask):
            maskInput = "--brain_mask " + self.inputs.in_mask
        else:
            maskInput = ""

        # define output folders
        if not self.inputs.output_dir:
            self.inputs.output_dir = os.path.abspath('') 
        
        # define names for output files
        self.binary_atlas = os.path.join(self.inputs.output_dir, 'bundle_segmentations.nii.gz')

        if not self.inputs.tract_definition:
            self.inputs.tract_definition = 'TractQuerier+'
     
        # initialise and run tractseg
        if self.inputs.args:
            _tractseg = TRACTSEG(input_bvals= self.inputs.in_bvals,
                    input_bvecs=self.inputs.in_bvecs, 
                    input_file=self.inputs.in_file, 
                    input_mask= maskInput,
                    output_dir=self.inputs.output_dir, 
                    tract_definition=self.inputs.tract_definition,
                    extra_arg=self.inputs.args)
        
        _tractseg.run()

        # get the tract labels dictionary based on the tract definition
        self.tract_labels = self.fetch_tract_labels(self.inputs.tract_definition)
        
        return runtime
    
    def fetch_tract_labels(self, tract_definition):
        
        current_dir = os.path.dirname(os.path.abspath(__file__))

        if tract_definition.lower() == 'xtract':
            file_path = os.path.join(current_dir, '..', 'misc', 'atlases', 
                                     'xtract', 'xtract_labels.txt')
        elif tract_definition.lower() == 'tractquerier+':
            file_path = os.path.join(current_dir, '..', 'misc', 'atlases', 
                                     'tractquerier', 'tractquerier_labels.txt')
        
        tract_labels = {}
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip().rstrip(',')
                key, value = line.split(': ')
                value = value.strip("'")
                tract_labels[int(key)] = value

        return tract_labels
        
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_binary_atlas'] = self.binary_atlas
        outputs['out_labels'] = self.tract_labels
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
    tract_definition = traits.Str(desc='tract definition', mandatory=True, argstr="--tract_definition %s", position=5)
    extra_arg = traits.Str(desc='extra arguments', argstr="%s", position=6)
    
class TRACTSEG(CommandLine):
    input_spec = TRACTSEGinputSpec
    _cmd = 'TractSeg'


### COMPUTE METRICS IN WM TRACTS 

class TractMetricsInputSpec(BaseInterfaceInputSpec): 
    in_binary_atlas = File(exists=True, desc='binary atlas of tracts', mandatory=True)
    tract_labels = traits.Dict(desc='dictionary of tract labels', mandatory=True)
    in_md = File(desc='MD map to compute metrics on')
    in_fa = File(desc='FA map to compute metrics on')
    in_ad = File(desc='AD map to compute metrics on')
    in_rd = File(desc='RD map to compute metrics on')
    subject_scan = traits.List(traits.Str, desc='list of subject ID and scan date', mandatory=True)
    thresh1 = traits.Float(desc='MD QC threshold 1', mandatory=True)
    thresh2 = traits.Float(desc='MD QC threshold 2', mandatory=True)

class TractMetricsOutputSpec(TraitedSpec): 
    out_csv_report = File(desc='csv file with metrics of the tracts')
    out_csv_summary = File(desc='csv file with metrics of the tracts in single row')

class TractMetrics(BaseInterface):
    input_spec = TractMetricsInputSpec
    output_spec = TractMetricsOutputSpec

    def _run_interface(self, runtime): 
        # define names for the output CSV files
        self.out_csv_name = os.path.join(os.path.abspath(''), 'TractSeg_report.csv')
        self.out_csv_summaryname = os.path.join(os.path.abspath(''), 'TractSeg_summary.csv')
        
        # load the binary atlas and determine the number of ROIs (i.e., tracts from the atlas)
        binary_atlas, numberOfROIs = self.load_atlas(self.inputs.in_binary_atlas)
        
        # specify the metrics to be computed and load the data for each specified metric
        metrics = ['md', 'fa', 'ad', 'rd']
        data = {metric: self.load_data(getattr(self.inputs, f'in_{metric}')) for metric in metrics}
        
        # calculate the single voxel volume in milliliters
        voxelVolume_ml = np.prod(nib.load(self.inputs.in_binary_atlas).header.get_zooms()[:3]) / 1000
        
        # looping through each ROI
        ROI_volume_ml = np.zeros(numberOfROIs, dtype=np.float32)
        perc_MD_WML = np.zeros(numberOfROIs, dtype=np.float32)
        metric_results = {metric: self.init_metric_results(numberOfROIs, data[metric]) for metric in metrics}
        for r in range(numberOfROIs):
            roi = binary_atlas[:, :, :, r]

            # calculate the ROI volume
            ROI_volume_ml[r] = np.sum(roi) * voxelVolume_ml
            
            # calculating each metric
            for metric in metrics:
                if data[metric] is not None:
                    # extracting metric values within ROI
                    tmp = data[metric][roi == 1]
                    # calculate mean and std for that metric
                    metric_results[metric]['mean'][r] = np.mean(tmp)
                    metric_results[metric]['std'][r] = np.std(tmp)
                    # QC: calculate the percentage of MD values within the specified thresholds
                    if metric == 'md':
                        perc_MD_WML[r] = np.count_nonzero((tmp > self.inputs.thresh1) & (tmp < self.inputs.thresh2)) / tmp.size * 100.0 if tmp.size > 0 else 0.0

        # save the detailed and summary metrics to a CSV files
        self.save_metrics_csv(self.out_csv_name, numberOfROIs, ROI_volume_ml, perc_MD_WML, 
                              metric_results, self.inputs.tract_labels)
        self.save_summary_csv(self.out_csv_summaryname, self.inputs.subject_scan, numberOfROIs, 
                              ROI_volume_ml, perc_MD_WML, metric_results, metrics, self.inputs.tract_labels)
        
        return runtime

    def load_atlas(self, atlas_path):
        # load the binary atlas file and return the data and number of ROIs
        atlas_proxy = nib.load(atlas_path)
        atlas_data = np.squeeze(atlas_proxy.get_fdata())
        return atlas_data, atlas_data.shape[3]

    def load_data(self, file_path):
        # load the data from the specified file path if it is defined
        return nib.load(file_path).get_fdata() if isdefined(file_path) else None

    def init_metric_results(self, numberOfROIs, data):
        # initialize dictionaries to store the mean and std of metrics
        if data is not None:
            return {'mean': np.zeros(numberOfROIs, dtype=np.float32), 'std': np.zeros(numberOfROIs, dtype=np.float32)}
        else:
            return {'mean': ['n/a'] * numberOfROIs, 'std': ['n/a'] * numberOfROIs}

    def save_metrics_csv(self, filename, numberOfROIs, ROI_volume_ml, perc_MD_WML, metric_results, tract_labels):
        # save the detailed metrics to a CSV file
        with open(filename, 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['metric'] + [tract_labels[i] for i in range(numberOfROIs)])
            filewriter.writerow(['ROI_vol_ml'] + list(ROI_volume_ml))
            filewriter.writerow(['perc_MD_WML'] + list(perc_MD_WML))
            for metric in metric_results:
                filewriter.writerow([f'ROI_mean_{metric.upper()}'] + list(metric_results[metric]['mean']))
                filewriter.writerow([f'ROI_std_{metric.upper()}'] + list(metric_results[metric]['std']))

    def save_summary_csv(self, filename, subject_scan, numberOfROIs, ROI_volume_ml, perc_MD_WML, metric_results, metrics, tract_labels):
        # extract subject ID and scan date from the subject_scan list
        subject_id, scan_id = subject_scan

        # save the summary metrics to a CSV file
        with open(filename, 'w') as f:
            filewriter = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            # create headers dynamically based on the specified metrics
            header_list = ['subject', 'date']
            for metric in metrics:
                header_list.extend([f'mean_{metric.upper()}_{tract_labels[i]}' for i in range(numberOfROIs)])
                header_list.extend([f'std_{metric.upper()}_{tract_labels[i]}' for i in range(numberOfROIs)])
            header_list.extend([f'tract_vol_{tract_labels[i]}' for i in range(numberOfROIs)])
            header_list.extend([f'perc_MD_WML_{tract_labels[i]}' for i in range(numberOfROIs)])
            filewriter.writerow(header_list)
            # create the row for the summary CSV
            row = [subject_id, scan_id]
            for metric in metrics:
                row.extend(metric_results[metric]['mean'])
                row.extend(metric_results[metric]['std'])
            row.extend(ROI_volume_ml)
            row.extend(perc_MD_WML)
            filewriter.writerow(row)

    def _list_outputs(self):
        # list the outputs of the interface
        outputs = self._outputs().get()
        outputs['out_csv_report'] = self.out_csv_name
        outputs['out_csv_summary'] = self.out_csv_summaryname
        return outputs

    def _gen_filename(self, name):
        # generate filenames for the outputs
        if name == 'out_csv_report':
            return self._gen_outfilename()
        return None

#-----------------------------------------------------------------------------------------------------#
# GET BRAIN MASK VOLUME  
#-----------------------------------------------------------------------------------------------------#
class QCgetBrainMaskVolInputSpec(BaseInterfaceInputSpec):
    in_mask = traits.File(exists=True, desc='mask volume', mandatory=True)

class QCgetBrainMaskVolOutputSpec(TraitedSpec):
    out_maskvolume = traits.Float(exists=True, desc='QC output: volume of brain mask')

class QCgetBrainMaskVol(BaseInterface):
    input_spec = QCgetBrainMaskVolInputSpec
    output_spec = QCgetBrainMaskVolOutputSpec

    def _run_interface(self, runtime): 
        imgFile = nib.load(self.inputs.in_mask)
        img = imgFile.get_fdata()
        voxlVolume = np.prod(imgFile.header.get_zooms())
        self.out_maskvolume = np.sum(img) * voxlVolume / 1000  #volume in ml

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_maskvolume'] = self.out_maskvolume

        return outputs


#-----------------------------------------------------------------------------------------------------#
# GET WM VOLUME  
#-----------------------------------------------------------------------------------------------------#
class QC_wm_volInputSpec(BaseInterfaceInputSpec):
    in_file = traits.File(exists=True, desc='binary masks', mandatory=True)

class QC_wm_volOutputSpec(TraitedSpec):
    out_wmvolume = traits.Float(exists=True, desc='QC output: volume of white matter')

class QC_wm_vol(BaseInterface):
    input_spec = QC_wm_volInputSpec
    output_spec = QC_wm_volOutputSpec

    def _run_interface(self, runtime): 
        imgFile = nib.load(self.inputs.in_file)
        binary_atlas = np.squeeze(imgFile.get_fdata())
       
        tractsum = np.sum(binary_atlas,axis = 3)

        voxlVolume = np.prod(imgFile.header.get_zooms())
        self.out_wmvolume = np.count_nonzero(tractsum) * voxlVolume / 1000  #volume in ml

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_wmvolume'] = self.out_wmvolume

        return outputs

#-----------------------------------------------------------------------------------------------------#
# QC MD - GET PERCENTAGE OF VOXELS IN MASK WITH MD LARGER THAN THRES
#-----------------------------------------------------------------------------------------------------#

class QC_MDInputSpec(BaseInterfaceInputSpec): 
    in_md = File(exists=True, desc='MD map csv file containing tract metric', mandatory=True)
    in_thres = traits.Float(exists=True, desc='MD threshold')


class QC_MDOutputSpec(TraitedSpec): 
    out_perc = traits.Float(desc='percentage of MD larger than 2.5')

class QC_MD(BaseInterface):
    input_spec = QC_MDInputSpec
    output_spec = QC_MDOutputSpec

    def _run_interface(self, runtime): 
        md_load = nib.load(self.inputs.in_md)
        md_img = md_load.get_fdata()  
    
        if self.inputs.in_thres:
            in_thres = self.inputs.in_thres
        else:
            in_thres = 0.0025

        mask_vox = 0
        mdtrhes_vox = 0

        # loop over MD map and check values
        for k in range(md_img.shape[0]):
            for j in range(md_img.shape[1]):
                for i in range(md_img.shape[2]):
                    if md_img[k,j,i] > 0.0:
                        mask_vox = mask_vox + 1
                        if md_img[k,j,i] > in_thres:
                            mdtrhes_vox = mdtrhes_vox + 1
        
        
        self.out_perc = mdtrhes_vox/mask_vox * 100
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_perc'] = self.out_perc

        return outputs

#-----------------------------------------------------------------------------------------------------#
# CREATE SESSION SUMMARY CSV
#-----------------------------------------------------------------------------------------------------#

import os
import csv
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits

# Define the input specifications for the sessionSummaryCSV interface
class sessionSummaryCSVInputSpec(BaseInterfaceInputSpec): 
    in_csv = File(exists=True, desc='initial csv file containing tract metric', mandatory=True)
    in_wmvolume = traits.Float(desc='wm volume', mandatory=True)
    in_maskvolume = traits.Float(desc='mask volume', mandatory=True)
    in_mdperc = traits.Float(desc='percentage of MD in mask larger than 2.5', mandatory=True)
    out_filename = traits.Str(desc='Filename', mandatory=True)

# Define the output specifications for the sessionSummaryCSV interface
class sessionSummaryCSVOutputSpec(TraitedSpec): 
    out_csv_summary = File(desc='csv file with metrics of the tracts in single row')

# Define the sessionSummaryCSV interface
class sessionSummaryCSV(BaseInterface):
    input_spec = sessionSummaryCSVInputSpec
    output_spec = sessionSummaryCSVOutputSpec

    def _run_interface(self, runtime): 
        # Define name for output file 
        self.out_csv_name = os.path.join(os.path.abspath(''), self.inputs.out_filename)
        
        # Save metrics in CSV file
        with open(self.out_csv_name, 'w', newline='') as csvout:
            filewriter = csv.writer(csvout, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            
            with open(self.inputs.in_csv, 'r', newline='') as csvin:
                filereader = csv.reader(csvin)
                
                # Read the header and append new column names
                csv_headings = next(filereader)
                additional_headings = [
                    'wm_vol', 'mask_vol', 'perc_MD_CSF'
                ]
                csv_headings.extend(additional_headings)
                filewriter.writerow(csv_headings)

                # Read the values and append new metrics
                csv_vals = next(filereader)
                additional_values = [
                    round(self.inputs.in_wmvolume, 2),
                    round(self.inputs.in_maskvolume, 2),
                    round(self.inputs.in_mdperc, 2)
                ]
                csv_vals.extend(additional_values)
                filewriter.writerow(csv_vals)
            
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_csv_summary'] = self.out_csv_name
        return outputs

    def _gen_filename(self, name):
        if name == 'out_csv_summary':
            return self.out_csv_name
        return None


class cohortSummaryCSVInputSpec(BaseInterfaceInputSpec):
    in_csv_p = File(exists=True, desc='initial csv file containing tract metric', mandatory=True)
    in_csv_c = File(desc='cohort summary csv', mandatory=True)

class cohortSummaryCSVOutputSpec(TraitedSpec):
    out_csv_c = File(desc='cohort summary csv')

class cohortSummaryCSV(BaseInterface):
    """
    Interface to add an extra row to a cohort summary CSV file.
    """

    input_spec = cohortSummaryCSVInputSpec
    output_spec = cohortSummaryCSVOutputSpec

    def _run_interface(self, runtime):
        try:
            from filelock import SoftFileLock
            self._have_lock = True
        except ImportError:
            from warnings import warn
            warn("Python module filelock was not found: cohortSummaryCSV will not be thread-safe in multi-processor execution")
            self._have_lock = False

        if self._have_lock:
            self._lock = SoftFileLock(f"{self.inputs.in_csv_c}.lock")
            self._lock.acquire()

        try:
            # Read the initial CSV file and get the second row
            df_p = pd.read_csv(self.inputs.in_csv_p)

            # Check if the cohort summary CSV file exists
            if not os.path.exists(self.inputs.in_csv_c):
                # If the file does not exist, create it with the same headers as the initial CSV file
                df_p.to_csv(self.inputs.in_csv_c, index=False)
            else:
                # If the file exists, append the second row to the cohort summary CSV file
                df_c = pd.read_csv(self.inputs.in_csv_c)
                df_c = pd.concat([df_c, df_p], ignore_index=True)
                df_c.to_csv(self.inputs.in_csv_c, index=False)

        finally:
            if self._have_lock:
                self._lock.release()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_csv_c'] = self.inputs.in_csv_c
        return outputs
