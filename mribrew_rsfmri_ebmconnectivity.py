import os
from nipype import config, logging
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
from nipype.interfaces import Function
from nipype.interfaces.io import DataSink, DataGrabber
from nipype import JoinNode
import numpy as np
import pandas as pd
import abagen

from mribrew.utils import colours

# ---------------------- Set up directory structures and constant variables ----------------------
cwd = os.getcwd()
data_dir = os.path.join(cwd, 'data')
proc_dir = os.path.join(data_dir, 'proc')
wf_dir = os.path.join(cwd, 'wf')
res_dir = os.path.join(data_dir, 'res')
log_dir = os.path.join(wf_dir, 'log')

subject_list = next(os.walk(proc_dir))[1]  # processed subjects

# Computational variables
processing_type = 'MultiProc' # or 'Linear'
total_memory = 6 # in GB
n_cpus = 6 # number of nipype processes to run at the same time
os.environ['OMP_NUM_THREADS'] = str(n_cpus)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cpus)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

plugin_args = {
    'n_procs': n_cpus,
    'memory_gb': total_memory,
    'raise_insufficient': True,
    'scheduler': 'mem_thread',  # Prioritize jobs by memory consumption then nr of threads
}

# RSfMRI variables
atlas_data = abagen.fetch_desikan_killiany()
atlas_info = pd.read_csv(atlas_data['info'])
atlas_info['labelname'] = atlas_info['label'] + "_" + atlas_info['hemisphere']
ebm_stages = {
    'ebm_I': ['entorhinal', 'amygdala', 'hippocampus'],
    'ebm_II': ['bankssts', 'fusiform', 'inferiortemporal', 'middletemporal', 'parahippocampal', 
               'superiortemporal', 'temporalpole'],
    'ebm_III': ['caudalmiddlefrontal', 'inferiorparietal', 'isthmuscingulate', 'lateraloccipital', 
                'posteriorcingulate', 'precuneus', 'superiorparietal', 'supramarginal'],
    'ebm_IV': ['caudalanteriorcingulate', 'frontalpole', 'insula', 'lateralorbitofrontal', 
               'medialorbitofrontal', 'parsopercularis', 'parsorbitalis', 'parstriangularis', 
               'rostralanteriorcingulate', 'rostralmiddlefrontal', 'superiorfrontal'],
}
ebm_labels = [f"{stage}_{hem}" for stage in ebm_stages for hem in ['L', 'R']]

# Set up logging
os.makedirs(log_dir, exist_ok=True)
config.update_config({'logging': {'log_directory': log_dir,'log_to_file': True}})
logging.update_logging(config)

# ---------------------- INPUT SOURCE NODES ----------------------
print(colours.CGREEN + "Creating Source Nodes." + colours.CEND)

# Set up infosource node
infosource = pe.Node(util.IdentityInterface(fields=['subject_id']),
                     name="infosource")
infosource.iterables = [('subject_id', subject_list)]

# Set up datasource node
datasource = pe.Node(interface=DataGrabber(infields=['subject_id'],
                                           outfields=['rsfmri_file']),
                     name='datasource')
datasource.inputs.base_directory = proc_dir
datasource.inputs.template = '%s/RSfMRI/processed_and_censored_32bit.nii.gz'
datasource.inputs.sort_filelist = True

# ---------------------- PROCESSING NODES ----------------------
print(colours.CGREEN + "Creating Processing Nodes." + colours.CEND)

# Compute correlation
def compute_correlation(rsfmri_file, atlas_file, roi_labels, ebm_stages):
    from nilearn import maskers
    import numpy as np
    import os
    masker = maskers.NiftiLabelsMasker(labels_img=atlas_file, standardize='zscore_sample').fit()
    all_timeseries = masker.transform(rsfmri_file)

    aggregated_timeseries = []
    for stage, regions in ebm_stages.items():
        for hem in ['L', 'R']:
            regions_with_hem = [f"{region}_{hem}" for region in regions]
            indices = [roi_labels.index(region) for region in regions_with_hem if region in roi_labels]
            aggregated = np.mean(all_timeseries[:, indices], axis=1)
            aggregated_timeseries.append(aggregated)

    aggregated_timeseries = np.column_stack(aggregated_timeseries)
    ebm_corr_matrix = np.corrcoef(aggregated_timeseries.T)

    # Save numpy array to a file
    ebm_corr_file = os.path.abspath("./corr_matrix.npy")
    np.save(ebm_corr_file, ebm_corr_matrix)
    
    return ebm_corr_file, ebm_corr_matrix

# Set up a node for computing correlation matrix for one subject
compute_corr_node = pe.Node(Function(input_names=['rsfmri_file', 'atlas_file', 'roi_labels', 'ebm_stages'],
                                     output_names=['ebm_corr_file', 'ebm_corr_matrix'],
                                     function=compute_correlation),
                             name="compute_correlation")
compute_corr_node.inputs.atlas_file = atlas_data['image']
compute_corr_node.inputs.roi_labels = atlas_info['labelname'].tolist()
compute_corr_node.inputs.ebm_stages = ebm_stages

def aggregate_matrices(matrices, subject_ids, ebm_labels):
    import pandas as pd
    import numpy as np
    import os
    
    multi_columns = pd.MultiIndex.from_product([ebm_labels, ebm_labels], names=['from_region', 'to_region'])
    df = pd.DataFrame(np.array(matrices).reshape(len(subject_ids), -1), index=subject_ids, columns=multi_columns)
    
    file_path = os.path.abspath("aggregated_correlation_matrices.xlsx")
    df.to_excel(file_path)
    return file_path

aggregate_matrices_node = JoinNode(Function(input_names=['matrices', 'subject_ids', 'ebm_labels'],
                                            output_names=['aggregated_matrix_file'],
                                            function=aggregate_matrices),
                                   joinsource='infosource',
                                   joinfield=['matrices', 'subject_ids'],
                                   name="aggregate_matrices")
aggregate_matrices_node.inputs.subject_ids = subject_list
aggregate_matrices_node.inputs.ebm_labels = ebm_labels

# ---------------------- OUTPUT NODES ----------------------
print(colours.CGREEN + "Creating Output Nodes." + colours.CEND)

# DataSink
datasink = pe.Node(DataSink(base_directory=res_dir, container='connectivity'), name='datasink')

# ---------------------- CREATE WORKFLOW AND CONNECT NODES ----------------------
print(colours.CGREEN + 'Connecting Nodes.\n' + colours.CEND)

# Workflow setup
workflow = pe.Workflow(name='ebmconnectivity_wf', base_dir=wf_dir)
workflow.connect([
    (infosource, datasource, [('subject_id', 'subject_id')]),

    (datasource, compute_corr_node, [('rsfmri_file', 'rsfmri_file')]),
    (compute_corr_node, datasink, [('ebm_corr_file', 'compute_correlation.ebm_corr_matrices_file')]),

    (infosource, aggregate_matrices_node, [('subject_id', 'subject_ids')]),
    (compute_corr_node, aggregate_matrices_node, [('ebm_corr_matrix', 'matrices')]),
    (aggregate_matrices_node, datasink, [('aggregated_matrix_file', '@aggregated_matrix')]),
])


if __name__ == '__main__':
    workflow.write_graph(graph2use='orig')
    workflow.run(plugin=processing_type, plugin_args=plugin_args)
