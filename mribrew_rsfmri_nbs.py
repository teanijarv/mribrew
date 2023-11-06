#%% 1 - DATA SETUP

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, image, plotting

from mribrew.utils import colours

# Set working directory
cwd = os.getcwd()
data_dir = os.path.join(cwd, 'data')
res_dir = os.path.join(data_dir, 'res')
connectivity_dir = os.path.join(res_dir, 'connectivity')

# Create NBS directory for later exporting
nbs_dir = os.path.join(res_dir, 'nbs')
os.makedirs(nbs_dir, exist_ok=True)

# Read in all subjects directories
all_subjects = next(os.walk(connectivity_dir))[1]
all_subjects_dirs = [os.path.join(connectivity_dir, subdir) for subdir in all_subjects]

# Load Schaefer atlas
schaefer_dict = datasets.fetch_atlas_schaefer_2018(n_rois=400) 
schaefer_img = nib.load(schaefer_dict['maps'])
schaefer_labels = {label.decode('utf-8'): float(idx+1) for idx, label in enumerate(schaefer_dict['labels'])}

# Tau asymmetry dataframe setup
df_ebm_tau_asymmetry = pd.read_csv(os.path.join(data_dir, 'bf2_ebm_tau_asymmetry_Ab+_EBM_I_agetaumatching_last_affected_minimal.csv'), index_col=0)
df_ebm_tau_asymmetry['raw_img_date_Tau_PET'] = pd.to_datetime(df_ebm_tau_asymmetry['raw_img_date_Tau_PET'].astype(int).astype(str))
print(f"Data shape (tau asymmetry): {df_ebm_tau_asymmetry.shape}")

# RSfMRI dataframe setup
df_mri = pd.DataFrame(data={'subject_res_id': all_subjects})
df_mri['mid'] = df_mri['subject_res_id'].str.replace('_subject_id_', '', 1).str.split('__').str[0]
df_mri['mri_date'] = pd.to_datetime(df_mri['subject_res_id'].str.split('__').str[1])
print(f"Data shape (MRI): {df_mri.shape}")

### MERGE PET AND MRI DATA

# Merging the dataframes to compute differences between PET and MRI dates
df_mri_tau = df_ebm_tau_asymmetry.merge(df_mri, on='mid', how='left').dropna()
df_mri_tau['pet_mri_date_diff'] = (df_mri_tau['raw_img_date_Tau_PET'] - df_mri_tau['mri_date']).abs()
print(f"Data shape (merging tau and MRI): {df_mri_tau.shape}")

# Selecting the rows with minimum date difference for each 'mid'
idx = df_mri_tau.groupby('mid')['pet_mri_date_diff'].idxmin()
df_mri_tau = df_mri_tau.loc[idx]
print(f"Data shape (closest PET and MRI only): {df_mri_tau.shape}")

# Drop participants with over 1-year difference in PET and MRI
df_mri_tau = df_mri_tau[df_mri_tau['pet_mri_date_diff'] <= pd.Timedelta('365 days')]
print(f"Data shape (less than a year difference only): {df_mri_tau.shape}")

# Create columns for all RSfMRI correlation matrices file locations
df_mri_tau['subject_res_dir'] = df_mri_tau['subject_res_id'].map(lambda x: os.path.join(connectivity_dir, x))
df_mri_tau['desikan_corr_file'] = df_mri_tau['subject_res_dir'].map(lambda x: os.path.join(x, 'sub_corrmat_desikan.pkl'))
df_mri_tau['schaefer_corr_file'] = df_mri_tau['subject_res_dir'].map(lambda x: os.path.join(x, 'sub_corrmat_schaefer.pkl'))
print(f"Data shape (adding file paths): {df_mri_tau.shape}")

### USER INPUT FOR THE SAMPLE

# Ask the user for input
ebm_group = input("What EBM group do you want to analyse? (leave empty for all OR write 'I', 'II', 'III', or 'IV'): ")
valid_groups = ['I', 'II', 'III', 'IV']
if ebm_group in valid_groups:
    df_mri_tau = df_mri_tau.loc[df_mri_tau['last_affected_ebm']==ebm_group]
    print(colours.CGREEN + f"Using EBM {ebm_group} group for analysis." + colours.CEND)
else:
    print(colours.CGREEN + "Using the entire group for analysis." + colours.CEND)

### SAMPLE GROUPING BASED ON TAU ASYMMETRY

def merge_corrmatrices_for_group(df_group, corr_file_col, atlas_labels):
    import numpy as np
    from mribrew.data_io import read_pickle
    corrmatrices_group = np.zeros((len(atlas_labels.keys()), len(atlas_labels.keys()), len(df_group)))
    for idx in range(len(df_group)):
        sub_corrmat = read_pickle(df_group.iloc[idx][corr_file_col])
        corrmatrices_group[:, :, idx] = sub_corrmat
    return corrmatrices_group

# Define columns for grouping asymmetry
asymmetry_binary_grouping_col = 'last_affected_ebm_asymmetry_index_group_binary'
asymmetry_hemispheric_grouping_col = 'last_affected_ebm_asymmetry_index_group'

# Create dataframes for each of the asymmetry groups
df_symmetric = df_mri_tau[df_mri_tau[asymmetry_binary_grouping_col]==0].reset_index(drop=True)
df_asymmetric = df_mri_tau[df_mri_tau[asymmetry_binary_grouping_col]==1].reset_index(drop=True)
df_right_asymmetric = df_mri_tau[df_mri_tau[asymmetry_hemispheric_grouping_col]=='right_asymmetric'].reset_index(drop=True)
df_left_asymmetric = df_mri_tau[df_mri_tau[asymmetry_hemispheric_grouping_col]=='left_asymmetric'].reset_index(drop=True)

print(f"Data shape (symmetric group): {df_symmetric.shape}")
print(f"Data shape (asymmetric group): {df_asymmetric.shape}")
print(f"Data shape (right asymmetric group): {df_right_asymmetric.shape}")
print(f"Data shape (left asymmetric group): {df_left_asymmetric.shape}")

schaefer_corrmatrices_symmetric = merge_corrmatrices_for_group(df_symmetric, 'schaefer_corr_file', schaefer_labels)
schaefer_corrmatrices_asymmetric = merge_corrmatrices_for_group(df_asymmetric, 'schaefer_corr_file', schaefer_labels)
schaefer_corrmatrices_right_asymmetric = merge_corrmatrices_for_group(df_right_asymmetric, 'schaefer_corr_file', schaefer_labels)
schaefer_corrmatrices_left_asymmetric = merge_corrmatrices_for_group(df_left_asymmetric, 'schaefer_corr_file', schaefer_labels)

### MASKING

def create_interhemispheric_mask(labels):
    import numpy as np
    num_regions = len(labels)
    hemisphere_boundary = num_regions // 2  # Assuming an equal split
    mask = np.zeros((num_regions, num_regions), dtype=int)
    mask[:hemisphere_boundary, hemisphere_boundary:] = 1
    mask[hemisphere_boundary:, :hemisphere_boundary] = 1
    return mask

def apply_corrmat_mask(matrices, mask):
    import numpy as np
    masked_matrices = np.zeros_like(matrices)
    for i in range(matrices.shape[-1]):
        masked_matrices[:, :, i] = matrices[:, :, i] * mask
    return masked_matrices

# Create HCP structural connectivity mask
hcp_sc_ctx = np.load(os.path.join(data_dir, 'hcp_data/sc_ctx_schaefer400.npy'))
hcp_sc_mask = np.where(hcp_sc_ctx >= np.percentile(hcp_sc_ctx, 95), 1, 0)
# Create HCP functional connectivity mask
hcp_fc_ctx = np.load(os.path.join(data_dir, 'hcp_data/fc_ctx_schaefer400.npy'))
hcp_fc_mask = np.where(hcp_fc_ctx >= np.percentile(hcp_fc_ctx, 95), 1, 0)
# Create inter-hemispheric mask
inter_hemispheric_mask = create_interhemispheric_mask(schaefer_labels)

# Ask user which mask to use
mask_option = input("What masks do you want to apply (leave empty for no mask OR write 'hcp_sc', 'hcp_fc', 'interhemi', 'hcp_sc+interhemi', 'hcp_fc+interhemi', or 'hcp_sc+hcp_fc'): ")
valid_options = ['hcp_sc', 'hcp_fc', 'interhemi', 'hcp_sc+interhemi', 'hcp_fc+interhemi', 'hcp_sc+hcp_fc']
if mask_option in valid_options:
    if mask_option=='hcp_sc':
        mask = hcp_sc_mask
    elif mask_option=='hcp_fc':
        mask = hcp_fc_mask
    elif mask_option=='interhemi':
        mask = inter_hemispheric_mask
    elif mask_option=='hcp_sc+interhemi':
        mask = inter_hemispheric_mask * hcp_sc_mask
    elif mask_option=='hcp_fc+interhemi':
        mask = inter_hemispheric_mask * hcp_fc_mask
    elif mask_option=='hcp_sc+hcp_fc':
        mask = hcp_sc_mask * hcp_fc_mask
    print(colours.CGREEN + f"Using {mask_option} mask(s) for analysis." + colours.CEND)
else:
    mask = np.ones((400, 400))
    print(colours.CGREEN + "Using no masks for analysis." + colours.CEND)

schaefer_corrmatrices_symmetric_masked = apply_corrmat_mask(schaefer_corrmatrices_symmetric, mask)
schaefer_corrmatrices_asymmetric_masked = apply_corrmat_mask(schaefer_corrmatrices_asymmetric, mask)
schaefer_corrmatrices_right_asymmetric_masked = apply_corrmat_mask(schaefer_corrmatrices_right_asymmetric, mask)
schaefer_corrmatrices_left_asymmetric_masked = apply_corrmat_mask(schaefer_corrmatrices_left_asymmetric, mask)

# %% 2 - NBS

### NETWORK-BASED STATISTIC

def apply_nbs_and_plot(corrmatrices_group1, corrmatrices_group2, nbs_kwargs, 
                       template_img, coords, plot_title, nbs_dir):
    import numpy as np
    from bct import nbs
    import matplotlib.pyplot as plt
    from nilearn import plotting
    pvals, adj_matrix, null = nbs.nbs_bct(corrmatrices_group1, corrmatrices_group2, **nbs_kwargs)

    fig = plt.figure(dpi=150)
    fig = plotting.plot_connectome(adj_matrix, coords, black_bg=template_img, node_size=2, 
                                   edge_vmin=np.min(adj_matrix), edge_vmax=np.max(adj_matrix),
                                   edge_cmap='Set1', title=plot_title, colorbar=True)
    plt.show()
    fig.savefig(os.path.join(nbs_dir, f"NBS_{plot_title}.png"), dpi=300)
    
    return pvals, adj_matrix, null

# Define the NBS parameters
nbs_threshold = float(input("Enter NBS threshold: "))
nbs_permutations = int(input("Enter the amount of permutations: "))
nbs_tail = str(input("Enter the tail of the test ('left' for asymmetric<symmetric , 'right' for asymmetric>symmetric, 'both'): "))
nbs_kwargs = dict(thresh=nbs_threshold, k=nbs_permutations, tail=nbs_tail, paired=False)
print(colours.CGREEN + f"NBS running with following parameters: {nbs_kwargs}" + colours.CEND)

template_img = image.load_img(datasets.load_mni152_template(resolution=1))
coords = plotting.find_parcellation_cut_coords(schaefer_img)

# Apply NBS and plot connectome for the left asymmetric vs symmetric groups
plot_title_Lasym_sym = f"left_asym (n={schaefer_corrmatrices_left_asymmetric_masked.shape[-1]}) " \
                       f"vs sym (n={schaefer_corrmatrices_symmetric_masked.shape[-1]})\n" \
                       f"(thresh={nbs_kwargs['thresh']}, perms={nbs_kwargs['k']}, mask={mask_option})"
pvals_Lasym_sym, adjmat_Lasym_sym, null_Lasym_sym = apply_nbs_and_plot(schaefer_corrmatrices_left_asymmetric_masked, 
                                                                       schaefer_corrmatrices_symmetric_masked, 
                                                                       nbs_kwargs, template_img, coords, 
                                                                       plot_title_Lasym_sym, nbs_dir)

# Apply NBS and plot connectome for the right asymmetric vs symmetric groups
plot_title_Rasym_sym = f"right_asym (n={schaefer_corrmatrices_right_asymmetric_masked.shape[-1]}) " \
                       f"vs sym (n={schaefer_corrmatrices_symmetric_masked.shape[-1]})\n" \
                       f"(thresh={nbs_kwargs['thresh']}, perms={nbs_kwargs['k']}, mask={mask_option})"
pvals_Rasym_sym, adjmat_Rasym_sym, null_Rasym_sym = apply_nbs_and_plot(schaefer_corrmatrices_right_asymmetric_masked, 
                                                                       schaefer_corrmatrices_symmetric_masked, 
                                                                       nbs_kwargs, template_img, coords, 
                                                                       plot_title_Rasym_sym, nbs_dir)

# %%

def get_significant_adj_matrix(adj_matrix, pvals, alpha=0.05):
    import numpy as np
    # Go through each unique value/component in the adjacency matrix
    significant_adj_matrix = np.copy(adj_matrix)
    for comp_val, pval in enumerate(pvals, start=1):
        # If the p-value is greater than alpha, set the corresponding values in the matrix to zero
        if pval > alpha:
            significant_adj_matrix[significant_adj_matrix==comp_val] = 0
        else:
            print(f"Component {comp_val} has p-value of {pval} (i.e., < {alpha} -> significant)")
    return significant_adj_matrix

def adj_matrix_to_binary_masks(matrix):
    import numpy as np
    # Get all unique values except zero
    unique_components = np.unique(matrix[matrix != 0])
    # Create binary masks for all unique values wih the shape (N, N, C)
    masks = np.zeros(matrix.shape + (len(unique_components),), dtype=int)
    for index, component in enumerate(unique_components):
        masks[:, :, index] = (matrix == component).astype(int)
    return masks

def plot_nbs_connectome(adj_matrix, template_img, coords, plot_title, nbs_dir, cmap='bwr', cbar=True):
    import matplotlib.pyplot as plt
    from nilearn import plotting

    fig = plt.figure(dpi=150)
    fig = plotting.plot_connectome(adj_matrix, coords, black_bg=template_img, node_size=2, 
                                   title=plot_title, edge_cmap=cmap, colorbar=cbar)
    plt.show()
    fig.savefig(os.path.join(nbs_dir, f"NBS_significant_{plot_title}.png"), dpi=300)

adjmat_Lasym_sym_significant = get_significant_adj_matrix(adjmat_Lasym_sym, pvals_Lasym_sym, alpha=0.05)
significant_Lasym_sym_masks = adj_matrix_to_binary_masks(adjmat_Lasym_sym_significant)
for i in range(significant_Lasym_sym_masks.shape[-1]):
    plot_nbs_connectome(significant_Lasym_sym_masks[:, :, i], template_img, coords, 
                        plot_title_Lasym_sym+f"_{i}", nbs_dir, cmap='tab10')
    corrmat_left_asym_component = np.mean(schaefer_corrmatrices_left_asymmetric, axis=2) * significant_Lasym_sym_masks[:, :, i]
    plot_nbs_connectome(corrmat_left_asym_component, template_img, coords, "corr "+plot_title_Lasym_sym+f"_{i}", nbs_dir, cmap='bwr', cbar=True)

adjmat_Rasym_sym_significant = get_significant_adj_matrix(adjmat_Rasym_sym, pvals_Rasym_sym, alpha=0.05)
significant_Rasym_sym_masks = adj_matrix_to_binary_masks(adjmat_Rasym_sym_significant)
for i in range(significant_Rasym_sym_masks.shape[-1]):
    plot_nbs_connectome(significant_Rasym_sym_masks[:, :, i], template_img, coords, 
                        plot_title_Rasym_sym+f"_{i}", nbs_dir, cmap='tab10')
    corrmat_right_asym_component = np.mean(schaefer_corrmatrices_right_asymmetric, axis=2) * significant_Rasym_sym_masks[:, :, i]
    plot_nbs_connectome(corrmat_right_asym_component, template_img, coords, "corr "+plot_title_Rasym_sym+f"_{i}", nbs_dir, cmap='bwr', cbar=True)


# %%

from nilearn import plotting

def generate_roi_to_network(schaefer_labels, network_mapping):
    roi_to_network = {}
    for label, roi_num in schaefer_labels.items():
        network_abbrev = label.split("_")[2]
        roi_to_network[roi_num] = network_mapping.get(network_abbrev, "Unknown")
    return roi_to_network

def plot_specific_network(network_name, networks_of_nodes, coords, adj_matrix, template_img, plot_title): 
    import numpy as np
    from nilearn import plotting
    import matplotlib.pyplot as plt 
    # Filter nodes corresponding to the specific network
    specific_network_nodes = [i for i, network in enumerate(networks_of_nodes) if network==network_name]
    specific_network_coords = [coords[node] for node in specific_network_nodes]
    # Update the adjacency matrix to reflect only connections of the specific network nodes
    specific_network_adj_matrix = adj_matrix[np.ix_(specific_network_nodes, specific_network_nodes)]
    # Check if there are any non-zero connections in the adjacency matrix
    if not np.any(specific_network_adj_matrix):
        colorbar=False
    else:
        colorbar=True
        print(f"{np.count_nonzero(specific_network_adj_matrix)} significant connections " \
            f"found for {network_name} network")

        # Plot the connectome for the specific network nodes
        #fig = plt.figure(dpi=150)
        fig = plotting.plot_connectome(specific_network_adj_matrix, specific_network_coords, node_size=2,
                                        title=f"{network_name} Network - {plot_title}",
                                        black_bg=template_img,
                                        colorbar=colorbar)
        plt.show()
        #fig.savefig(os.path.join(nbs_dir, f"{plot_title}.png"), dpi=300)

# Define the mapping from network abbreviations to full network names
network_mapping = {
    "Vis": "Visual",
    "SomMot": "SomatoMotor",
    "DorsAttn": "DorsalAttention",
    "SalVentAttn": "SalienceVentralAttention",
    "Limbic": "Limbic",
    "Cont": "Contingent",
    "Default": "DefaultMode"
}

roi_to_network = generate_roi_to_network(schaefer_labels, network_mapping)

# Map the nodes to their respective networks
networks_of_nodes = [roi_to_network[node] for node in roi_to_network]
for network in network_mapping.values():
    try:
        plot_title = f"L_asym (n={schaefer_corrmatrices_left_asymmetric_masked.shape[-1]})"
        plot_specific_network(network, networks_of_nodes, coords, corrmat_left_asym_component, template_img, plot_title)
    except:
        continue
    try:
        plot_title = f"L_asym (n={schaefer_corrmatrices_right_asymmetric_masked.shape[-1]})"
        plot_specific_network(network, networks_of_nodes, coords, corrmat_right_asym_component, template_img, plot_title)
    except:
        continue
# %%
