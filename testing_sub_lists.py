import os
import pandas as pd

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
subject_list_hc = pd.read_csv(os.path.join(misc_dir, 'hc_subjects.csv'), header=None)[0].tolist()

# Generate a list of all [subject, scan] sublists and list of controls whose response function will be averaged
subject_scan_list = []
subject_scan_hc_list = []
sub_hc_temp = []
for sub in subject_list:
    scans = next(os.walk(os.path.join(data_dir, 'proc', 'dwi_proc', sub)))[1]
    for scan in scans:
        subject_scan_list.append([sub, scan])
        if sub in subject_list_hc:
            sub_hc_temp.append(sub)
            subject_scan_hc_list.append([sub, scan])
    
print('All subjects-scans:', len(subject_scan_list))
print('HC subjects-scans:', len(sub_hc_temp))
# Select all non-HC and select only one batch
subject_scan_list = [[sub, scan] for [sub, scan] in subject_scan_list if sub not in sub_hc_temp]
sub_list_temp = set([sub for [sub, scan] in subject_scan_list])
print('Non-HC subjects-scans:', len(subject_scan_list))
print('Non-HC subjects:', len(sub_list_temp))
batch_len = len(subject_scan_list) // 3

batch1 = subject_scan_list[0:batch_len]
b1_sub_list_temp = set([sub for [sub, scan] in batch1])
print(f'Batch1 subjects: {len(b1_sub_list_temp)}')
print(f'Batch1 subjects-scans: {len(batch1)}')

batch2 = subject_scan_list[batch_len:2*batch_len]
b2_sub_list_temp = set([sub for [sub, scan] in batch2])
print(f'Batch2 subjects: {len(b2_sub_list_temp)}')
print(f'Batch2 subjects-scans: {len(batch2)}')

batch3 = subject_scan_list[2*batch_len:]
b3_sub_list_temp = set([sub for [sub, scan] in batch3])
print(f'Batch3 subjects: {len(b3_sub_list_temp)}')
print(f'Batch3 subjects-scans: {len(batch3)}')

