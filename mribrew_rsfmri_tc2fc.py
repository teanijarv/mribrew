import os
import numpy as np
import pandas as pd
from mribrew.data_io import read_mat
from mribrew.atlases import fetch_dk_atlas
import ray

ray.init()

# define the data directory and other constants
data_dir = '/Users/to8050an/Documents/data/BF2/aparcaseg_rs_timecourses'
subfolder = 'aparcaseg'
tc_fname = 'aparcaseg.TC.mat'
labels_fname = 'aparcaseg.labels.mat'
export_dir = '/Users/to8050an/Documents/data/BF2/fc'

@ray.remote
def process_subject(mid):
    try:
        # read all time courses and labels
        tc_f_labels = read_mat(os.path.join(data_dir, mid, subfolder, labels_fname)).flatten()
        tc_f = read_mat(os.path.join(data_dir, mid, subfolder, tc_fname))

        # select only labels which are present in dk_labels
        _, labels = fetch_dk_atlas(lut_idx=True)

        # timecourses into df where rows = regions and columns = timepoints
        df_tc = pd.DataFrame(tc_f, index=tc_f_labels)
        df_tc = df_tc.loc[labels.values()]

        # calculate functional connectivity via pearson correlation
        df_fc = df_tc.T.corr()

        # export the functional connectivity matrix
        export_dir_ = os.path.join(export_dir, mid)
        os.makedirs(export_dir_, exist_ok=True)
        df_fc.to_csv(os.path.join(export_dir_, 'fc_aparcaseg.csv'))

        print(f'Successfully processed {mid}')
    except Exception as e:
        print(f'Error processing {mid}: {e}')

# list of subjects (directory names)
subjects = next(os.walk(data_dir))[1]

# run the processing function for all subjects in parallel
futures = [process_subject.remote(mid) for mid in subjects]
ray.get(futures)

ray.shutdown()