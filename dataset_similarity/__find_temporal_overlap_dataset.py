import os
from tqdm import tqdm
import pandas as pd
import numpy as np

'''
Input: The data from erddap has been downloaded in the directory 'erddap data'

This file traverse the timestamp of all datasets and calculate the overlapping dataset similarity
the window size of 24 hours has been set to check overlap
Output: date_df_temporal_matching_score
'''

if __name__ == '__main__':
    dataset_date_map = {}
    for root, dir, files in os.walk("../erddap data/"):
        pbar_ = tqdm(files)
        for fname in pbar_:
            pbar_.set_description(fname)
            if fname.lower().endswith(".csv"):
                fl_path_ = os.path.join(root, fname)
                df = pd.read_csv(fl_path_).dropna()
                if 'latitude' not in df.columns:
                    print(f"Skipping [{fname}]")
                    continue

                s_ = pd.to_datetime(df['time'].iloc[1:], format='%Y-%m-%dT%H:%M:%SZ').dt.strftime('%Y%m%d').astype(int)
                dataset_date_map[fname] = s_.unique()

    list_of_das_variables_ = []
    item_labels = []
    for d_name, dates_list in dataset_date_map.items():
        item_labels.append(d_name)
        list_of_das_variables_.append(dates_list)

    num_items = len(list_of_das_variables_)
    # Initialize a 50x50 matrix with zeros
    matching_matrix = np.zeros((num_items, num_items), dtype=int)

    lod = list_of_das_variables_
    # Calculate matching scores
    for i in tqdm(range(num_items)):
        for j in range(num_items):
            total_ = len(lod[i])
            match_count = len(set(lod[i]).intersection(set(lod[j])))
            match_count = int(float(match_count / total_) * 100)

            matching_matrix[i][j] = match_count

    df_matching = pd.DataFrame(matching_matrix, index=item_labels, columns=item_labels)
    df_matching.to_csv(f"./res/date_df_temporal_matching_score.csv")

