import os

import traceback
import pandas as pd
from functools import reduce
import numpy as np
from tqdm import tqdm
from p_tqdm import p_umap
from antropy import sample_entropy, svd_entropy
from sklearn.cluster import DBSCAN
import numpy as np


from qa_qc.ai_utils import QartodFlags, generate_time_windows
import math




def group_with_dbscan(values, eps=0.001):
    # Convert to 2D array as required by DBSCAN

    X = np.array(values).reshape(-1, 1)

    # Fit DBSCAN
    db = DBSCAN(eps=eps, min_samples=1)
    labels = db.fit_predict(X)

    # Group values by cluster label
    groups = []
    for label in sorted(set(labels)):
        group = [v for v, l in zip(values, labels) if l == label]
        groups.append(group)

    return groups


eov_range = (-2.0, 40)
eov_col_name = 'temperature'
eov_flag_name = 'qc_flag_temperature'
window_hour = 12
min_rows_in_a_chunk = 6
minimum_rows_for_each_group = 50

from scipy.stats import skew, kurtosis
import pickle

if __name__ == '__main__':

    chunk_dir = "CIOOS-Full-Data/"



    ################# REPLACE FLAGS FROM STRING TO INT FROM CSV##################
    # def custom_replacement(value):
    #     if value == 'Not Evaluated':
    #         return QartodFlags.UNKNOWN
    #     elif value == 'Pass':
    #         return QartodFlags.GOOD
    #     elif value == 'Suspect/Of Interest':
    #         return QartodFlags.SUSPECT
    #     elif value == 'Fail':
    #         return QartodFlags.FAIL
    #     elif math.isnan(float(value)):
    #         return -1
    #     else:
    #         print(f"Unknown [{value}]")
    #
    #     return value
    #
    # df_chunks = pd.read_csv(csv_name, chunksize=10000)
    # columns_ = None
    # header_written = False
    # for df in df_chunks:
    #     if columns_ is None:
    #         lst_col  = list(df.columns)
    #         columns_ = [col for col in lst_col if "qc" in col.lower()]
    #     for col in columns_:
    #         df[col] = df[col].apply(custom_replacement)
    #
    #     df.to_csv(csv_name.replace(".csv", "_FlagCode.csv") , index=False, mode='a', header= not header_written)
    #     header_written = True
    #######################################################


    # ############ GROUPING DATA BY STATION AND SENSOR ##############
    dir_name = os.path.join(chunk_dir,"chunking")
    for entry in os.listdir(chunk_dir):

        if not entry.endswith(".csv"):
            continue

        csv_name = os.path.join(chunk_dir, entry)
        new_directory_name =  os.path.join(dir_name, entry.split(" ")[0])
        if os.path.exists(new_directory_name):
            print(f"---- SKIPPING --- [{new_directory_name}]")
            continue
        print(f"Processing :-> [{entry}]")
        os.mkdir(new_directory_name)
        save_name =  os.path.join(new_directory_name, os.path.basename(csv_name).replace("_FlagCode.csv",""))
        df_ = pd.read_csv(csv_name, parse_dates=['time'], skiprows=[1])
        df_['latitude'] = df_['latitude'].astype(np.float32).round(4)
        df_['longitude'] = df_['longitude'].astype(np.float32).round(4)

        groups_ = df_.groupby(by=['station', 'sensor_serial_number'])
        id_ = 1
        for grp_, chunk in groups_:
            if chunk.shape[0] < minimum_rows_for_each_group:
                continue
            lat_uni_ = chunk['latitude'].unique()
            lon_uni_ = chunk['longitude'].unique()
            a_ = lat_uni_.std()
            b_ = lon_uni_.std()
            d_threshold = (pd.to_datetime(chunk['time'].max()) - pd.to_datetime(chunk['time'].min())).days > 1
            # check if the data of more than 1 day is collected
            if not d_threshold:
                continue

            if a_ > 0.001:
                rows__ = chunk.shape[0]
                total_rows__ = 0
                lat_groups_ = group_with_dbscan(values=lat_uni_)
                for j, lat_grp in enumerate(lat_groups_):
                    sub_chunk_ = chunk[ (chunk['latitude'] >= min(lat_grp)) &  (chunk['latitude'] <= max(lat_grp)) ]
                    sub_b_ = sub_chunk_['longitude'].astype(np.float32).unique()
                    assert sub_b_.std() <= 0.001, f"many longitudes {lat_uni_}"
                    sub_chunk_.to_csv(save_name + f"-{id_}-{j}.csv", index=False)
                    total_rows__ += sub_chunk_.shape[0]
                    print(save_name + f"-{id_}-{j}.csv")
                assert total_rows__ == rows__, f"LAT sub chunk rows not equal to main chunk [{lat_grp}] - [{lat_uni_}]"

            else:
                if b_ > 0.001:
                    rows__ = chunk.shape[0]
                    total_rows__ = 0
                    lon_groups_ = group_with_dbscan(values=lon_uni_)
                    for j, lon_grp in enumerate(lon_groups_):
                        sub_chunk_ = chunk[(chunk['longitude'] >= min(lon_grp)) & (chunk['longitude'] <= max(lon_grp))]
                        sub_chunk_.to_csv(save_name + f"-{id_}-{j}.csv", index=False)
                        print(save_name + f"-{id_}-{j}.csv")
                        total_rows__ += sub_chunk_.shape[0]

                    assert total_rows__ == rows__, f"LON sub chunk rows not equal to main chunk [{lon_groups_}] - [{lon_groups_}]"
                else:
                    chunk.to_csv(save_name+f"-{id_}.csv", index=False)
            id_+=1

        print(f"==== FINSIHED === {entry}")
    ##################################################



