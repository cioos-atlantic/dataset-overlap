import os

import traceback
import pandas as pd
from functools import reduce
import numpy as np
from tqdm import tqdm
from p_tqdm import p_umap
from antropy import sample_entropy, svd_entropy


from qa_qc.ai_utils import QartodFlags, get_xy_from_dataframe
import math

def merge_datasets(parent_dicr_ = "../full data/Maritimes Region Atlantic Zone Off-Shelf Monitoring Program (AZOMP) Rosette/"):
    lst_of_fname_ = []

    for root, dir, files in os.walk(parent_dicr_):
        for f in files:
            df_path = os.path.join(parent_dicr_, f)
            lst_of_fname_.append(df_path)


    left_ = pd.read_csv(lst_of_fname_[0], skiprows=[1], parse_dates=['measurement_time','time'])
    for i in range(1, len(lst_of_fname_)):
        right = pd.read_csv(lst_of_fname_[i], skiprows=[1], parse_dates=['measurement_time','time'])
        print(f"merging [ {i} ] -> [{lst_of_fname_[i]}]")
        merged_all = pd.merge(left_, right, on=['platform_id','measurement_time', 'depth','latitude', 'longitude', 'time'], how='outer')
        left_ = merged_all
        left_.to_csv(
            f'{parent_dicr_}Maritimes Region Atlantic Zone Off-Shelf Monitoring Program (AZOMP) Rosette_FLAGGED.csv', index=False)


def get_file_names(dir__):
    file_names = []
    for root_, dr_, files in os.walk(dir__):
        for fl in files:
            if fl.lower().endswith(".csv"):
                file_names.append(os.path.join(root_, fl))
    return file_names


def controlled_undersample(X, majority_class=QartodFlags.GOOD, reference_class=QartodFlags.SUSPECT, ratio=10):
    mask_majority = X[:, -1] == majority_class
    mask_reference = X[:, -1] == reference_class
    mask_other = X[:, -1] != majority_class

    # Subsets
    X_majority = X[mask_majority]
    X_other = X[mask_other]

    # Reference count from class 3
    n_reference = np.sum(mask_reference)
    n_majority_desired = min(len(X_majority), ratio * n_reference)

    # Sample from majority class
    indices = np.random.choice(len(X_majority), size=n_majority_desired, replace=False)
    X_majority_sampled = X_majority[indices]

    # Combine
    X_final = np.concatenate([X_other, X_majority_sampled], axis=0)
    print(f"ALL / Controlled Sampling : {X.shape} --> {X_final.shape}")
    return X_final



def remove_duplicates_from_nparray(nparr_, target_flag):
    """
    last column is the label/flag column in nparr_
    :param nparr_:
    :param flag:
    :return:
    """
    mask_g = nparr_[:, -1] == target_flag
    good_rows = nparr_[mask_g]
    mask = nparr_[:, -1] != target_flag
    other_rows_ = nparr_[mask]

    unique_unique_rows = np.unique(good_rows, axis=0)
    X = np.concatenate([unique_unique_rows, other_rows_])
    return X

eov_range = (-2.0, 40)
eov_col_name = 'temperature'
eov_flag_name = 'qc_flag_temperature'
window_hour = 12
min_rows_in_a_chunk = 6
minimum_rows_for_each_group = 50

from scipy.stats import skew, kurtosis
import pickle

if __name__ == '__main__':

    # merge_datasets()
    # chunk_dir = "CIOOS-Full-Data/"
    chunk_dir = "D:/CIOOS-Full-Data/"
    # csv_name = "D:/CIOOS-Full-Data/Digby County Water Quality Data_FlagCode.csv"
    # with open(csv_name, 'r') as file:
    #     with open(csv_name.replace(".csv", "_SAMPLE.csv"), "w") as outfile:
    #         count_ = 0
    #         for line in file:
    #             outfile.write(line)  # Process each line
    #             count_ += 1
    #             if count_ > 5000:
    #                 break



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
    # dir_name = os.path.join(chunk_dir,"chunking")
    # for entry in os.listdir(chunk_dir):
    #
    #     if not entry.endswith(".csv"):
    #         continue
    #
    #     csv_name = os.path.join(chunk_dir, entry)
    #     new_directory_name =  os.path.join(dir_name, entry.split(" ")[0])
    #     if os.path.exists(new_directory_name):
    #         print(f"---- SKIPPING --- [{new_directory_name}]")
    #         continue
    #     print(f"Processing :-> [{entry}]")
    #     os.mkdir(new_directory_name)
    #     save_name =  os.path.join(new_directory_name, os.path.basename(csv_name).replace("_FlagCode.csv",""))
    #     df_ = pd.read_csv(csv_name, parse_dates=['time'], skiprows=[1])
    #     df_['latitude'] = df_['latitude'].astype(np.float32).round(4)
    #     df_['longitude'] = df_['longitude'].astype(np.float32).round(4)
    #
    #     groups_ = df_.groupby(by=['station', 'sensor_serial_number'])
    #     id_ = 1
    #     for grp_, chunk in groups_:
    #         if chunk.shape[0] < minimum_rows_for_each_group:
    #             continue
    #         lat_uni_ = chunk['latitude'].unique()
    #         lon_uni_ = chunk['longitude'].unique()
    #         a_ = lat_uni_.std()
    #         b_ = lon_uni_.std()
    #         d_threshold = (pd.to_datetime(chunk['time'].max()) - pd.to_datetime(chunk['time'].min())).days > 1
    #         # check if the data of more than 1 day is collected
    #         if not d_threshold:
    #             continue
    #
    #         if a_ > 0.001:
    #             rows__ = chunk.shape[0]
    #             total_rows__ = 0
    #             lat_groups_ = group_with_dbscan(values=lat_uni_)
    #             for j, lat_grp in enumerate(lat_groups_):
    #                 sub_chunk_ = chunk[ (chunk['latitude'] >= min(lat_grp)) &  (chunk['latitude'] <= max(lat_grp)) ]
    #                 sub_b_ = sub_chunk_['longitude'].astype(np.float32).unique()
    #                 assert sub_b_.std() <= 0.001, f"many longitudes {lat_uni_}"
    #                 sub_chunk_.to_csv(save_name + f"-{id_}-{j}.csv", index=False)
    #                 total_rows__ += sub_chunk_.shape[0]
    #                 print(save_name + f"-{id_}-{j}.csv")
    #             assert total_rows__ == rows__, f"LAT sub chunk rows not equal to main chunk [{lat_grp}] - [{lat_uni_}]"
    #
    #         else:
    #             if b_ > 0.001:
    #                 rows__ = chunk.shape[0]
    #                 total_rows__ = 0
    #                 lon_groups_ = group_with_dbscan(values=lon_uni_)
    #                 for j, lon_grp in enumerate(lon_groups_):
    #                     sub_chunk_ = chunk[(chunk['longitude'] >= min(lon_grp)) & (chunk['longitude'] <= max(lon_grp))]
    #                     sub_chunk_.to_csv(save_name + f"-{id_}-{j}.csv", index=False)
    #                     print(save_name + f"-{id_}-{j}.csv")
    #                     total_rows__ += sub_chunk_.shape[0]
    #
    #                 assert total_rows__ == rows__, f"LON sub chunk rows not equal to main chunk [{lon_groups_}] - [{lon_groups_}]"
    #             else:
    #                 chunk.to_csv(save_name+f"-{id_}.csv", index=False)
    #         id_+=1
    #
    #     print(f"==== FINSIHED === {entry}")
    ##################################################



    ################### DATA PREPARATION FOR ML MODEL ####################
    chunk_dir = os.path.join(chunk_dir, "chunking")
    datasets__ = []
    for entry in os.listdir(chunk_dir):
        folder_name = os.path.join(chunk_dir, entry)
        if os.path.isdir(folder_name):
            datasets__.append(entry)

    datasets__.sort()
    datasets__ = [ "Victoria"]
    # datasets__ = ["Cape"]

    for dataset_ in datasets__:

        file_names = get_file_names(os.path.join(chunk_dir, dataset_))
        # file_names = ["D:/CIOOS-Full-Data/chunking/Digby/Digby County Water Quality Data.csv-17-0.csv"]

        x_np_array_file = os.path.join(chunk_dir, dataset_, f"{dataset_}-X_np_array.pkl")
        if os.path.exists(x_np_array_file):
            print(f"-- Skipping  - [{dataset_}] because NPARRAY file exist")
            continue
        print(f"-- Looking Into - [{dataset_}]")



        def parl_func(file_name):
            print(f"Processing : [{file_name}]")
            df = pd.read_csv(file_name, usecols=['time', eov_flag_name, eov_col_name])
            df.dropna(inplace=True, axis=0)
            if df.shape[0] < (min_rows_in_a_chunk*2):
                return None

            df['time'] = pd.to_datetime(df['time'])

            # Feature engineering from window
            X1= get_xy_from_dataframe(df, window_hours=window_hour, min_rows_in_chunk=min_rows_in_a_chunk, eov_col_name=eov_col_name,
                                           eov_flag_name=eov_flag_name , dataset_name_string=file_name)
            if np.sum(np.isnan(X1)) > 0:
                print(f" ---> DATASET WITH NAN: {file_name}")

            return X1


        F_A = None
        for fname in tqdm(file_names):
            X1 = parl_func(fname)
        # lst_of_xy = p_umap(parl_func, file_names)
        # for X1 in lst_of_xy:
            if X1 is None:
                continue
            if F_A is None:
                F_A = X1
            else:
                F_A = np.concatenate((F_A, X1), axis=0)

        X = controlled_undersample(F_A, majority_class=QartodFlags.GOOD, reference_class=QartodFlags.SUSPECT, ratio=10)
        t_ = F_A.shape
        F_A = remove_duplicates_from_nparray(F_A, QartodFlags.GOOD)

        X_ = F_A[:, :-1]
        Y_ = F_A[:, -1]
        print(f"{t_}  --->  {F_A.shape} ->   {Y_.shape}  ,  {X_.shape}   ===>  .PKL ")

        pickle.dump(X_, open(x_np_array_file, 'wb'))
        pickle.dump(Y_, open(os.path.join(chunk_dir, dataset_,f"{dataset_}-Y_np_array.pkl"), 'wb'))

    #####################################



    ### MERGING NPARRAY#########
    # parent_dir = "D:/CIOOS-Full-Data/chunking/"
    # train_ = ["Antigonish NPARRAY/", "Cape Breton NPARRAY/", "Colchester NPARRAY/"]
    # x_s, y_s  = [], []
    # for tr_ in train_:
    #     X = pickle.load(open(os.path.join(parent_dir,tr_, 'X_np_array.pkl'), 'rb'))
    #     y = pickle.load(open(os.path.join(parent_dir,tr_, 'Y_np_array.pkl'), 'rb'))
    #     x_s.append(X)
    #     y_s.append(y)
    #     print(f"{tr_} :->   {X.shape}")
    #
    # X = np.concatenate(x_s, axis=0)
    # y = np.concatenate(y_s, axis=0)
    # print(f"TOTAL: {X.shape}")
    #
    # pickle.dump(X, open(os.path.join(parent_dir, 'X_np_array_antigonish_cape_colchester.pkl'), 'wb'))
    # pickle.dump(y, open(os.path.join(parent_dir, 'Y_np_array_antigonish_cape_colchester.pkl'), 'wb'))