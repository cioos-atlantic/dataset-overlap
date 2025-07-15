import pickle

import pandas as pd

import numpy as np
import pandas as pd
from ioos_qc.config import QcConfig
from ioos_qc.axds import valid_range_test

from qa_qc.qaqc_utils_ import cluster_geodetic_points, get_eov_info, u2c_, get_qaqc_config, \
    check_unit_similarity
from qa_qc.unit_converter import unit_convert

if __name__ == '__main__':

    # pkl_nparray = "D:/CIOOS-Full-Data/chunking/Annapolis/Annapolis-X_np_array.pkl"
    #
    # X = pickle.load(open(pkl_nparray, 'rb'))
    # for i in range(0, X.shape[1]):
    #     print(f"{i} ->  {np.unique(np.isnan(X[:, i]))}")
    # X = np.round(X, 3)

    csv_ = pd.read_csv("D:/CIOOS-Full-Data/Antigonish County Water Quality Data_FlagCode.csv", skiprows=[1])

    print(csv_['latitude'].unique())

    a = 10
