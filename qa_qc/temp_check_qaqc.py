
import pandas as pd

import numpy as np
import pandas as pd
from ioos_qc.config import QcConfig
from ioos_qc.axds import valid_range_test

from qa_qc.qaqc_utils_ import cluster_geodetic_points, get_eov_info, u2c_, get_qaqc_config, \
    check_unit_similarity
from qa_qc.unit_converter import unit_convert

if __name__ == '__main__':


    filepath = "../full data/SMA_bay_of_exploits_bab9_79bf_9fe6.csv"
    unit_df_ = pd.read_csv(filepath, nrows=1)
    df_ = pd.read_csv(filepath,skiprows = [1])
    valid_cols = [vc for vc in df_.columns if get_eov_info(vc) is not None]

    u_df_ = cluster_geodetic_points(df_, max_distance_km=2.0)
    groups_ = u_df_.groupby(by=['cluster_id'])
    for label, grp_chunk in groups_:
        for col in valid_cols:
            eov_values = grp_chunk[col]
            eov_unit = unit_df_[col].iloc[0]
            c1_ = u2c_[str(eov_unit).lower()]
            standard_name = get_eov_info(col)
            c2_ = get_eov_info(standard_name)
            assert c1_ == c2_, "different categories"
            confg_ = get_qaqc_config(standard_name)
            need_conversion_ = check_unit_similarity(confg_['units'], eov_unit)
            qc_config = {
                "qartod": confg_
            }
            qc = QcConfig(qc_config)
            if need_conversion_:
                eov_values = unit_convert(eov_unit, eov_values, confg_['units'])

            qc_results = qc.run(
                    inp=eov_values,
                    tinp=grp_chunk["time"],
            )
            res_len = len(qc_results['qartod'].keys())
            a = 10
