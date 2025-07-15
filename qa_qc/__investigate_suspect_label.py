from qa_qc.ai_utils import get_file_names
import pandas as pd
import os
from qa_qc.ai_utils import generate_time_windows, QartodFlags
from tqdm import tqdm

if __name__ == '__main__':
    parent_ = "CIOOS-Full-Data/chunking/"
    chunk_dir = os.path.join(parent_,"Inverness/")
    eov_col_name = 'temperature'
    eov_flag_name = 'qc_flag_temperature'
    window_hour = 12
    file_names = get_file_names(chunk_dir)
    suspect_df = []
    for file_name in tqdm(file_names):
        print(f"Processing : [{file_name}]")
        df = pd.read_csv(file_name, usecols=['time', eov_flag_name, eov_col_name])
        df['time'] = pd.to_datetime(df['time'])

        # Feature engineering from window
        lst_of_seq_ = generate_time_windows(df, window_hours=window_hour, min_rows_in_chunk=10)
        for current_, past_, future_ in lst_of_seq_:
            label = current_[eov_flag_name]
            if label == QartodFlags.SUSPECT:
                su_df = pd.concat([past_.reset_index(drop=True), current_.to_frame().T.reset_index(drop=True), future_.reset_index(drop=True)])
                suspect_df.append(su_df)


    df__ = pd.concat(suspect_df, axis=0)
    df__.drop_duplicates(inplace=True)
    df__.to_csv(os.path.join(parent_, f"{os.path.basename(os.path.dirname(chunk_dir))}__SUSPECT.csv"), index=False)

    # dir_ = "D:/CIOOS-Full-Data/chunking/"
    # for rt, dir_, files in os.walk(dir_):
    #     for fl in files:
    #         if "SUSPECT.csv" in fl:
    #             fapath_ = os.path.join(rt, fl)
    #             df_ = pd.read_csv(fapath_)
    #             print(df_.shape)
    #             df_.drop_duplicates(inplace=True)
    #             fapath_ = fapath_.replace(".csv","_noduplicate.csv")
    #             df_.to_csv(fapath_, index=False)