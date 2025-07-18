
import pandas as pd
from datetime import timedelta
import numpy as np
import os
from sklearn.cluster import DBSCAN
import traceback

from qa_qc.qaqc_test_rolling_std import apply_rolling_sd_with_time_reset


class QartodFlags:
    """Primary flags for QARTOD."""
    GOOD = 1
    UNKNOWN = 2
    SUSPECT = 3
    FAIL = 4
    MISSING = 9


def get_file_names(dir__, ext_ = ".csv"):
    file_names = []
    for root_, dr_, files in os.walk(dir__):
        for fl in files:
            if fl.lower().endswith(ext_):
                file_names.append(os.path.join(root_, fl))
    return file_names
def generate_time_windows(df, time_col='time', window_hours=2, min_rows_in_chunk=10, filter_flag = None, filter_flag_col = None):
    """
    Rolling over dataframe, generating defined slices based on time-window with step-size = 1
    :param df: Dataframe
    :param time_col:   Timestamp column name
    :param window_hours:  window size in hours
    :param min_rows_in_chunk:  minimum rows in a chunk/slice/sequence
    :return: list of Tuple( current row, past_time-window-dataframe, previous_time-window-dataframe)
    """
    df = df.dropna(axis=0).sort_values(by=time_col).reset_index(drop=True)
    df['time_diff_sec'] = df['time'].diff().dt.total_seconds()
    time_delta = timedelta(hours=window_hours, minutes=10)
    windowed_samples = []
    minimum_seconds_ = window_hours * 60* 59

    for i in range(1, len(df)):
        current_row = df.iloc[i]
        if (filter_flag is not None) and (filter_flag_col is not None):
            if current_row[filter_flag_col] != filter_flag:
                continue

        # OPTIMIZED ###########
        future_stop_, past_start_ = i + 1, i
        time_sum_past, time_sum_ = 0, 0
        past_stop, future_stop = False, False
        past_not_satisfied, future_not_satisfied = False, False
        while True:
            if not (time_sum_ > minimum_seconds_):
                if future_stop_ >= len(df):
                    future_not_satisfied = True
                else:
                    next_row = df.iloc[future_stop_]
                    next_row_time_diff = next_row['time_diff_sec']
                    time_sum_ = time_sum_ + next_row_time_diff
                    future_stop_ += 1
            else:
                future_stop = True

            if not (time_sum_past > minimum_seconds_):
                if past_start_ < 0:
                    past_not_satisfied = True
                else:
                    prev_row_time_diff = current_row['time_diff_sec']
                    time_sum_past = time_sum_past + prev_row_time_diff
                    past_start_ -= 1
            else:
                past_stop = True

            if future_not_satisfied or past_not_satisfied:
                break
            if future_stop and past_stop:
                break

        if (not past_not_satisfied) and (not future_not_satisfied):
            past_slot_df = df.iloc[past_start_:i]
            future_slot_df = df.iloc[i + 1: future_stop_]
            if (len(past_slot_df) >= min_rows_in_chunk) and ( len(future_slot_df) >= min_rows_in_chunk):
                tpl_ = (current_row, past_slot_df.drop_duplicates().reset_index(drop=True),
                        future_slot_df.drop_duplicates().reset_index(drop=True))
                windowed_samples.append(tpl_)


    return windowed_samples


def get_xy_from_dataframe(df,eov_flag_name,eov_col_name, window_hours, min_rows_in_chunk=6, ignore_labels = [QartodFlags.UNKNOWN, -1], dataset_name_string = None):
    dt_ = df['time'].dt
    df['yr'] = dt_.year
    df['mon'] = dt_.month
    year_month_stats_map = {}  # (year, month) stats
    group_by = df.groupby(by=['yr', 'mon'])
    for grp, chunk in group_by:
        values_ = chunk[eov_col_name]
        q_997 = values_.quantile(0.997)
        q_003 = values_.quantile(0.003)
        q_mean = values_.mean()
        q_std = values_.std()
        new_chunk = chunk.copy()
        new_chunk = new_chunk.set_index('time')
        df_hourly = new_chunk[eov_col_name].resample('1h').mean()
        avg_hourly_change = abs(df_hourly.diff()).mean()
        year_month_stats_map[grp] = (q_997, q_003, q_mean, q_std, avg_hourly_change)
    df['group'] = 1
    df = apply_rolling_sd_with_time_reset(
        df,
        time_col='time',
        value_col=eov_col_name,
        group_col='group',
        time_gap_threshold_min=2 * 60,
        rolling_window_minutes= 2 * window_hours * 60
    )
    df.drop(columns=['group'], inplace=True)
    lst_of_seq_ = generate_time_windows(df, window_hours=window_hours, min_rows_in_chunk=min_rows_in_chunk)
    # Feature engineering from window
    X, y = [], []
    for current_, past_, future_ in lst_of_seq_:
        label = current_[eov_flag_name]
        current_value = current_[eov_col_name]
        current_rolling = current_[eov_col_name+"_rolling_sd"]

        if label in ignore_labels:
            continue
        past_window = past_[eov_col_name]
        future_window = future_[eov_col_name]
        if label == QartodFlags.GOOD: #ignoring bad samples in past
            Y = past_[eov_flag_name]
            mask = ~((Y == QartodFlags.SUSPECT) | (Y == QartodFlags.FAIL))
            past_window = past_window[mask]


        if (past_window.shape[0] < min_rows_in_chunk) or (future_window.shape[0] < min_rows_in_chunk):
            continue


        # past_curent_window = pd.concat([past_window, pd.Series([current_value])])
        # future_cuurent_window = pd.concat([pd.Series([current_value]), future_window])
        full_window = pd.concat([past_window, pd.Series([current_value]), future_window])

        q_997, q_003, q_mean, q_std, avg_hourly_change = year_month_stats_map[
            (current_['time'].year, current_['time'].month)]
        q_997 = abs(current_value - q_997) if current_value > q_997 else 0
        q_003 = abs(q_003 - current_value) if current_value < q_003 else 0

        fwq_997 = full_window.quantile(0.997)
        fwq_003 = full_window.quantile(0.003)
        fwq_997 = abs(current_value - fwq_997) if current_value > fwq_997 else 0
        fwq_003 = abs(fwq_003 - current_value) if current_value < fwq_003 else 0

        # if label == QartodFlags.SUSPECT:
        #     a = 10
        spike_ref_ = ((past_window.tail(1).values[0] + future_window.head(1).values[0])/2)

        month_feature = np.eye(12, dtype=int)[current_['time'].month - 1]
        month_feature[current_['time'].month - 1] = 1
        past_mean = past_window.mean()
        future_mean = future_window.mean()
        features = np.array([
            current_rolling,
            abs(past_mean - current_value),
            abs(future_mean - current_value),
            abs(future_mean - past_mean),
            abs( current_value - spike_ref_ ),
            abs(avg_hourly_change - abs(past_window.tail(1).values[0] - current_value)),
            abs(avg_hourly_change - abs(future_window.head(1).values[0] - current_value)),
            q_997,
            q_003,
            fwq_997,
            fwq_003,
            *month_feature,
            label
        ])

        # if np.sum(np.isnan(np.array(features))) > 0:
        #     a =10
        X.append(features)

    X = np.vstack(X)
    X = np.round(X, 3)
    mask = X[:, -1] == QartodFlags.SUSPECT
    suspect_row = X[mask]
    mask_f = X[:, -1] == QartodFlags.FAIL
    fail_rows = X[mask_f]
    mask_g = X[:, -1] == QartodFlags.GOOD
    good_rows = X[mask_g]
    unique_unique_rows = np.unique(good_rows, axis=0)
    print(f"[{dataset_name_string}] \n Unique Extraction:  {good_rows.shape}  :-> {unique_unique_rows.shape}  + {fail_rows.shape} + {suspect_row.shape}")
    X = np.concatenate([suspect_row, good_rows, fail_rows])
    X = X.astype(np.float32)
    X = np.round(X, 3)
    return X


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


def normalize_column_inplace(data: np.ndarray, column_index: int, value_range: tuple):
    """
    Normalize a specific column in a 2D NumPy array in-place using a given (min, max) range.

    Parameters:
    - data: 2D numpy array (will be modified in-place)
    - column_index: index of the column to normalize
    - value_range: (min, max) tuple representing the known range of that column
    """
    min_val, max_val = value_range
    if max_val == min_val:
        raise ValueError("Provided min and max are equal; cannot normalize.")

    # In-place normalization
    data[:, column_index] -= min_val
    data[:, column_index] /= (max_val - min_val)