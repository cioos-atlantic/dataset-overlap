import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
import datetime as dt
import matplotlib.pyplot as plt


def apply_rolling_sd_with_time_reset(
        df,
        time_col='time',
        value_col='value',
        group_col='group',
        time_gap_threshold_min=15,
        rolling_window_minutes=60
):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    min_wind = int(rolling_window_minutes / time_gap_threshold_min)
    # Sort by group and time
    df.sort_values([time_col], inplace=True)

    results = []

    # Process each group separately
    for _, group_df in df.groupby(group_col):
        group_df = group_df.copy()

        # Calculate time difference in minutes
        group_df['time_diff_min'] = group_df[time_col].diff().dt.total_seconds() / 60

        # Start a new "segment" whenever time gap is too big
        group_df['segment'] = (group_df['time_diff_min'] > time_gap_threshold_min).cumsum()

        # Within each segment, apply rolling std
        def compute_rolling(g):
            g = g.set_index(time_col).sort_index()
            rolling_window = f'{rolling_window_minutes}min'
            g[f'{value_col}_rolling_sd'] = (
                g[value_col]
                .rolling(rolling_window, center=True, min_periods=min_wind)
                .std()
            )
            g = g.reset_index()
            start_ = g.iloc[0][time_col_name]
            for j in range(1, len(g[time_col_name])):
                prev_point_ = g.iloc[j][time_col_name]
                mints = (prev_point_ - start_).seconds / 60
                if not (mints < (rolling_window_minutes/2)):
                    g.loc[0:j-1 , f'{value_col}_rolling_sd'] = np.nan
                    break
            start_ = g.iloc[-1][time_col_name]
            for j in range(len(g[time_col_name])-1, 0, -1):
                prev_point_ = g.iloc[j][time_col_name]
                mints = (start_ - prev_point_).seconds / 60
                if not (mints < (rolling_window_minutes / 2)):
                    g.loc[j+1: , f'{value_col}_rolling_sd'] = np.nan
                    break
            return g

        rolled = group_df.groupby('segment', group_keys=False).apply(compute_rolling)
        results.append(rolled)

    # Combine all grouped results
    final_df = pd.concat(results, ignore_index=True).drop(columns=['time_diff_min', 'segment'])
    return final_df


if __name__ == '__main__':
    period_hours = 24
    max_interval_hours = 2 # haven't applied yet

    eov_col_name = 'dissolved_oxygen_percent_saturation'
    time_col_name = 'timestamp_utc'
    std_col_name = eov_col_name + '_rolling_sd'
    group_by_col = ['county', 'station', 'deployment_range', 'sensor_serial_number']
    qt_df_ = pd.read_csv("2024-10-24_cmar_water_quality_thresholds.csv")
    sample_df_ = pd.read_csv("sample_rolling_sd_data.csv")



    retu_ = apply_rolling_sd_with_time_reset(
        sample_df_,
        time_col=time_col_name,
        value_col=eov_col_name,
        group_col=group_by_col,
        time_gap_threshold_min=2 * 60,
        rolling_window_minutes=24 * 60
    )
    conditions = [
                retu_[std_col_name].isna(),
                retu_[std_col_name] > 2.99
    ]
    choices = [9, 2]
    default_val = 1
    retu_['dissolve_oxygen_flag'] = np.select( conditions, choices, default=default_val)
    category_colors = {2: 'red', 9: 'blue', 1: 'green'}
    colors = retu_['dissolve_oxygen_flag'].map(category_colors)
    plt.figure(figsize=(6, 4))
    plt.scatter( [i for i in range(len(retu_['dissolved_oxygen_percent_saturation']))] , retu_['dissolved_oxygen_percent_saturation'], c=colors)

    for cat in category_colors:
        plt.scatter([], [], c=category_colors[cat], label=cat)
    # Optional: add legend
    plt.legend(title='Category')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot Colored by Category")
    plt.show()
    a = 10
