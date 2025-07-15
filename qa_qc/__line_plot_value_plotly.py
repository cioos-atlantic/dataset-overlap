import plotly.express as px
import pandas as pd
import os
import re
from ai_utils import get_file_names
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if __name__ == '__main__':
    dir_ = "D:/CIOOS-Full-Data/chunking/Antigonish/"
    fnames_ = get_file_names(dir_)
    # fnames_ = ["Cape Breton County Water Quality Data.csv-1.csv", "Cape Breton County Water Quality Data.csv-2.csv"]
    eov_col_name = 'temperature'
    eov_flag_name = 'qc_flag_temperature'
    lst_of_dfs = []
    map__ = {}
    for i, fname in enumerate(fnames_):
        fname = os.path.basename(fname)
        fname = os.path.join(dir_, fname)
        df = pd.read_csv(fname, usecols=['time', eov_flag_name, eov_col_name])
        df['time'] = pd.to_datetime(df['time'])
        df['ID'] = i
        map__[i] = fname
        lst_of_dfs.append(df)

    new_df_ = pd.concat(lst_of_dfs)

    new_df_.sort_values(by=['ID', 'time'], inplace=True)

    # Unique IDs determine number of plots
    unique_ids = sorted(new_df_['ID'].unique())
    num_plots = len(unique_ids)

    # Create subplot layout
    fig = make_subplots(
        rows=num_plots, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.002,
        subplot_titles=[f"ID: {map__[id_]}" for id_ in unique_ids]
    )

    # Add single line trace for each ID
    for idx, id_ in enumerate(unique_ids):
        sub_df = new_df_[new_df_['ID'] == id_]
        fig.add_trace(
            go.Scatter(
                x=sub_df['time'],
                y=sub_df[eov_col_name],
                mode='lines',
                name=f'ID {id_}',
                showlegend=False
            ),
            row=idx + 1, col=1
        )

    # Set overall figure layout
    fig.update_layout(
        height=400 * num_plots,
        title="Scrollable Subplots: Temperature by ID"
    )

    # Save scrollable HTML
    output_path = os.path.join(dir_, "visualize_scrollable_subplots.html")
    fig.write_html(output_path)

    print(f"Saved scrollable subplot HTML to: {output_path}")
