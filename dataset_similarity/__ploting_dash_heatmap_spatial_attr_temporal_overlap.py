import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import pandas as pd

if __name__ == '__main__':
    attr_score_ = pd.read_csv("../res/df_attributal_matching_score.csv", index_col=0)
    spatial_score_ = pd.read_csv("../res/grid-ocean-lakes-20KM-EPSG-3857_df_spatial_matching_score.csv", index_col=0)
    temporal_score_ = pd.read_csv("../res/date_df_temporal_matching_score.csv", index_col=0)

    new_spatial_score_ = pd.DataFrame(columns=attr_score_.columns)
    col_row_to_remove = []
    rename_columns_ = {}
    for i, row in attr_score_.iterrows():
        up_i = i.replace(",", " ").replace(":", "").replace("/", " ").replace("|", "")
        up_i = up_i + ".csv"
        if up_i not in spatial_score_.index:
            col_row_to_remove.append(i)
        else:
            rename_columns_[i] = up_i

    attr_score_.drop(columns=col_row_to_remove, inplace=True)
    attr_score_.drop(col_row_to_remove, inplace=True)
    attr_score_.rename(columns=rename_columns_, inplace=True)
    attr_score_.rename(index=rename_columns_, inplace=True)


    attr_min__ = 0
    attr_max__ = max(attr_score_.max())
    spatial_min__ = 0
    spatial_max__ = max(spatial_score_.max())
    temporal_min__ = 0
    temporal_max__ = min(temporal_score_.max())


    # Initialize the Dash app
    app = dash.Dash(__name__)
    app.title = "Interactive Heatmap with Thresholds"

    # Define the layout
    app.layout = html.Div([
        html.H1("Interactive Heatmap with Threshold Filters", style={'textAlign': 'center'}),

        html.Div([
            html.Div([
                html.Label("Temporal Threshold:", style={'font-weight': 'bold'}),
                dcc.Slider(
                    id='temporal-threshold',
                    min=temporal_min__,
                    max=temporal_max__,
                    step=1,
                    value=40,  # Default value
                    marks={i: str(i) for i in range(0, 101, 10)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], style={'width': '45%', 'display': 'inline-block', 'padding': '0 20'}),

            html.Div([
                html.Label("Attributes Threshold:", style={'font-weight': 'bold'}),
                dcc.Slider(
                    id='attr-threshold',
                    min=attr_min__,
                    max=attr_max__,
                    step=1,
                    value=0,  # Default value
                    marks={i: str(i) for i in range(0, 101, 10)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], style={'width': '45%', 'display': 'inline-block', 'padding': '0 20'}),

            html.Div([
                html.Label("Spatial Threshold:", style={'font-weight': 'bold'}),
                dcc.Slider(
                    id='spatial-threshold',
                    min=spatial_min__,
                    max=spatial_max__,
                    step=1,
                    value=0,  # Default value
                    marks={i: str(i) for i in range(0, 101, 10)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], style={'width': '45%', 'display': 'inline-block', 'padding': '0 20'}),
        ], style={'padding': '40px'}),

        dcc.Graph(id='heatmap')
    ])


    # Define the callback to update the heatmap
    @app.callback(
        Output('heatmap', 'figure'),
        [Input('attr-threshold', 'value'),
         Input('spatial-threshold', 'value'),
         Input('temporal-threshold', 'value')]
    )


    def update_heatmap(attr_threshold, spatial_threshold, temporal_threshold_):
        # Create a mask based on the thresholds
        attr_score_df_, spatial_score_df_, temporal_score_df_ = attr_score_.copy(), spatial_score_.copy(), temporal_score_.copy()
        mask1 = attr_score_df_ >= attr_threshold
        mask2 = spatial_score_df_ >= spatial_threshold
        mask3 = temporal_score_df_ >= temporal_threshold_
        # print(f"Attr: {attr_threshold}  - Spatial: {spatial_threshold} - Temp: {temporal_threshold_}")
        combined_mask = mask1 & mask2 & mask3
        default_value  = -1
        attr_score_df_.where(combined_mask, default_value, inplace=True)
        spatial_score_df_.where(combined_mask, default_value,  inplace=True)
        temporal_score_df_.where(combined_mask, default_value, inplace=True)

        #validation
        flag_ = False
        for col in attr_score_df_.columns:
            for idx in attr_score_df_.index.values:
                if (attr_score_df_.loc[idx][col] != default_value) and (attr_score_df_.loc[idx][col] < attr_threshold):
                    flag_ = True
                    print(f"ATTR [{attr_score_df_.loc[idx][col]}  < {attr_threshold}]")
                    break
                if (spatial_score_df_.loc[idx][col] != default_value) and (spatial_score_df_.loc[idx][col] < spatial_threshold):
                    print(f"SPAT [{spatial_score_df_.loc[idx][col]}  < {spatial_threshold}]")
                    flag_ = True
                    break
                if (temporal_score_df_.loc[idx][col] != default_value) and (temporal_score_df_.loc[idx][col] < temporal_threshold_):
                    print(f"TEMP [{temporal_score_df_.loc[idx][col]} < {temporal_threshold_}]")
                    flag_ = True
                    break
            if flag_:
                break
        if flag_:
            print("Something is wrong.....!")


        colors = ['Greens', 'Blues', 'Oranges']
        # colors = ['Purp', 'Oryel','Reds']
        # Create the heatmap
        heatmap1 = go.Heatmap(
            z=attr_score_df_.values,
            x=list(attr_score_df_.columns),
            y=list(attr_score_df_.index),
            opacity=0.2,
            colorscale=colors[0],
            zmax=150, zauto=False, zmin=0,
            colorbar=dict(title='Attribute', orientation='h', len=0.33, x=0.1),
            showscale=True,
            hovertemplate=(
                "Row: %{y}<br>"  # Display y-coordinate as row index
                "Col: %{x}<br>"  # Display x-coordinate as column index
                "Spatial: %{customdata[0]}<br>"  # Value from df1 based on x,y
                "Temp: %{customdata[2]}<br>"  # Value from df3 based on x,y
                "Attr: %{z}<br>"  # Value from df2 based on x,y
                "<extra></extra>"  # Removes extra trace info from hover
            ),
            customdata=np.dstack((spatial_score_df_, attr_score_df_, temporal_score_df_)),

        )
        heatmap2 = go.Heatmap(
            z=spatial_score_df_.values,
            x=list(spatial_score_df_.columns),
            y=list(spatial_score_df_.index),
            colorscale=colors[1],
            opacity=0.2,zmax=150, zauto=False, zmin=0,
            colorbar=dict(
                title='Spatial',
                x=0.8,  # Position at the bottom center
                orientation='h', len=0.33
            ),
            showscale=True
        )

        # Create the heatmap
        heatmap3 = go.Heatmap(
            z=temporal_score_df_.values,
            x=list(temporal_score_df_.columns),
            y=list(temporal_score_df_.index),
            opacity=0.2,zmax=150, zauto=False, zmin=0,
            colorscale=colors[2],
            colorbar=dict(
                title='Temporal',
                titlefont=dict(size=12),
                tickfont=dict(size=10),
                x=0.46,  # Position at the bottom center
                orientation='h', len=0.33
            ),
            showscale=True
        )

        fig = go.Figure(data=[heatmap3, heatmap2, heatmap1])
        fig.update_layout(
            width=1800,  # Increase width
            height=1800,  # Increase height
            xaxis=dict(tickangle=90, tickfont=dict(size=4)),
            yaxis=dict(tickfont=dict(size=4)),
            title=dict(font=dict(size=20)),
            font=dict(size=10)

        )
        return fig


    app.run_server(debug=True)