import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import pandas as pd

if __name__ == '__main__':
    attr_score_ = pd.read_csv("./res/df_attributal_matching_score.csv", index_col=0)
    spatial_score_ = pd.read_csv("./res/grid-ocean-lakes-20KM-EPSG-3857_df_spatial_matching_score.csv", index_col=0)
    temporal_score_ = pd.read_csv("./res/date_df_temporal_matching_score.csv")

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

    attr_min__ = min(attr_score_.min())
    attr_max__ = max(attr_score_.max())
    spatial_min__ = min(spatial_score_.min())
    spatial_max__ = max(spatial_score_.max())

    # Initialize the Dash app
    app = dash.Dash(__name__)
    app.title = "Interactive Heatmap with Thresholds"

    # Define the layout
    app.layout = html.Div([
        html.H1("Interactive Heatmap with Threshold Filters", style={'textAlign': 'center'}),

        html.Div([
            html.Div([
                html.Label("Attributes Threshold:", style={'font-weight': 'bold'}),
                dcc.Slider(
                    id='score1-threshold',
                    min=attr_min__,
                    max=attr_max__,
                    step=1,
                    value=50,  # Default value
                    marks={i: str(i) for i in range(0, 101, 10)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], style={'width': '45%', 'display': 'inline-block', 'padding': '0 20'}),

            html.Div([
                html.Label("Spatial Threshold:", style={'font-weight': 'bold'}),
                dcc.Slider(
                    id='score2-threshold',
                    min=spatial_min__,
                    max=spatial_max__,
                    step=1,
                    value=50,  # Default value
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
        [Input('score1-threshold', 'value'),
         Input('score2-threshold', 'value')]
    )


    def update_heatmap(score1_threshold, score2_threshold):
        # Create a mask based on the thresholds
        mask1 = attr_score_ >= score1_threshold
        mask2 = spatial_score_ >= score2_threshold
        combined_mask = mask1 & mask2

        # Apply the mask: set cells that do not meet the criteria to None
        z1 = attr_score_.copy().astype(float)
        z1[~combined_mask] = None  # Cells not meeting both criteria will be transparent

        z2 = spatial_score_.copy().astype(float)
        z2[~combined_mask] = None

        # Create the heatmap
        heatmap1 = go.Heatmap(
            z=z1.values,
            x=list(z1.columns),
            y=list(z1.index),
            opacity=0.3,
            colorscale='Purp',
            colorbar=dict(title='Score 1'),
            hovertemplate=(
                '<b>X: %{x}</b><br>'
                '<b>Y: %{y}</b><br>'
                '<b>Score 1:</b> %{z}<br>'
                '<extra></extra>'
            ),
            showscale=True
        )
        heatmap2 = go.Heatmap(
            z=z2.values,
            x=list(z2.columns),
            y=list(z2.index),
            colorscale='Oryel',
            customdata=attr_score_.values,
            colorbar=dict(
                title='Spatial',
                titleside='bottom',
                titlefont=dict(size=12),
                tickfont=dict(size=10),
                len=0.7,
                x=0.5,  # Position at the bottom center
                y=-0.1,  # Slightly below the heatmap
                orientation='h'
            ),
            hovertemplate=(
                '<b>X: %{x}</b><br>'
                '<b>Y: %{y}</b><br>'
                '<b>Attributes :</b> %{customdata:.2f}<br>'
                '<b>Spatial 2:</b> %{z:.2f}<br>'
                '<extra></extra>'
            ),
            showscale=True
        )
        fig = go.Figure(data=[heatmap1, heatmap2])
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