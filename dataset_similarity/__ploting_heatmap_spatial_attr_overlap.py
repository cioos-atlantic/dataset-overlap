import pandas as pd
import plotly.graph_objects as go
import jellyfish as jf

if __name__ == '__main__':
    attr_score_ = pd.read_csv("../res/df_attributal_matching_score.csv", index_col=0)
    spatial_score_ = pd.read_csv("../res/grid-ocean-lakes-20KM-EPSG-3857_df_spatial_matching_score.csv", index_col=0)


    new_spatial_score_ = pd.DataFrame(columns=attr_score_.columns)
    col_row_to_remove = []
    rename_columns_ = {}
    for i, row in attr_score_.iterrows():
        up_i = i.replace(","," ").replace(":","").replace("/"," ").replace("|","")
        up_i = up_i+".csv"
        if up_i not in spatial_score_.index:
            col_row_to_remove.append(i)
        else:
            rename_columns_[i] = up_i

    attr_score_.drop(columns=col_row_to_remove, inplace=True)
    attr_score_.drop(col_row_to_remove, inplace=True)
    attr_score_.rename(columns=rename_columns_, inplace=True)
    attr_score_.rename(index=rename_columns_, inplace=True)

    # Initialize the figure
    fig = go.Figure()

    # cs_light_green = [[0.0, '#ccffcc'],  [1.0, '#006600'] ]
    # cs_light_yellow = [ [0.0, '#ffffcc'], [1.0, '#ff9900']]
    cs_light_green = 'Purp'
    cs_light_yellow = 'Oryel'
    # Add the first heatmap for Score 1
    fig.add_trace(
        go.Heatmap(
            z=attr_score_.values,
            x=list(attr_score_.columns),
            y=list(attr_score_.index),
            colorscale=cs_light_green,
            opacity=0.3,
            name='Attributes',
            colorbar=dict(title='Attributes'),
            showscale=True
        )
    )


    # Add the second heatmap for Score 2
    fig.add_trace(
        go.Heatmap(
            z=spatial_score_.values,
            x=list(spatial_score_.columns),
            y=list(spatial_score_.index),
            colorscale=cs_light_yellow,
            opacity=0.3,
            name='Spatial',
            showscale=True,
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
            )
        )
    )

    # Update layout for better visualization
    fig.update_layout(
        title='Overlayed Heatmaps of Score 1 and Score 2',
        xaxis_title='Attributes',
        yaxis_title='Attributes',
        # Adjust the legend if needed
        legend=dict(
            title='Scores',
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        # Improve the appearance
        width=700,
        height=700
    )
    fig.update_layout(
        width=1800,  # Increase width
        height=1800,  # Increase height
        xaxis=dict(tickangle=90, tickfont=dict(size=4)),
        yaxis=dict(tickfont=dict(size=4)),
        title=dict(font=dict(size=20)),
        font=dict(size=10)
    )

    fig.write_html(f"./res/d2s_combined_similarity.html")
