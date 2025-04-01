import pickle

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
import os
from tqdm import tqdm
import pandas as pd
import plotly.express as px


'''
Calculating spatial matching score between datasets. the spatial window of 20KM is set.

Input: 'grid-ocean-lakes-20KM-EPSG-3857.shp' contains lakes map, generated from QGIS

Output: '_df_spatial_matching_score.csv' and _d2d_spatial_similarity.html
'''
if __name__ == '__main__':
    # this .shp file is created from QGIS which contains map of all lakes
    grid_filename = "/res/grid-ocean-lakes-20KM-EPSG-3857.shp"
    gdf_filename = grid_filename.replace(".shp","_gdf.pkl")
    ckdtree_filename = grid_filename.replace(".shp","_ckdtree.pkl")
    dataset_grid_dict_filename = grid_filename.replace(".shp", "_das_gridcenter_dict.pkl")

    if not os.path.exists(gdf_filename):
        gdf_ = gpd.read_file(grid_filename)
        gdf_['centroid'] = gdf_.geometry.centroid
        target_crs = 'EPSG:4269'
        centroids_gdf = gdf_[['centroid']].copy()
        centroids_gdf.set_geometry('centroid', inplace=True)
        gdf_['centroid'] = centroids_gdf.to_crs(target_crs)
        gdf_['cen_x'] = gdf_['centroid'].x
        gdf_['cen_y'] = gdf_['centroid'].y

        nB = np.array(list(gdf_['centroid'].geometry.apply(lambda x: (x.x, x.y))))
        btree = cKDTree(nB)
        pickle.dump(btree, open(ckdtree_filename, 'wb'))
        pickle.dump(gdf_, open(gdf_filename, 'wb'))
    else:
        print("Loading GDF and ")
        gdf_ = pickle.load(open(gdf_filename, 'rb'))
        btree = pickle.load(open(ckdtree_filename, 'rb'))

    dataset_grids_map = {} #dataset_idx = [ grid_centroids,.. ]
    if not os.path.exists(dataset_grid_dict_filename):
        for root, dir, files in os.walk("../erddap data/"):
            pbar_ = tqdm(files)
            for fname in pbar_:
                pbar_.set_description(fname)
                if fname.lower().endswith(".csv"):
                    fl_path_ = os.path.join(root, fname)
                    df = pd.read_csv(fl_path_).dropna()
                    if 'latitude' not in df.columns:
                        print(f"Skipping [{fname}]")
                        continue

                    points = set(zip(df['latitude'].iloc[1:].astype('float64'), df['longitude'].iloc[1:].astype('float64')))
                    points_ndarray = np.array(list(points))
                    dist, idx = btree.query(points_ndarray, k=1)
                    new_points = btree.data[idx]
                    new_points_df_ = pd.DataFrame(new_points, columns=['cen_x', 'cen_y'])
                    merged_df = gdf_.merge(new_points_df_, on=['cen_x', 'cen_y'], how='inner')
                    matched_ids = set([int(x) for x in merged_df['id'].tolist()])
                    dataset_grids_map[fname] = matched_ids

        pickle.dump(dataset_grids_map, open(dataset_grid_dict_filename, 'wb'))

    else:
        dataset_grids_map = pickle.load(open(dataset_grid_dict_filename, 'rb'))


    list_of_das_variables_ = []
    item_labels = []
    for d_name, grids_ in dataset_grids_map.items():
        item_labels.append(d_name)
        list_of_das_variables_.append(grids_)

    num_items = len(list_of_das_variables_)
    # Initialize a 50x50 matrix with zeros
    matching_matrix = np.zeros((num_items, num_items), dtype=int)

    lod = list_of_das_variables_
    # Calculate matching scores
    for i in tqdm(range(num_items)):
        for j in range(num_items):
            total_ = len(lod[i])
            match_count = len(set(lod[i]).intersection(set(lod[j])))
            match_count = int(float(match_count / total_) * 100)
            # if match_count != 100:
            #     match_count = 0
            matching_matrix[i][j] = match_count

    df_matching = pd.DataFrame(matching_matrix, index=item_labels, columns=item_labels)
    df_matching.to_csv(f"./res/{os.path.basename(grid_filename).replace('.shp','_df_spatial_matching_score.csv')}")
    # Create a Seaborn heatmap
    # Set up the plot
    fig = px.imshow(
        df_matching,
        labels=dict(x="Items", y="Items", color="Matching Score"),
        x=item_labels,
        y=item_labels,
        color_continuous_scale='Oranges',
        title='Interactive Matching Score Heatmap'
    )
    fig.update_traces(
        hovertemplate='Item X: %{x}<br>Item Y: %{y}<br>Matching Score: %{z}<extra></extra>'
    )
    # Update layout for better readability
    # Adjust layout for better readability
    fig.update_layout(
        width=1800,  # Increase width
        height=1800,  # Increase height
        xaxis=dict(tickangle=90, tickfont=dict(size=4)),
        yaxis=dict(tickfont=dict(size=4)),
        title=dict(font=dict(size=20)),
        font=dict(size=10)
    )
    basename_ = os.path.basename(grid_filename).replace(".shp","_d2d_spatial_similarity.html")
    # Show the plot
    fig.write_html(f"./res/{basename_}")
