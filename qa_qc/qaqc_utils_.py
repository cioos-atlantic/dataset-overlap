import pickle
import codecs
import json
import unicodedata
from pyvis.network import Network
from community import community_louvain
import networkx as nx
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd


erdap_dict_ = pickle.load(open("../res/alldataset_das_dict_NEW.pkl", 'rb'))
file_graph_gen_ = "../res/graph_variable_instrument_relation_4_GEN.pkl"
mjg_filepath = "../res/graph_variable_instrument_relation_mannual_joining.pkl"
gen_graph_ = pickle.load(open(file_graph_gen_, 'rb'))
man_graph = pickle.load(open(mjg_filepath, 'rb'))
nomenc_ = json.load(open("nomenclature.json",'r', encoding='utf-8'))
unit_json = json.load(open("unit_2_unit.json", 'r', encoding='utf-8'))
qaqc_conf = json.load(open("ocean_qa_qc_config_v4.json",'r', encoding='utf-8'))

test_order_ = ["Deepest Pressure Test",
               "Platform Identification",
               "Impossible Date Test",
               "Impossible Location Test",
               "Position on Land Test",
               "Impossible Speed Test",
               "Global Range Test",
               "Regional Range Test",
               "Pressure Increasing Test",
               "Spike Test",
               "Top and Bottom Spike Test : removed",
               "Gradient Test",
               "Digit Rollover Test",
               "Stuck Value Test",
               "Density Inversion",
               "Grey List",
               "Gross salinity or temperature sensor drift",
               "Frozen profile",
               "Visual QC"
               ]

def normalize(text):
    return unicodedata.normalize('NFC', text)

def codec_normalize(text):
    return codecs.decode(text, 'unicode_escape')
def get_node_unit_association():
    unit_asso_dict_ = pickle.load(open("../res/unit_association_dict.pkl", 'rb'))
    unit_asso_dict_ = { k.lower():v for k, v in unit_asso_dict_.items()}
    unit_asso_inverse_ = {}
    for k, v in unit_asso_dict_.items():
        for nd in v:
            if nd in unit_asso_inverse_: # if more than 1 unit is associated with a field name
                t_ = unit_asso_inverse_[nd]
                if isinstance(t_, str):
                    unit_asso_inverse_[nd] = [t_]
                unit_asso_inverse_[nd.lower()].append(k.lower())
            else:
                unit_asso_inverse_[nd.lower()] = k.lower()

    return unit_asso_inverse_, unit_asso_dict_


def dict_inv(json_dict):
    inv_dict = {}

    def extract_elements(lst):
        if isinstance(lst, str):
            return [lst]
        top_list = []
        for it in lst:
            if isinstance(it, list):
                top_list.extend(extract_elements([el.lower() for el in it]))
            else:
                # it = codecs.decode(it, 'unicode_escape')
                top_list.append(it.lower())
        return top_list

    for k, v in json_dict.items():
        elements = extract_elements(v)
        for el in elements:
            inv_dict[el] = k
    return inv_dict



def find_unit_dataset_name(unit_name, node_name=None):
    def handle_dict(dict_obj, unit_association_ = [], parent=None, repo = None):
        for k, v in dict_obj.items():
            if isinstance(v, dict):
                p_arg = k if (k != 'attr') else parent
                unit_association_ = handle_dict(v, unit_association_, parent=p_arg, repo=repo)

            elif isinstance(v, tuple):
                if (k.strip() == 'units') and (v[1].lower() == unit_name):
                    if node_name is None:
                        unit_association_.add(repo['NC_GLOBAL']['title'][1])
                    else:
                        if parent.lower() == node_name:
                            unit_association_.add(repo['NC_GLOBAL']['title'][1])
                        elif ("standard_name" in dict_obj) and dict_obj["standard_name"][1].lower() == node_name:
                            unit_association_.add(repo['NC_GLOBAL']['title'][1])
                        elif ("long_name" in dict_obj) and (dict_obj["long_name"][1].lower() == node_name):
                            unit_association_.add(repo['NC_GLOBAL']['title'][1])
        return set(unit_association_)

    unit_association_ = []
    for dataset_ in erdap_dict_:
        unit_association_.extend(handle_dict(dataset_, repo=dataset_))
    return unit_association_


def find_unit_variable_name(unit_name, return_other_names = False):
    def handle_dict(dict_obj, unit_association_ = [], parent=None, repo = None):

        for k, v in dict_obj.items():
            if isinstance(v, dict):
                p_arg = k if (k != 'attr') else parent
                unit_association_ = handle_dict(v, unit_association_, parent=p_arg, repo=repo)

            elif isinstance(v, tuple):
                if (k.strip() == 'units'):
                    if (v[1].lower() == unit_name) and ("actual_range" in dict_obj):
                        unit_association_.append(parent.lower())

        return unit_association_

    unit_association_ = []
    for dataset_ in erdap_dict_:
        unit_association_.extend(handle_dict(dataset_, repo=dataset_))

    return unit_association_

def ext_keys_val_list(dict__):
    l_ = []
    if isinstance(dict__, dict):
        l_.extend(ext_keys_val_list(list(dict__.keys())))
        for vl in dict__.values():
            l_.extend(ext_keys_val_list(vl))
    elif isinstance(dict__, tuple) or isinstance(dict__, list):
        for el in dict__:
            l_.extend(ext_keys_val_list(el))
    else:
        l_.append(codec_normalize(dict__).lower())

    return l_

def find_node_s_dataset_name(node_name, return_dict_obj=False):
    dsts_ = []


    for dataset_ in erdap_dict_:
        keywords = set(ext_keys_val_list(dataset_))

        if codec_normalize(node_name.lower()) in keywords:
            if return_dict_obj:
                dsts_.append((dataset_['NC_GLOBAL']['title'][1], dataset_) )
            else:
                dsts_.append(dataset_['NC_GLOBAL']['title'][1])
        # if codec_normalize(node_name.lower()) in codec_normalize(str(dataset_)).lower():
        #     dsts_.append( dataset_['NC_GLOBAL']['title'][1] )

    if dsts_.__len__() == 0:
        return False
    return dsts_



def create_pyviz_network_graph(g_, output_filename):
    net = Network(notebook=True, width='3000px', height='3000px', bgcolor='#222222', font_color='white')
    node_degree = dict(g_.degree)
    node_degree = {k: (v + 1) + 1 for k, v in node_degree.items()}

    communities = community_louvain.best_partition(g_)

    nx.set_node_attributes(g_, node_degree, 'size')
    nx.set_node_attributes(g_, communities, 'group')
    net.from_nx(g_)
    net.show_buttons(filter_=True)
    net.show(output_filename)

    print("..Done..")


def get_cat(nod_unit_, u2c_):
    if isinstance(nod_unit_, list):
        return [u2c_[it] for it in nod_unit_]
    else:
        return [u2c_[nod_unit_]]



def cluster_geodetic_points(
        df: pd.DataFrame,
        lon_col: str = "longitude",
        lat_col: str = "latitude",
        max_distance_km: float = 2.0
) -> pd.DataFrame:
    """
    Groups geographic points (longitude, latitude) so that every point in a cluster is within
    `max_distance_km` of at least one other point in that cluster.

    Returns
    -------
    pandas.DataFrame
        Original dataframe with two extra columns:
        - 'cluster_id' : integer label (-1 means noise if min_samples > 1)
        - 'cluster_centroid' : tuple (lat, lon) of the cluster’s centroid
    """
    # 1. Prepare coordinates in radians -------------------------------
    coords_deg = df[[lat_col, lon_col]].to_numpy()
    coords_rad = np.radians(coords_deg)

    # 2. Convert the distance threshold (km) to *radians* -------------
    earth_radius_km = 6_371.0088  # mean Earth radius (WGS-84)
    eps_rad = max_distance_km / earth_radius_km

    # 3. Run DBSCAN with haversine metric -----------------------------
    db = DBSCAN(
        eps=eps_rad,
        min_samples=1,  # every point belongs to a cluster
        metric="haversine",
        algorithm="ball_tree"
    ).fit(coords_rad)

    labels = db.labels_
    df = df.copy()
    df["cluster_id"] = labels

    # 4. Compute simple centroid for each cluster ---------------------
    centroids = (
        df.groupby("cluster_id")[[lat_col, lon_col]]
        .mean()
        .rename(columns={lat_col: "centroid_lat", lon_col: "centroid_lon"})
    )
    # Map centroid tuple back onto each row
    df["cluster_centroid"] = df["cluster_id"].map(
        centroids.apply(lambda r: (r.centroid_lat, r.centroid_lon), axis=1)
    )

    return df


def get_eov_info(alias_name):

    def handle_(item, parent_ = None):
        if isinstance(item, dict):
            for k, v in item.items():
                if k == alias_name:
                    return parent_ if parent_ is not None else k
                r_ = handle_(v, k)
                if r_ is not None:
                    return r_
        elif isinstance(item, list):
            if alias_name in item:
                return parent_

    r_ = handle_(nomenc_)
    return r_

def get_qaqc_config(eov_std_name):
    def handle__(qc_dict, parent_ = None):
        if not isinstance(qc_dict, dict):
            return None
        if eov_std_name in qc_dict.keys():
            return qc_dict[eov_std_name]
        for k, v in qc_dict.items():
            if k == eov_std_name:
                return v
            else:
                r_ = handle__(v, k)
                if r_ is not None:
                    return r_

        return None

    return handle__(qaqc_conf)


def check_unit_similarity(unit_1, unit_2):
    for k, v in unit_json.items():
        for lst in v:
            if (unit_1 in lst) and (unit_2 in lst):
                return True
    return False

node_unit_asso_, unit_node_asso = get_node_unit_association()
u2c_ = dict_inv(unit_json)






