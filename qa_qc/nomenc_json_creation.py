import pickle
import json

import networkx as nx

from qa_qc.qaqc_utils_ import get_node_unit_association, unit_to_category, find_node_s_dataset_name, find_unit_variable_name,\
    find_unit_dataset_name, normalize, ext_keys_val_list, man_graph, gen_graph_, get_cat, node_unit_asso_, u2c_, unit_node_asso


def nd_unit_notin_unit_json(nd_unit_, u2c_):
    if isinstance(nd_unit_, list):
        for el in nd_unit_:
            if el not in u2c_:
                return True
        return False
    else:
        return nd_unit_ not in u2c_





if __name__ == '__main__':
    # nomc_g_ = pickle.load(open("../res/Nomenclature_updated_graph.pkl", 'rb'))

    eov = "density"


    node_unit_asso_keys_ = [normalize(nd) for nd in node_unit_asso_.keys()]
    nodes_ = [n.lower() for n in gen_graph_]
    nodes_of_unit = set([ n for n in find_unit_variable_name("1") ])
    nomc_g_ = nx.compose(gen_graph_,man_graph)

    eov_columns = []
    for nd in nomc_g_:
        nd = nd.lower()
        if nd not in node_unit_asso_:
            ds_name_ = find_node_s_dataset_name(nd)
            # print(f"[{nd}] NO Unit Found - Repo: {ds_name_}")
            continue
        nd_unit_ = node_unit_asso_[nd]

        if nd_unit_notin_unit_json(nd_unit_, u2c_):
            ds_name_ = find_node_s_dataset_name(nd)
            # print(f"[{nd_unit_}] Invalid Unit Found of [{nd}] - Repo [{ds_name_}]")
            continue

        cate_ = set(get_cat(nd_unit_, u2c_))
        if eov in cate_:
            if len(cate_) != 1:
                print(f"Warning:  [{nd}]  is used for {cate_}")
            eov_columns.append(nd)
    # 1. traverse each node from nomenclature graph
    # 2. check unit of each node
    # 3. check category of each node
    # 4. filter out temperature communities for now
    # 5. save it as temperature .json
    # 6. Associate CF naming convention
    # lst_of_total_units = ext_keys_val_list(c2u["temperature"])
    # lst_of_total_nodes = []
    # lst_of_upper_nodes_ = []
    # for unit__ in lst_of_total_units:
    #     lst_of_total_nodes.extend( unit_node_asso[unit__])
    #     lst_of_upper_nodes_.extend(find_nodename_by_unit(unit__))




    print(json.dumps(eov_columns))