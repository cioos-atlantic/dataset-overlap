import pickle
import networkx as nx
from pyvis.network import Network
from community import community_louvain
from qaqc_utils_ import gen_graph_, man_graph
from qa_qc.qaqc_utils_ import get_cat, u2c_, node_unit_asso_, create_pyviz_network_graph


#create a graph of this and then check

def check_cooccured(node1, node2, erdap_dict_, graph ):
    for dataset_ in erdap_dict_:
        for k, __val in dataset_.items():
            if k == "nc_global":
                continue
            long_name = __val['attr']['long_name'][1].lower() if 'long_name' in __val['attr'] else k
            std_name = __val['attr']['standard_name'][1].lower() if 'standard_name' in __val['attr'] else k
            g_l_ = [k, long_name, std_name]
            if (node1 in g_l_) or (node2 in g_l_):
                graph.add_edge(k, long_name)
                graph.add_edge(k, std_name)

            if (node1 in g_l_) and (node2 in g_l_):
                graph.add_edge(node1, node2, weight=1)
                return True


    return False

if __name__ == '__main__':

    eov = "density"
    lst_of_node_s = ["in situ density from lag corrected salinity", "sea_water_sigma_t", "sea_water_sigma_theta", "mass concentration of nitrate in sea water", "mass concentration of oxygen in sea water", "concentration_of_colored_dissolved_organic_matter_in_sea_water_expressed_as_equivalent_mass_fraction_of_quinine_sulfate_dihydrate", "mass_concentration_of_chlorophyll_in_sea_water", "sigmatheta", "sigteq01", "density_lag_correct", "mass_concentration_of_oxygen_in_sea_water", "dissolved_oxygen_uncorrected", "chlorophyll concentration by fluorometer", "cphlpr02", "mass_concentration_of_nitrate_in_sea_water", "satslb0045_nitrate_mg", "cphlpr01", "ccomd002", "chlorophyll a fluorescence", "sci_flbbcd_chlor_units", "mass_concentration_of_chlorophyll_a_in_sea_water", "concentration of coloured dissolved organic matter {cdom gelbstoff} per unit volume of the water body [dissolved plus reactive particulate phase] by fluorometry", "cdomzz02", "sigteqst", "cdomzz01", "concentration_of_chlorophyll_in_sea_water", "volume scattering function chlorophyll (695 nm, 110 deg)", "oxygen(mg/l)", "oxygen_mg_l", "sci_flbbbbv1_fl_scaled", "sea water density", "density", "in situ density", "sigma-t", "sea_water_density", "fluorescence", "fluorescence, wet labs cdom"]


    erdap_dict_ = pickle.load(open("../res/alldataset_das_dict_NEW.pkl", 'rb'))
    t_list = []
    for dataset_ in erdap_dict_:
        t_dict = {}
        for k, v in dataset_.items():
            t_dict[k.lower()] = v
        t_list.append(t_dict)
    erdap_dict_ = t_list

    g_ = nx.Graph()
    for i1 in range(0,len(lst_of_node_s)):
        for i2 in range(i1+1, len(lst_of_node_s)):
            n1, n2 = lst_of_node_s[i1], lst_of_node_s[i2]
            t_ = check_cooccured(n1, n2, erdap_dict_, g_)
            if t_:
                print(f"[{n1}]  -> [{n2}]")

    list_of_accepted_nodes_ = [nd for nd in g_]
    for nod in list_of_accepted_nodes_:
        if (nod not in gen_graph_) and (nod not in man_graph):
            try:
                cats_ = set(get_cat(node_unit_asso_[nod], u2c_))
                if (len(cats_) == 1) and eov in cats_:
                    continue
            except:
                pass
            g_.remove_node(nod)
            print(f"Removing:  {nod}")


    pickle.dump(g_, open(f"../res/Eovs_Graphs/{eov}.pkl", 'wb'))
    create_pyviz_network_graph(g_, f"../res/Eovs_Graphs/{eov}.html")

    print("..Done..")