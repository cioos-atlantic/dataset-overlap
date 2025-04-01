import json
import time

from requests_html import HTMLSession
import pickle
from tqdm import tqdm

'''
This file will download t
'''

def parse_json(json_object_):
    ds_profile_dict = {}
    lst_of_rows_ = json_object_['table']['rows']

    last_ = None
    for rtype, v_name, att_name, dtype, val_  in lst_of_rows_:
        if rtype == "variable":
            if v_name not in ds_profile_dict:
                ds_profile_dict[v_name] = {"dtype": dtype, "attr": {}}
            last_ = (rtype, v_name, att_name, dtype, val_)
        elif rtype == 'attribute':
            if v_name not in ds_profile_dict:
                ds_profile_dict[v_name] = {}
            else:
                if (last_ is not None) and last_[0] == 'variable':
                    assert att_name not in ds_profile_dict[v_name]["attr"], "error unknown"
                    ds_profile_dict[v_name]["attr"][att_name] = ( dtype, val_)
                else:
                    ds_profile_dict[v_name][att_name] = (dtype, val_)
        else:
            print("=====> Row type Error (Not found)")
            exit(-1)

    return ds_profile_dict



if __name__ == '__main__':
    # base_url = "https://cioosatlantic.ca/erddap/tabledap/index.html?page=1&itemsPerPage=1000"
    base_url  = "https://catalogue.cioosatlantic.ca/api/3/action/resource_search?query=name:erddap"
    session = HTMLSession()
    # Send an HTTP GET request to the URL
    response = session.get(base_url)
    # Render JavaScript if needed (optional)
    # response.html.render()
    json_list_links = json.loads(response.text)
    all_elements = [r_id['url'] for r_id in json_list_links['result']['results']]
    list_of_dictionaries_ = []
    for elem in tqdm(all_elements):
        new_requ_ = elem
        if not new_requ_.endswith(".html"):
            new_requ_ = (new_requ_+".html")
        if "/erddap/info" not in new_requ_:
            new_requ_ = new_requ_.replace("/erddap/tabledap", "/erddap/info")
        if "index.html" in new_requ_:
            new_requ_ = new_requ_.replace("index.html", "index.json")
        if ".html" in new_requ_:
            new_requ_ = new_requ_.replace(".html", "/index.json")
        das_response = session.get(new_requ_)
        if das_response.reason == '404':
            print(f"URL not found: {elem}")
            continue
        json_obj_ = das_response.json()
        ds_profile_dictionary_ = parse_json(json_obj_)
        list_of_dictionaries_.append(ds_profile_dictionary_)
        time.sleep(1)

    print(f"Number of das files: {list_of_dictionaries_.__len__()}")
    pickle.dump(list_of_dictionaries_, open("./alldataset_das_dict_NEW.pkl", 'wb'))