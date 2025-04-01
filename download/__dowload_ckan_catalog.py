import json
import pickle
import time

from requests_html import HTMLSession
from tqdm import tqdm

if __name__ == '__main__':
    base_url = "https://catalogue.cioos.ca/api/3/action/"
    ckan_url_ = f"{base_url}package_list"
    session = HTMLSession()
    # Send an HTTP GET request to the URL
    response = session.get(ckan_url_)
    # Render JavaScript if needed (optional)
    # response.html.render()
    json_list_links = response.json()
    all_elements = [r_id for r_id in json_list_links['result']]
    lst_of_das_dict = []
    for id__ in tqdm(all_elements):
        new_requ_ = f"{base_url}package_show?id={id__}"
        das_response = session.get(new_requ_)
        ds_info_ = das_response.json()
        lst_of_das_dict.append(ds_info_['result'])
        time.sleep(2)

    pickle.dump(lst_of_das_dict, open("../res/national_alldataset_das_dict_NEW.pkl", "wb"))
    print()



