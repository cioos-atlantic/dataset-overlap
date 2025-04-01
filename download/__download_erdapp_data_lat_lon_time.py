
import random
import json
import time
import os
from requests_html import HTMLSession
import re
import pickle
from tqdm import tqdm
import sys
import csv
import urllib.request

def download_csv(url, save_path, title=None):

        urllib.request.urlretrieve(url, save_path)
        # print(f"CSV file successfully downloaded and saved to '{save_path}'.")


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
    erddap_data_dir_= "erddap national-data"
    # base_url = "https://cioosatlantic.ca/erddap/tabledap/index.html?page=1&itemsPerPage=1000"
    # base_url  = "https://catalogue.cioosatlantic.ca/api/3/action/resource_search?query=name:erddap"
    base_url = "https://catalogue.cioos.ca/api/3/action/resource_search?query=name:erddap"
    session = HTMLSession()
    # Send an HTTP GET request to the URL
    response = session.get(base_url)
    # Render JavaScript if needed (optional)
    # response.html.render()
    json_list_links = json.loads(response.text)
    all_elements = [r_id['url'] for r_id in json_list_links['result']['results']]
    list_of_dictionaries_ = []
    pbar_ = tqdm(all_elements)
    for elem in pbar_:
        try:
            new_requ_ = elem
            if not new_requ_.endswith(".html"):
                new_requ_ = (new_requ_+".html")

            csv_url_copy = new_requ_
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
            title_ = ds_profile_dictionary_['NC_GLOBAL']['title'][1].replace(","," ").replace(":","").replace("/"," ").replace("|","")
            csv_filepath = f"./{erddap_data_dir_}/{title_}.csv"
            if os.path.exists(csv_filepath):
                pbar_.set_description(f"Skipping [{title_}]" )
                continue

            pbar_.set_description(f"downloading..  [{title_}]")

            session = HTMLSession()
            if ".html" in csv_url_copy:
                csv_url_copy = csv_url_copy.replace(".html", ".csv?time%2Clatitude%2Clongitude")
            download_csv(csv_url_copy, csv_filepath, title=title_)
            # das_response = session.get(csv_url_copy)
            # if das_response.reason == '404':
            #     print(f"===> URL CSV not found: {elem}")
            #     continue
            # decoded_content = das_response.content
            # csv_file = open(csv_filepath, 'wb')
            # csv_file.write(decoded_content)
            # csv_file.close()
            time.sleep(random.randint(2, 6))
        except urllib.error.HTTPError as http_err:
            pbar_.set_description(f"Error..  [{title_}]")
            print(f" \n HTTP error occurred: {http_err} \n [{elem}] -> [{title_}]")  # HTTP error
        except urllib.error.URLError as url_err:
            pbar_.set_description(f"Error..  [{title_}]")
            print(f"\n URL error occurred: {url_err} \n [{elem}] -> [{title_}] ")  # URL error
        except Exception as err:
            pbar_.set_description(f"Error..  [{title_}]")
            print(f"\n An error occurred: {err} \n [{elem}] -> [{title_}] ")  # Other errors
        except:

            # sys.stdout.write(f" \n Error Dealing with {elem}")
            pbar_.set_description(f"Error..  [{title_}]")
            print(f" \n Error Dealing with [{title_}] [{elem}] [{csv_url_copy}]")
            session = HTMLSession()


