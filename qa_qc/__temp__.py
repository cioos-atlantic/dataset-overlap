import pandas as pd

from qa_qc.qaqc_utils_ import erdap_dict_

def handle_cus_dic(dict_obj, dataset_name = None):
    if isinstance(dict_obj, dict):
        if "NC_GLOBAL" in dict_obj.keys():
            dataset_name = dict_obj["NC_GLOBAL"]["title"][1]+"\t "+dict_obj["NC_GLOBAL"]['institution'][1]
        for k , v in dict_obj.items():
            if "qc" in k.lower():
                print(f" {dataset_name}")
                return dataset_name
            r_ = handle_cus_dic(v, dataset_name)
            if r_ is not None:
                return r_
        return None
    elif isinstance(dict_obj, list) or isinstance(dict_obj, tuple):
        for it in dict_obj:
            if "quality flag" in it.lower():
                print(f"{dataset_name}")
                return dataset_name

    return None

def extract_quality_columns(diction, parent_ = None):
    lst_ = []
    if isinstance(diction, dict):
        for k, v in diction.items():
            if k == "NC_GLOBAL":
                continue
            if "q" in k.lower():
                lst_.append(diction[k]['attr']['long_name'][1])
            else:
                r_ = extract_quality_columns(v, k)
                lst_.extend(r_)
    elif isinstance(diction, list) or isinstance(diction, tuple):
        for it in diction:
            if "quality flag" in it.lower():
                lst_.append(parent_)


    return lst_

if __name__ == '__main__':
   a = 10

    # for dt_set_ in erdap_dict_:
    #     ds_name_ = handle_cus_dic(dt_set_)
        # if ds_name_ is not None:
        #     lst_of_cols_ = extract_quality_columns(dt_set_)
        #     for it in lst_of_cols_:
        #         print(f"\t\t------  {it}")

    # df_ = pd.read_csv("2024-10-24_cmar_water_quality_thresholds.csv")
    # grouped_ = df_.groupby(by=["qc_test", "variable"])
    # for grp, chunk in grouped_:
    #     unq_ = chunk['threshold'].unique()
    #     for vl in unq_:
    #         if "min" in vl:
    #             threshold_ = chunk[chunk['threshold'] == vl]['threshold_value'].min()
    #         else:
    #             threshold_ = chunk[chunk['threshold'] == vl]['threshold_value'].max()
    #
    #         print(f"{grp[0]} \t {grp[1]} \t {vl} \t {threshold_}")