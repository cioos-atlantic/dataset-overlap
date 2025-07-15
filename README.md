This repository contains several packages

# download
This package contains code to download metadata from CKAN and Erddap.


> `__dowload_ckan_catalog.py` download metadata from CKAN and generates `national_alldataset_das_dict_NEW.pkl` 

> `__download_erdapp_data_lat_lon_time.py` saves .csv file for each dataset with its time, longitude, and latitude.

> `__download_erddap_metadata.py` downloades the metadata from erddap and generated `alldataset_das_dict_NEW.pkl`

# Dataset Similarity
This package analysis the metadata of all dataset and find the similarity between datasets in terms of attributes, time window, and spatial coverage. 



# qa_qc
