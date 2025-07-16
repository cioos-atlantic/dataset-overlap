This repository contains several packages:

# download
This package contains code to download metadata from CKAN and Erddap.


> `__dowload_ckan_catalog.py` download metadata from CKAN and generates `national_alldataset_das_dict_NEW.pkl` 

> `__download_erdapp_data_lat_lon_time.py` saves .csv file for each dataset with its time, longitude, and latitude.

> `__download_erddap_metadata.py` downloades the metadata from erddap and generated `alldataset_das_dict_NEW.pkl`

# Dataset Similarity
This package analysis the metadata of all dataset and find the similarity between datasets in terms of attributes, time window, and spatial coverage. The heatmap plot can be generated in .html file. 

Run any `__ploting*.py` then it will start a visualization server, click `http://127.0.0.1:PORT/` in the console which open a window in the browser to view the responsive heatmap.  

![](/res/heatmap_data_overlap.png)

# qa_qc
This package include QA/QC related code. Some highlights include:

- It contains code to generate line plots of dataset to see how values are changing over time. 

![](/res/Plotly_Sample.png)

- Unit Converter

- Extracting SUSPECT label window points from datasets

- Multi-layer perceptron Model

- Random Forest Model