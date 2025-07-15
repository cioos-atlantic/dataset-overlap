from qa_qc.ai_utils import get_file_names
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

def get_dirs(dir_n):
    datasets__ = []
    for entry in os.listdir(dir_n):
        folder_name = os.path.join(dir_n, entry)
        if os.path.isdir(folder_name):
            datasets__.append(entry)
    return datasets__

if __name__ == '__main__':


    eov_col_name = 'temperature'
    eov_flag_name = 'qc_flag_temperature'

    parent_ = "D:/CIOOS-Full-Data/chunking/"
    subdir_ = get_dirs(parent_)
    for ds__ in subdir_:
        # chunk_dir = os.path.join(parent_, "Annapolis/")
        chunk_dir = os.path.join(parent_, ds__,"")
        file_names = get_file_names(chunk_dir)
        with PdfPages(os.path.join(parent_, os.path.dirname(chunk_dir)+".pdf")) as pdf_:
            for file_name in tqdm(file_names):
                print(f"Processing : [{file_name}]")
                df = pd.read_csv(file_name, usecols=['time', eov_flag_name, eov_col_name])
                df['time'] = pd.to_datetime(df['time'])

                # Step 2: Extract the values
                values = df[eov_col_name]

                # Step 3: Create bins with step size 0.5
                min_val = np.floor(values.min())
                max_val = np.ceil(values.max())
                bins = np.arange(min_val, max_val + 0.5, 0.5)  # +0.5 to include last bin

                # Step 4: Plot histogram
                plt.figure(figsize=(10, 6))
                plt.hist(values, bins=bins, edgecolor='black', align='left')
                plt.xlabel('Sensor Value (binned every 0.5)')
                plt.ylabel('Count')
                plt.title(file_name.replace(".csv",""))
                plt.grid(True)
                plt.xticks(bins, rotation=90)
                plt.tight_layout()
                save_path_ = os.path.join(parent_, os.path.basename(file_name.replace(".csv",".png")))
                print(save_path_)
                pdf_.savefig()
                plt.close()
            # plt.savefig(save_path_)
