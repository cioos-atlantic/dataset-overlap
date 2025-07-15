import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from qa_qc.qaqc_utils_ import get_eov_info


"""
This file create  a line plot in PDF from .csv file of dataset. 
The data is plotted month-wise in each page
"""

def is_file_bigsize(file_path, threshold_mb=100):
    ''' check if the size of file is bigger than given threshold'''
    if not isinstance(file_path, str) or not file_path:
        print("Error: Invalid filepath provided.")
        return None

    threshold_bytes = threshold_mb * 1024 * 1024
    file_size_bytes = os.stat(file_path).st_size
    file_size_mb = file_size_bytes / (1024 * 1024)  # Calculate size in MB for display
    print(f"File [{format_filename(file_path)}] size: {file_size_mb:.2f} MB")

    return file_size_bytes > threshold_bytes


def format_filename(file_path):
    fname_ = os.path.basename(file_path)
    if len(fname_) > 12:
        return fname_[0:4] + "**" + fname_[-5:]
    else:
        return fname_


def has_valid_unit(col_name, unit_df_):
    if col_name.lower() in ['time', 'longitude', 'latitude']:
        return None
    v_ = unit_df_.iloc[0][col_name]
    if pd.isna(v_):
        return None
    else:
        return v_


def grid_size(tot_col):
    i, j = 1, 1
    b = True
    while (i * j) < tot_col:
        if b:
            i += 1
        else:
            j += 1
        b = not b
    return i, j


if __name__ == '__main__':
    f_path = "../full data/SMA_bay_of_exploits_bab9_79bf_9fe6.csv"
    unit_df = pd.read_csv(f_path, nrows=1)

    if is_file_bigsize(f_path):
        rows_in_chunk_ = 10000
        chunk_reader = pd.read_csv(f_path, skiprows=[1], chunksize=rows_in_chunk_, parse_dates=['time'])
        for data_ in chunk_reader:
            pass  # Modify if you want to process big files chunk-wise
    else:
        data_ = pd.read_csv(f_path, skiprows=[1], parse_dates=['time'])
        data_ = data_.sort_values(by='time')

        valid_cols = [col for col in data_.columns if has_valid_unit(col, unit_df)]
        valid_cols = [vc for vc in valid_cols if get_eov_info(vc) is not None]

        if not valid_cols:
            print("No valid data columns with units found.")
            exit()

        data_.set_index('time', inplace=True)
        monthly_groups = data_[valid_cols].groupby(pd.Grouper(freq='M'))

        pdf_name = f_path + "_monthly_plots.pdf"
        with PdfPages(pdf_name) as pdf:
            for month, group in monthly_groups:
                if group.dropna(how='all').empty:
                    continue  # skip empty months

                nrows, ncols = grid_size(len(valid_cols))
                fig, axs = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
                fig.suptitle(f'Month: {month.strftime("%B %Y")}', fontsize=16)

                if len(valid_cols) == 1:
                    axs = [axs]

                axs = np.array(axs).reshape(nrows, ncols)

                x_ = (group.index.astype(np.int64) // 10**9)
                x_ = (x_ - x_.min()) / 60  # minutes from start of month

                for col, (i, j) in zip(valid_cols, [(i, j) for i in range(nrows) for j in range(ncols)]):
                    ax = axs[i, j]
                    ax.plot(x_, group[col], linewidth=0.2)
                    ax.set_title(col)
                    ax.set_xlabel("Minutes")
                    ax.set_ylabel(col)

                # Hide any unused subplots
                for k in range(len(valid_cols), nrows * ncols):
                    i, j = divmod(k, ncols)
                    axs[i, j].axis('off')

                plt.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig)
                plt.close(fig)

        print(f"Monthly plots saved to: {pdf_name}")
