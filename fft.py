import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import spectrogram
import csv
import os

from utils.dataset_preprocessing import df_normalization_nonlinear, df_normalization
from utils.plot_greyscale import plot_greyscale_for_singledf, plot_trace_heatmap_return_fig




#参数配置
fs = 20000000  # 采样率，需与 USRP 保持一致
fc = 30000000  #中心频率 (Hz)
iq_data_file="iq_data_1655.bin"
nfft =1024


iqfilename=iq_data_file.split(".bin")[0].split("\\")[-1].split("/")[-1]
print(iqfilename)
#读取IQ数据
iq_data = np.fromfile(iq_data_file, dtype=np.complex64)

#生成spectrogram
noverlap = nfft // 2
f_raw, t, Sxx = spectrogram(iq_data, fs=fs, nperseg=nfft, noverlap=noverlap, mode='magnitude', scaling='spectrum')

#频率轴对齐
Sxx = np.fft.fftshift(Sxx, axes=0)
f_baseband = np.fft.fftshift(f_raw)
f_rf = (f_baseband + fc)
Sxx_dB = 20 * np.log10(Sxx + 1e-10)

#切分保存
num_freq_bins = len(f_rf)
#这里希望每一个csv文件的行数不能太多，最多为列数的5倍吧（不然太细长了）
# 当超出行数时就截断成一个新的文件
max_rows_per_file = num_freq_bins * 1
num_total_rows = len(t)
num_files = (num_total_rows + max_rows_per_file - 1) // max_rows_per_file  # 向上取整

#output_dir = "output_csv"
dir_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
output_dir = f"output/{dir_name}"

os.makedirs(os.path.join(output_dir, "csv"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "img"), exist_ok=True)

for i in range(num_files):
    start_idx = i * max_rows_per_file
    end_idx = min((i + 1) * max_rows_per_file, num_total_rows)

    filename = os.path.join(output_dir, f"csv/{iqfilename}_spectrogram_{i}.csv")
    print(f"writing {filename}")
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['time'] + [f'{freq}' for freq in f_rf]
        writer.writerow(header)
        for j in range(start_idx, end_idx):
            writer.writerow([f'{t[j]:.6f}'] + list(Sxx_dB[:, j]))

    df=pd.read_csv(os.path.join(output_dir, f"csv/{iqfilename}_spectrogram_{i}.csv"))
    df=df_normalization_nonlinear(df)
    plot_greyscale_for_singledf(df,image_name=os.path.join(output_dir, f"img/{iqfilename}_spectrogram_{i}.png"))

print(f"保存完毕，共生成 {num_files} 个文件。")
