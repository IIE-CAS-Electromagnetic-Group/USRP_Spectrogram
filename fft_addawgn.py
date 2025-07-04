import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import csv
import os


def add_awgn_given_existing_snr(iq_data, snr_orig_db, snr_target_db):
    # 计算原始 IQ 平均功率
    P_signal_plus_noise = np.mean(np.abs(iq_data)**2)

    # 原始 SNR 比值
    snr_orig_linear = 10**(snr_orig_db / 10)
    P_signal = P_signal_plus_noise * snr_orig_linear / (1 + snr_orig_linear)

    # 目标噪声功率计算
    snr_target_linear = 10**(snr_target_db / 10)
    P_noise_target = P_signal / snr_target_linear

    # 当前已有的噪声功率
    P_noise_current = P_signal_plus_noise - P_signal

    # 需要补加多少噪声
    P_noise_to_add = max(P_noise_target - P_noise_current, 0)
    print(f"当前SNR约为{snr_orig_db:.1f} dB，目标SNR={snr_target_db} dB，补加噪声功率={P_noise_to_add:.2e}")

    # 添加高斯白噪声
    noise = np.sqrt(P_noise_to_add / 2) * (np.random.randn(len(iq_data)) + 1j * np.random.randn(len(iq_data)))
    return iq_data + noise



#参数配置
fs = 30000000  # 采样率，需与 USRP 保持一致
fc = 2415000000  #中心频率 (Hz)
iq_data_file='iq_data.bin'
nfft = 1024

snr_orig_db=30
snr_target_db=-10



#读取IQ数据
iq_data = np.fromfile(iq_data_file, dtype=np.complex64)
iq_data = add_awgn_given_existing_snr(iq_data, snr_orig_db=snr_orig_db, snr_target_db=snr_target_db)




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
max_rows_per_file = num_freq_bins * 5
num_total_rows = len(t)
num_files = (num_total_rows + max_rows_per_file - 1) // max_rows_per_file  # 向上取整

output_dir = "output_csv"
os.makedirs(output_dir, exist_ok=True)

num_digit=len(str(num_files))

for i in range(num_files):
    start_idx = i * max_rows_per_file
    end_idx = min((i + 1) * max_rows_per_file, num_total_rows)


    filename = os.path.join(output_dir, f"spectrogram_awgn_{snr_target_db}_{str(i).zfill(num_digit)}.csv")
    print(f"writing {filename}")
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['time'] + [f'{freq}' for freq in f_rf]
        writer.writerow(header)
        for j in range(start_idx, end_idx):
            writer.writerow([f'{t[j]:.6f}'] + list(Sxx_dB[:, j]))

print(f"保存完毕，共生成 {num_files} 个文件。")
