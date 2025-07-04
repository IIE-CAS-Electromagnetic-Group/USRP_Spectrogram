import datetime
import os
import csv

import numpy as np
import pandas as pd
from scipy.signal import get_window

from utils.dataset_preprocessing import df_normalization_nonlinear
from utils.plot_greyscale import plot_greyscale_for_singledf
from utils.visual_spectrum_single_file import plot_trace_heatmap

# ===== 参数配置 =====
fs = 20_000_000    # 采样率
fc = 30_000_000    # 中心频率
iq_data_file = "F:\dianji\iq_data.bin"


nfft = 1024
noverlap = nfft // 2
window_name = "hann"
chunk_samples = 50_000_000          # 每块读取多少 complex 样本（≈400 MB）
dtype = np.complex64

# =====🤪 初始化输出目录 =====
iqfilename = os.path.basename(iq_data_file).split(".bin")[0]
dir_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
output_dir = f"F:/usrp_spectrum/output/{dir_name}"
os.makedirs(os.path.join(output_dir, "csv"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "img"), exist_ok=True)

# ===== 频率轴（一次性预计算）=====
# 与原脚本保持一致：先按 baseband，再 fftshift，再加上 fc 成 RF 频率
f_base = np.fft.fftshift(np.fft.fftfreq(nfft, d=1/fs))
f_rf = f_base + fc
freq_header = ["time"] + [f"{f:.6f}" for f in f_rf]

# ===== STFT 工具 =====
hop = nfft - noverlap
win = get_window(window_name, nfft, fftbins=True).astype(np.float32)

def stft_frames(x: np.ndarray):
    """将一维复信号切帧（有重叠）并返回 [n_frames, nfft] 矩阵视图。"""
    if x.size < nfft:
        return np.empty((0, nfft), dtype=np.complex64)
    n_frames = 1 + (x.size - nfft) // hop
    # 用 stride 构造重叠帧视图
    s0 = x.strides[0]
    return np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, nfft),
        strides=(hop * s0, s0),
        writeable=False,
    )

def frames_to_spectrum(frames: np.ndarray):
    """逐帧加窗 FFT，返回幅度谱并 fftshift 到与原脚本相同的频率顺序。"""
    if frames.shape[0] == 0:
        return np.empty((nfft, 0), dtype=np.float32)
    # 加窗
    frames_w = frames * win[None, :]
    # 直接做全频 FFT，然后与原代码一致地在频率维 fftshift
    X = np.fft.fft(frames_w, n=nfft, axis=1)
    mag = np.abs(X).astype(np.float32).T  # [nfft, n_frames]
    mag = np.fft.fftshift(mag, axes=0)
    # 转 dB（与原公式一致）
    Sxx_dB = 20.0 * np.log10(mag + 1e-10)
    return Sxx_dB

# ===== 主流程：memmap + 分块流式计算 =====
print(iqfilename)
mm = np.memmap(iq_data_file, dtype=dtype, mode="r")
N = mm.size

# 逐块迭代；块之间保留 last_tail 以保证帧跨块连续
last_tail = np.zeros(0, dtype=dtype)
global_frame_idx = 0      # 全局帧计数，用于时间轴
file_index = 0            # 输出 CSV 的编号
rows_in_current_file = 0  # 当前 CSV 已写的行数

# 根据“最多为列数的 5 倍行数”的策略切分文件
max_rows_per_file = len(f_rf) * 5

# 打开第一个 CSV
def new_csv_writer(idx):
    fn = os.path.join(output_dir, f"csv/{iqfilename}_spectrogram_{idx}.csv")
    f = open(fn, "w", newline="")
    w = csv.writer(f)
    w.writerow(freq_header)
    print(f"writing {fn}")
    return f, w, fn

csv_fh, csv_writer, cur_csv_path = new_csv_writer(file_index)

# 时间刻度：每一帧中心时间 = (frame_start + nfft/2) / fs
while global_frame_idx * hop < N:
    # 读取一个数据块并与上一个 tail 拼接
    start = global_frame_idx * hop
    # 读多一点，确保本块内能容纳尽量多的完整帧
    stop = min(N, start + chunk_samples + nfft)  # +nfft 只是上限冗余
    buf = np.asarray(mm[start:stop])
    if last_tail.size:
        buf = np.concatenate([last_tail, buf])

    # 切帧并计算谱
    frames = stft_frames(buf)
    if frames.shape[0] == 0:
        break
    Sxx_dB = frames_to_spectrum(frames)

    # 计算这些帧的起始样本索引（相对全局信号）
    # buf 的第 0 帧对应全局 start - last_tail.size
    base_start = start - last_tail.size
    frame_starts = base_start + np.arange(frames.shape[0]) * hop
    # 帧中心时间
    times = (frame_starts + nfft / 2) / fs

    # 逐帧写入 CSV，按文件行数阈值切换新文件
    for k in range(frames.shape[0]):
        if rows_in_current_file >= max_rows_per_file:
            csv_fh.close()
            # 后处理：归一化与画图
            df = pd.read_csv(cur_csv_path)
            df = df_normalization_nonlinear(df)
            plot_greyscale_for_singledf(df, image_name=os.path.join(output_dir, f"img/{os.path.basename(cur_csv_path)[:-4]}.png"))
            plot_trace_heatmap(cur_csv_path, cur_csv_path.replace(".csv", ".html"))
            # 新文件
            file_index += 1
            rows_in_current_file = 0
            csv_fh, csv_writer, cur_csv_path = new_csv_writer(file_index)

        row = [f"{times[k]:.6f}"] + Sxx_dB[:, k].tolist()
        csv_writer.writerow(row)
        rows_in_current_file += 1
        global_frame_idx += 1

    # 计算下一个块的 tail：保留最后 (nfft - hop) + (hop-1) = nfft-1 个点足够了，
    # 实际保留 nfft-1 能覆盖“下一块拼接后”的第一帧对齐
    keep = min(buf.size, nfft - 1)
    last_tail = buf[-keep:].copy()

# 关闭最后一个 CSV 并做对应图像和 HTML
csv_fh.close()
df = pd.read_csv(cur_csv_path)
df = df_normalization_nonlinear(df)
plot_greyscale_for_singledf(df, image_name=os.path.join(output_dir, f"img/{os.path.basename(cur_csv_path)[:-4]}.png"))
#plot_trace_heatmap(cur_csv_path, cur_csv_path.replace(".csv", ".html"))






print(f"保存完毕。CSV 起始目录：{os.path.join(output_dir, 'csv')}")
