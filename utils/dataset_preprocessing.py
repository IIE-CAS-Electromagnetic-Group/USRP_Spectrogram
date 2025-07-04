'''数据集预处理
原始频谱 df 是功率谱密度（PSD），比如单位是 dB（对数能量值），
常见数值范围可能在 [-120, -20] 之间，很多点是负值且幅度跨度大。

这种原始数据直接送入CNN去做特征提取有几个问题：
数值跨度大（比如 [-120, -20]）——导致梯度更新不稳定，优化困难
分布偏斜（噪声密集区 vs 信号稀疏区）——模型容易偏向低能量区域，忽略稀有信号模式
负值存在——对 ReLU（只保留正值）不友好，易造成特征损失
'''
import math

import pandas as pd
from utils.global_features_extract import generate_signal_anchors
import numpy as np

def df_normalization(df_origin):
    '''将df归一化，线性映射到 [0, 1]'''
    # 确定底噪
    all_values = df_origin.values.flatten()
    background_noise = pd.Series(all_values).median()

    signal_anchors = generate_signal_anchors(df_origin, num_anchors=10, top_k=50, anchor_size=10)
    all_signal_power = 0
    for signal_anchor in signal_anchors:
        # 提取锚框区域
        cf, ct, w, h = signal_anchor
        f_min = int(cf - w / 2)
        f_max = int(cf + w / 2)
        t_min = int(ct - h / 2)
        t_max = int(ct + h / 2)

        # 边界检查
        f_min = max(0, f_min)
        f_max = min(df_origin.shape[1], f_max)
        t_min = max(0, t_min)
        t_max = min(df_origin.shape[0], t_max)
        arr_anchor = df_origin.values[t_min:t_max + 1, f_min:f_max + 1]
        all_signal_power += np.mean(arr_anchor)

    #signal_max = all_signal_power / len(signal_anchors)

    signal_max = df_origin.values.max()

    powerwidth = signal_max - background_noise

    # 使用 Pandas 的向量化操作来设置底噪
    df_clipped = df_origin.clip(lower=background_noise,upper=signal_max)

    df_new = (df_clipped - background_noise) / powerwidth
    return df_new

def df_normalization_nonlinear(df_origin):
    '''将df归一化，非线性映射到 [0, 1]
    使用 median + MAD 代替 mean + std，增强鲁棒性'''

    all_values = df_origin.values.flatten()
    s = pd.Series(all_values)

    # === 计算底噪参考值：中位数 ===
    background_noise = s.median()

    # === 计算波动范围：1.4826 × MAD（等价于标准差的稳健替代） ===
    mad_value = np.median(np.abs(s - background_noise))
    robust_std = 1.4826 * mad_value

    # === 信号参考值 ===
    signal_threshold = background_noise + 3 * robust_std
    # 对于大于 signal_threshold 的频谱点，我们基本上可以把它视作信号

    powerwidth = (signal_threshold - background_noise) * 2

    # === 线性压缩映射：底噪→-5，信号参考→0 ===
    df_tmp = 10 * (df_origin - background_noise) / powerwidth + (-5)

    # === 非线性 Sigmoid 映射到(0,1)区间 ===
    df_new = 1 / (1 + np.exp(-df_tmp))

    #print(f"background_noise{background_noise}")
    #print(f"robust_std{robust_std}")
    #print(f"old_background_noise:{s[(s <= s.quantile(0.9)) & (s >= s.quantile(0.001))].mean()}")
    #print(f"old_std:{s[(s <= s.quantile(0.8)) & (s >= s.quantile(0.01))].std()}")

    return df_new


def df_normalization_phase(df_origin):
    '''将df归一化，[-pi,+pi]线性映射到 [0, 1]'''
    all_values = df_origin.values.flatten()
    value_min = pd.Series(all_values).min()
    value_max = pd.Series(all_values).max()

    # 使用 Pandas 的向量化操作来设置底噪

    df_new = (df_origin-value_min) / (value_max-value_min)
    return df_new
