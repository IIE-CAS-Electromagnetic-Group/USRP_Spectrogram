import os
import pandas as pd

from utils.dataset_preprocessing import df_normalization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_flattened_power_density(df, bins=256, log_y=False, show=True, title="Flattened Power Spectrum Density"):
    """
    将频谱功率矩阵展平并绘制功率谱密度折线图。

    参数:
        df (DataFrame or ndarray): 频谱功率矩阵（单位 dB）
        bins (int): 分箱数，决定横轴分辨率（默认256）
        log_y (bool): 是否对纵轴取log显示
        show (bool): 是否立即显示图
        title (str): 图标题
    """
    if isinstance(df, pd.DataFrame):
        data = df.values.flatten()
    else:
        data = np.array(df).flatten()

    # 分箱统计
    hist, bin_edges = np.histogram(data, bins=bins, density=False)

    # 横轴为每个 bin 中心
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 绘图
    plt.figure(figsize=(8, 4))
    plt.plot(bin_centers, hist, linestyle='-', marker='', color='tab:blue')
    plt.title(title)
    plt.xlabel("Power (dB)")
    plt.ylabel("Frequency Count")
    if log_y:
        plt.yscale("log")
    plt.grid(True)
    if show:
        plt.show()
    return bin_centers, hist


if __name__ == "__main__":
    print("start test...")
    csv_dir = "..\output_csv"

    csv_files = []
    for f in os.listdir(csv_dir):
        if f.endswith(".csv"):
            csv_files.append(os.path.join(csv_dir, f))

    for csv_file in csv_files:
        filename = os.path.basename(csv_file).split('.csv')[0]
        df = pd.read_csv(csv_file)
        df = df.iloc[:, 1:]
        #df=df_normalization(df)
        #anchors_list=__generate_initial_anchors(df,num_anchors=20)
        #plot_greyscale_for_singledf_with_anthor_and_score(df,[],image_name=f"../imgs/{filename}.png")
        plot_flattened_power_density(df=df, bins=256, log_y=False, show=True, title=f"{filename} Flattened Power Spectrum Density")
