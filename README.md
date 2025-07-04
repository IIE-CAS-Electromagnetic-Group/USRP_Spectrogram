## usrp采集信号
https://www.cnblogs.com/soaring27221/p/18935352

配置好GNU和USRP后，打开GNU Radio

```bash
gnuradio-companion
```
这里提供了一个grc脚本，把usrp_spectrum.grc丢进虚拟机，用GNU打开即可。


## 转化

在USRP上采集到iq时域信号后保存下来

将时域信号转化为频谱图

在fft.py里主要是这几个参数要注意一下：
```python
fs = 30000000  # 采样率，需要与 USRP 保持一致（也就是samp_rate）
fc = 2415000000  #中心频率 (Hz)
iq_data_file='iq_data.bin'#保存下来的iq bin文件路径
nfft = 1024#决定了频率分辨率，也影响时间分辨率
```
运行fft.py后，频谱数据保存到output_csv目录下。

## 可视化频谱

跑一下utils/plot_greyscale.py，把可视化的频谱图片保存在imgs目录下。

> 整个流程里最麻烦的就是最初的USRP的环境配置，别的环节倒没那么多坑。