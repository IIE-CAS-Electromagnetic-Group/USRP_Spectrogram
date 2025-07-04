import numpy as np
from gnuradio import gr
import time


# --- 各调制函数 ---
def mod_BPSK(bits, freq, t):
    return np.exp(1j * (2 * np.pi * freq * t + np.pi * bits))


def mod_QPSK(bits, freq, t):
    # 保证比特数为偶数
    n_sym = len(bits) // 2
    bits = bits[:n_sym * 2]
    idx = bits[0::2] * 2 + bits[1::2]
    phases = np.pi / 4 + np.pi / 2 * idx
    # 生成每个符号的复数值
    sym = np.exp(1j * phases)
    # 重复符号以匹配采样点数量
    out = np.repeat(sym, int(np.ceil(len(t) / len(sym))))[:len(t)]
    return out * np.exp(1j * 2 * np.pi * freq * t)


def mod_GFSK(freq, t, bt=0.35):
    N = len(t)
    data = np.random.choice([-1, 1], N)
    samples_per_bit = max(2, int(len(t) / N))
    gauss = np.exp(-0.5 * (np.linspace(-2, 2, samples_per_bit * 2) ** 2) / bt ** 2)
    phase_dev = np.convolve(data, gauss, mode="same")
    phase = np.cumsum(phase_dev) * (np.pi / samples_per_bit)
    return np.exp(1j * (2 * np.pi * freq * t + phase))


def mod_16QAM(bits, freq, t):
    # 4 bit 一个符号
    n_sym = len(bits) // 4
    bits = bits[:n_sym * 4]
    b = bits.reshape(-1, 4)
    mI = (2 * b[:, 0] + b[:, 1]) * 2 - 3
    mQ = (2 * b[:, 2] + b[:, 3]) * 2 - 3
    sym = (mI + 1j * mQ) / np.sqrt(10)
    out = np.repeat(sym, int(np.ceil(len(t) / len(sym))))[:len(t)]
    return out * np.exp(1j * 2 * np.pi * freq * t)


def mod_2FSK(bits, freq, t, df=5e3):
    """二进制频移键控（2-FSK）"""
    # bits=0 → freq - df, bits=1 → freq + df
    freq_inst = freq + (2 * bits - 1) * df
    phase = np.cumsum(2 * np.pi * freq_inst / len(freq_inst))
    return np.exp(1j * phase)


def mod_OFDM(bits, freq, t, n_subcarriers=8, sub_spacing=1e3):
    """简化版 OFDM：多载波正交叠加"""
    # 生成多个子载波信号叠加
    carriers = []
    for k in range(n_subcarriers):
        sub_freq = freq + (k - n_subcarriers / 2) * sub_spacing
        phase = 2 * np.pi * sub_freq * t + np.pi * bits[k % len(bits)]
        carriers.append(np.exp(1j * phase))
    sig = np.sum(carriers, axis=0) / n_subcarriers
    return sig


# -------- 主 Block --------
class blk(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(self,
                               name="Simple Scheduler",
                               in_sig=None,
                               out_sig=[np.complex64]
                               )
        self.samp_rate = 1e6
        self.freq_carrier = 1e6
        self.last_switch = time.time()
        self.state = "GFSK"  # 当前模式

    def work(self, input_items, output_items):
        out = output_items[0]
        N = len(out)
        t = np.arange(N) / self.samp_rate

        # --- 生成比特序列 ---
        symbol_rate = 1e3
        samples_per_symbol = max(1, int(self.samp_rate / symbol_rate))
        num_symbols = int(np.ceil(N / samples_per_symbol))
        bits = np.random.randint(0, 2, num_symbols)
        bits = np.repeat(bits, samples_per_symbol)[:N]

        # --- 周期计时（毫秒）---
        elapsed = (time.time() - self.last_switch) * 1000
        frame_period = 100.0  # 100 ms 周期

        # -------- 段落判断 --------
        if elapsed < 3:  # 同步/前导
            out[:] = mod_BPSK(bits, 0, t)

        elif elapsed < 10:  # 信标
            out[:] = mod_GFSK(+5e4, t, bt=0.15)

        elif elapsed < 60:  # 业务数据
            # 可切换 QPSK / 16QAM
            if np.random.rand() < 0.5:
                out[:] = mod_QPSK(bits, +2e5, t)
            else:
                out[:] = mod_16QAM(bits, +2e5, t)

        elif elapsed < 65:  # 跳频控制
            f = np.random.choice([+1e5, -1e5, +3e5, -3e5, +5e5, -5e5])
            out[:] = mod_GFSK(f, t, bt=0.2)

        elif elapsed < 75:  # 应答窗口（可置静默）
            if np.random.rand() < 0.7:  # 70 % 有应答
                out[:] = mod_BPSK(bits, 0, t)
            else:
                out[:] = 0

        elif elapsed < 100:  # 空口
            out[:] = 0

        else:  # 周期重启
            self.last_switch = time.time()
            out[:] = 0

        return len(out)



