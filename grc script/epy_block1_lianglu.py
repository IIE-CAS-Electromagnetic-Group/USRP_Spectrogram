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

        # --- 比特序列 ---
        symbol_rate = 1e3
        samples_per_symbol = max(1, int(self.samp_rate / symbol_rate))
        num_symbols = int(np.ceil(N / samples_per_symbol))
        bits = np.random.randint(0, 2, num_symbols)
        bits = np.repeat(bits, samples_per_symbol)[:N]

        elapsed = (time.time() - self.last_switch) * 1000  # 毫秒级

        # -------- 跳频信号参数 --------
        hop_period = 10.0  # 每 10 ms 跳一次
        hop_range = 2e6
        hop_step = 2e5
        if elapsed >= hop_period:
            f_list = np.arange(-hop_range, hop_range + hop_step, hop_step)
            self.freq_carrier = float(np.random.choice(f_list))
            self.last_switch = time.time()
            print(f"[Hop] GFSK carrier = {self.freq_carrier / 1e3:.1f} kHz")

        # -------- 跳频 GFSK 信号 --------
        sig_hop = mod_GFSK(self.freq_carrier, t, bt=0.1)

        # -------- 静态 QPSK 信号 --------
        static_freq = 0
        sig_static = mod_QPSK(bits, static_freq, t)

        # -------- 叠加两路信号 --------
        power_ratio = 0.5  # 控制相对功率（0.5 ≈ –6 dB）
        out[:] = sig_static + power_ratio * sig_hop
        return len(out)



